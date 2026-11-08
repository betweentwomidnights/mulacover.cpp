#!/usr/bin/env python3
"""FP32 oracle for the experimental score clock; pinned upstream stays read-only.

The only model intervention is a float attention bias at the SDPA call inside
CrossAttentionRoPE. Original projections, RoPE, adaptors, backbone and local
decoder execute unchanged. This is an inference experiment, not stock upstream.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-window',action='store_true')
    p.add_argument('--width-beats',type=float,default=1)
    p.add_argument('--frames',type=int,default=3)
    p.add_argument('--bpms',type=float,nargs='+',default=[0,90,120,150])
    for name in ('upstream','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not 3<=a.frames<=8192 or not .0625<=a.width_beats<=16:raise ValueError('Invalid oracle dimensions')
    if a.output.exists():raise ValueError('Use a fresh oracle directory')
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip()
    if revision!=manifest['upstream_revision'] or dirty:raise ValueError('Expected clean pinned upstream')
    sys.path.insert(0,str(a.upstream/'src'))
    from mulacover import MuLaCover
    import mulacover.modeling as modeling
    torch.set_num_threads(4);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True);prepared={};preprocessing={}
    for key,item in manifest['preprocessing'].items():
        data=(a.prepared/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Prepared input hash mismatch')
        (a.output/item['file']).write_bytes(data);preprocessing[key]=item
        prepared[key[6:]]=torch.from_numpy(np.frombuffer(data,np.dtype(item['stored_dtype'])).reshape(item['shape']).copy()).cuda()
    prompt=prepared['tokens'].shape[1]
    if prompt+a.frames-1>8192:raise ValueError('Oracle exceeds backbone capacity')
    clock={};original_attention=modeling.CrossAttentionRoPE.forward
    original_sdpa=torch.nn.functional.scaled_dot_product_attention
    original_layer=modeling.CrossAttentionLayer.forward

    def layer(self,hidden_states,pianoroll,drum_pianoroll,chord,context_mask,previous_context,input_pos):
        if clock['bpm'] and clock['local_window']:
            center=4*clock['offset_beats']+max(0,int(input_pos[0,-1])-(prompt-1))*.08*clock['bpm']/15
            keys=torch.arange(context_mask.shape[-1],device=context_mask.device,dtype=torch.float64)
            distance=(keys-center).abs();context_mask=context_mask.clone()
            valid=(~context_mask[0]).nonzero().flatten();nearest=valid[distance[valid].argmin()]
            count=context_mask.shape[0]//2 if context_mask.shape[0]>1 else 1
            context_mask[:count]|=distance>4*clock['width_beats'];context_mask[:count,nearest]=False
        return original_layer(self,hidden_states,pianoroll,drum_pianoroll,chord,context_mask,previous_context,input_pos)

    def attention(self,x,context,mask,input_pos=None):
        if not clock['bpm']:return original_attention(self,x,context,mask,input_pos)
        # Independently form the prior in double precision, then round once to F32.
        frames=input_pos.to(torch.float64)-(prompt-1)
        valid=(~prepared['context_mask'][0]).nonzero().flatten()
        center=(4*clock['offset_beats']+frames*.08*clock['bpm']/15).clamp(valid[0].item(),valid[-1].item())
        keys=torch.arange(context.shape[1],dtype=torch.float64,device=x.device)
        bias=(-.5*((keys[None,None,:]-center[:,:,None])/(4*clock['width_beats']))**2).clamp_min(-80)
        bias=torch.where((frames<0)[:,:,None],0,bias).float()
        if x.shape[0]>1:bias[x.shape[0]//2:]=0
        def sdpa(q,k,v,attn_mask=None,**kwargs):
            additive=torch.zeros_like(attn_mask,dtype=q.dtype).masked_fill(~attn_mask,-float('inf'))
            return original_sdpa(q,k,v,attn_mask=additive+bias[:,None,:,:],**kwargs)
        torch.nn.functional.scaled_dot_product_attention=sdpa
        try:return original_attention(self,x,context,mask,input_pos)
        finally:torch.nn.functional.scaled_dot_product_attention=original_sdpa

    source=json.loads(Path('models/generator.json').read_text());cases=[]
    with torch.inference_mode():
        model,info=MuLaCover.from_pretrained(source['snapshot'],device_map='cuda',dtype=torch.float32,local_files_only=True,output_loading_info=True)
        if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Non-strict model load')
        model.eval().requires_grad_(False);model.setup_caches(2)
        if sum(isinstance(m,modeling.CrossAttentionRoPE) for m in model.modules())!=16:raise ValueError('Unexpected score attention layout')
        modeling.CrossAttentionRoPE.forward=attention
        modeling.CrossAttentionLayer.forward=layer
        original_sampler=modeling.sample_topk;original_cfg=model._sample_with_cfg
        base=torch.tensor([[0,8191,7,42,314,4096,123,777],[11,17,23,29,31,37,41,43],[60,64,67,72,4095,4097,8189,2]],device='cuda')
        teacher=base[torch.arange(a.frames,device='cuda')%3];checkpoints=[0,1,a.frames-1]
        for bpm in a.bpms:
            if bpm!=0 and not 20<=bpm<=400:raise ValueError('Invalid oracle BPM')
            clock=dict(bpm=bpm,width_beats=a.width_beats,offset_beats=0,local_window=int(a.local_window));name='clock-'+format(bpm,'g')
            root=a.output/name;root.mkdir();files={};call=[0];step=[0]
            def save(name,value):
                array=np.ascontiguousarray(value.detach().cpu().numpy());ext={'float32':'f32','int64':'i64'}[str(array.dtype)]
                data=array.tobytes();path=root/(name+'.'+ext);path.write_bytes(data)
                files[name]=dict(file=path.name,shape=list(array.shape),stored_dtype=str(array.dtype),source_dtype=str(value.dtype),sha256=hashlib.sha256(data).hexdigest())
            def sampler(logits,topk,temperature):
                q=call[0]%8;call[0]+=1
                if step[0] in checkpoints:save(f'frame{step[0]}.logits.{q}',logits)
                return teacher[step[0],q].reshape(1,1).to(torch.int32)
            def cfg(logits,*args):
                if step[0] in checkpoints:save(f'frame{step[0]}.raw_logits.{call[0]%8}',logits)
                return original_cfg(logits,*args)
            modeling.sample_topk=sampler;model._sample_with_cfg=cfg;model.reset_caches();current=dict(prepared)
            for f in range(a.frames):
                step[0]=f;sample=model.generate_frame(**current,temperature=1.,topk=250,cfg_scale=1.5,first_step=f==0)
                if not torch.equal(sample[0].long(),teacher[f]):raise ValueError('Teacher forcing failed')
                for group in ('backbone','decoder'):
                    for layer,block in enumerate(getattr(model,group).layers):
                        cache=block.attn.kv_cache;active=prompt+f if group=='backbone' else 8
                        for kind in ('k','v'):
                            tensor=getattr(cache,kind+'_cache')
                            if f in checkpoints:save(f'frame{f}.cache.{group}.{layer}.{kind}',tensor[:,:,:active])
                            if torch.count_nonzero(tensor[:,:,active:]):raise ValueError('Cache tail modified')
                tokens=torch.zeros(2,1,9,dtype=torch.long,device='cuda');tokens[:,:,:8]=sample[:,None,:]
                mask=torch.ones_like(tokens,dtype=torch.bool);mask[:,:,-1]=False
                current.update(tokens=tokens,tokens_mask=mask,input_pos=current['input_pos'][:,-1:]+1,qwen_indices=None,muq_indices=None,muq_embedding=None)
            save('teacher_frames',teacher)
            case=dict(name=name,guidance=1.5,frames=a.frames,batch=2,files=files,experimental_tempo=clock,checkpoints=checkpoints)
            (root/'case.json').write_text(json.dumps(case,indent=2)+'\n');cases.append(case)
            print(name,len(files),'oracle tensors',flush=True)
        modeling.sample_topk=original_sampler;model._sample_with_cfg=original_cfg;modeling.CrossAttentionRoPE.forward=original_attention;modeling.CrossAttentionLayer.forward=original_layer
    result={k:manifest[k] for k in ('upstream_revision','models','versions','generator_dtype','style_dtype')}
    result.update(preprocessing=preprocessing,cases=cases,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intervention='Gaussian conditional score-attention log prior, optionally local adaptor/key mask; stock upstream with no prior or local mask for clock-0')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
