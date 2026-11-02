#!/usr/bin/env python3
"""Upstream long-cache and coupled-sampling oracle, reusing frozen prepared inputs."""
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
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=32)
    p.add_argument('--prompt-frames',type=int,default=128);p.add_argument('--symbolic-frames',type=int,default=64)
    a=p.parse_args()
    if a.output.resolve()==a.prepared.resolve() or (a.output/'manifest.json').exists():raise ValueError('Use a fresh oracle directory')
    original=json.loads((a.prepared/'manifest.json').read_text())
    revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=original['upstream_revision']:raise ValueError('Upstream source differs from the frozen baseline')
    sys.path.insert(0,str(a.upstream/'src'))
    from mulacover import MuLaCover
    import mulacover.modeling as modeling
    torch.set_num_threads(4);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=True)
    root=a.output;files={}
    def load(item):
        data=(a.prepared/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Frozen input hash mismatch')
        return torch.from_numpy(np.frombuffer(data,np.dtype(item['stored_dtype'])).reshape(item['shape']).copy()).cuda()
    prepared={k[6:]:load(v) for k,v in original['preprocessing'].items() if k.startswith('input.')}
    for key in ('tokens','tokens_mask'):
        tensor=prepared[key];indices=torch.arange(a.prompt_frames,device='cuda')%tensor.shape[1];prepared[key]=tensor[:,indices]
    prepared['input_pos']=torch.arange(a.prompt_frames,device='cuda').expand(2,-1)
    for key in ('pianoroll','drum_pianoroll','chord','context_mask'):
        tensor=prepared[key];indices=torch.arange(a.symbolic_frames,device='cuda')%tensor.shape[1];prepared[key]=tensor[:,indices]
    def save(name,value):
        if isinstance(value,(tuple,list)):
            for i,v in enumerate(value):save(name+f'.{i}',v)
            return
        if not isinstance(value,torch.Tensor):return
        source_dtype=str(value.dtype);array=np.ascontiguousarray(value.detach().cpu().numpy())
        ext={'float32':'f32','int64':'i64','bool':'bool'}[str(array.dtype)]
        path=root/(name+'.'+ext);data=array.tobytes();path.write_bytes(data)
        files[name]=dict(file=path.name,shape=list(array.shape),source_dtype=source_dtype,stored_dtype=str(array.dtype),sha256=hashlib.sha256(data).hexdigest())
    for key,value in prepared.items():save('input.'+key,value)
    manifest={k:original[k] for k in ('upstream_revision','models','versions','generator_dtype','style_dtype')}
    manifest.update(preprocessing=files,cases=[],prepared_baseline_sha256=hashlib.sha256((a.prepared/'manifest.json').read_bytes()).hexdigest(),
        synthetic_extension='Prompt tokens/masks and symbolic sequences repeated from original frozen inputs; original prefix indices retained')
    source=json.loads(Path('models/generator.json').read_text())
    with torch.inference_mode():
        model,info=MuLaCover.from_pretrained(source['snapshot'],device_map='cuda',dtype=torch.float32,local_files_only=True,output_loading_info=True)
        if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Non-strict model load')
        model.eval().requires_grad_(False)
        base=torch.tensor([[0,8191,7,42,314,4096,123,777],[11,17,23,29,31,37,41,43],[60,64,67,72,4095,4097,8189,2]],device='cuda')
        for name,cfg,count,sampled in [('conditional',1.0,a.frames,False),('cfg-1.5',1.5,a.frames,False),('sample-cfg',1.5,8,True)]:
            root=a.output/name;root.mkdir();files={};batch=1 if cfg==1 else 2
            checkpoints=sorted(set([0,1,7,count-1]));checkpoints=[x for x in checkpoints if x<count]
            teacher=base[torch.arange(count,device='cuda')%3].clone()
            noise=torch.from_numpy(np.random.default_rng(314159).exponential(size=(count,8,8197)).astype(np.float32)).cuda()
            if sampled:save('sampling_noise',noise)
            model.setup_caches(batch);step=[0];call=[0];run=[0];hooks=[]
            def stage(n,o):
                if run[0]==0 and step[0] in checkpoints:save(f'frame{step[0]}.stage.'+n,o)
            for n,module in model.named_modules():
                if n in ('backbone.layers.0','backbone.layers.13','backbone.layers.27','backbone.norm','peft_layers.27','tag_mlp_layers.27'):
                    hooks.append(module.register_forward_hook(lambda m,i,o,n=n:stage(n,o)))
                if n in ('projection','decoder'):
                    hooks.append(module.register_forward_hook(lambda m,i,o,n=n:save(f'frame{step[0]}.local.{call[0]%8}.'+n,o) if run[0]==0 and step[0] in checkpoints else None))
            old_sampler=modeling.sample_topk;old_cfg=model._sample_with_cfg;logs=[];codes=[]
            def sampler(logits,topk,temperature):
                q=call[0]%8;call[0]+=1
                if run[0]==0 and step[0] in checkpoints:save(f'frame{step[0]}.logits.{q}',logits)
                if run[0]==0:logs.append(logits.detach().cpu())
                else:
                    if not torch.equal(logits.cpu(),logs[step[0]*8+q]):raise ValueError('Replay logits differ')
                if sampled:
                    scaled=logits/temperature;threshold=torch.topk(scaled,topk)[0][...,-1,None]
                    probs=torch.softmax(torch.log_softmax(scaled.masked_fill(scaled<threshold,-float('inf')),-1),-1)
                    chosen=torch.argmax(probs/noise[step[0],q],dim=-1,keepdim=True).to(torch.int32)
                    if run[0]==0:teacher[step[0],q]=chosen.item()
                    elif chosen.item()!=teacher[step[0],q].item():raise ValueError('Sampling replay codes differ')
                    return chosen
                return teacher[step[0],q].reshape(1,1).to(torch.int32)
            def cfg_sampler(logits,*args):
                if run[0]==0 and step[0] in checkpoints:save(f'frame{step[0]}.raw_logits.{call[0]%8}',logits)
                return old_cfg(logits,*args)
            modeling.sample_topk=sampler;model._sample_with_cfg=cfg_sampler
            try:
                for replay in range(2):
                    run[0]=replay;call[0]=0;model.reset_caches();current={k:v[:batch] for k,v in prepared.items()}
                    for i in range(count):
                        step[0]=i;sample=model.generate_frame(**current,temperature=1.0,topk=250,cfg_scale=cfg,first_step=i==0)
                        if not torch.equal(sample[0].long(),teacher[i]) or (sample>=8192).any():raise ValueError('Unexpected special/reference code')
                        valid=a.prompt_frames+i
                        if replay==0 and i in checkpoints:
                            for group in ('backbone','decoder'):
                                for layer,block in enumerate(getattr(model,group).layers):
                                    cache=block.attn.kv_cache;active=valid if group=='backbone' else 8
                                    for kind in ('k','v'):
                                        tensor=getattr(cache,kind+'_cache');save(f'frame{i}.cache.{group}.{layer}.{kind}',tensor[:,:,:active])
                                        if torch.count_nonzero(tensor[:,:,active:]):raise ValueError('Nonzero cache tail')
                        tokens=torch.zeros(batch,1,9,dtype=torch.long,device='cuda');tokens[:,:,:8]=sample.unsqueeze(1)
                        mask=torch.ones_like(tokens,dtype=torch.bool);mask[:,:,-1]=False
                        current.update(tokens=tokens,tokens_mask=mask,input_pos=current['input_pos'][:,-1:]+1,qwen_indices=None,muq_indices=None,muq_embedding=None)
            finally:
                modeling.sample_topk=old_sampler;model._sample_with_cfg=old_cfg
                for hook in hooks:hook.remove()
            save('teacher_frames',teacher)
            case=dict(name=name,guidance=cfg,frames=count,batch=batch,checkpoints=checkpoints,replay_bit_exact=True,cache_tail_zero=True,files=files)
            (root/'case.json').write_text(json.dumps(case,indent=2)+'\n');manifest['cases'].append(case)
            print(name,count,'frames, checkpoints',checkpoints,flush=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
