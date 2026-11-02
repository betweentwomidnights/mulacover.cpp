#!/usr/bin/env python3
"""Audition the pinned upstream FP32 pipeline against hash-checked native inputs."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','prepared','request','codec-source','codec-snapshot','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=1701);p.add_argument('--decoder-seed',type=int,default=1701)
    a=p.parse_args()
    harness_sha256=digest(__file__)
    if a.output.exists():raise ValueError('Use a fresh upstream output directory')
    frozen=json.loads((a.prepared/'manifest.json').read_text())
    revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip()
    if revision!=frozen['upstream_revision'] or dirty:raise ValueError('Expected clean pinned upstream source')
    sources={name:json.loads(Path('models',name+'.json').read_text()) for name in ('generator','style')}
    for name,value in sources.items():
        if value['revision']!=frozen['models'][name]['revision']:raise ValueError('Model revision mismatch')
    codec_source=json.loads(a.codec_source.read_text())
    if a.codec_snapshot.name!=codec_source['revision'] or not a.codec_snapshot.is_dir():raise ValueError('Codec snapshot mismatch')
    sq_source=json.loads((a.codec_source.parents[1]/'sq/provenance.json').read_text())
    if codec_source['revision']!=sq_source['source_revision']:raise ValueError('Codec differs from native SQ source')
    frames=frozen['requested_frames'];guidance=frozen['guidance']
    if frames<1 or frames>8192 or guidance!=1.5:raise ValueError('Unexpected control settings')
    request=json.loads(a.request.read_text());inputs={k:request[k] for k in ('lyrics','tags')};input_files={}
    for key in ('melody_midi','chord_midi','drum_midi'):
        if request.get(key) is None:continue
        path=(a.request.parent/request[key]).resolve();inputs[key]=path;input_files[key]=digest(path)
        if input_files[key]!=frozen['input_files'][key]['sha256']:raise ValueError('Request MIDI differs from native control')
    if set(input_files)!=set(frozen['input_files']):raise ValueError('Request MIDI set changed')
    sys.path.insert(0,str(a.upstream/'src'))
    from mulacover import MuLaCoverGenPipeline,MuLaCoverGenConfig,MuLaCoverConfig
    from tokenizers import Tokenizer
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    generator=Path(sources['generator']['snapshot'])
    paths=dict(mulacover=generator,qwen=Path(sources['style']['snapshot']),codec=a.codec_snapshot)
    devices=dict.fromkeys(('mulacover','qwen','codec','transcriptor'),torch.device('cuda'))
    pipeline=MuLaCoverGenPipeline(paths,devices,dict.fromkeys(devices,torch.float32),
        Tokenizer.from_file(str(generator/'tokenizer.json')),MuLaCoverGenConfig.from_file(generator/'gen_config.json'),
        MuLaCoverConfig.from_pretrained(generator,local_files_only=True),lazy_load=True)
    a.output.mkdir(parents=True)
    print('Running official preprocessing',flush=True)
    with torch.inference_mode():prepared=pipeline.preprocess(inputs,cfg_scale=guidance,max_audio_length_ms=frames*80,symbolic_save_dir=a.output/'condition-midi')
    matches={}
    if set('input.'+key for key in prepared)!=set(frozen['preprocessing']):raise ValueError('Prepared tensor set changed')
    for key,value in prepared.items():
        item=frozen['preprocessing']['input.'+key];array=np.ascontiguousarray(value.detach().cpu().numpy())
        checksum=hashlib.sha256(array.tobytes()).hexdigest()
        if digest(a.prepared/item['file'])!=item['sha256']:raise ValueError('Frozen input hash changed')
        matches[key]=checksum==item['sha256'] and list(array.shape)==item['shape'] and str(array.dtype)==item['stored_dtype']
    if not all(matches.values()):raise ValueError('Fresh official preprocessing differs from native control')
    if pipeline._qwen is not None:raise ValueError('Qwen was not released')
    (a.output/'preparation-check.json').write_text(json.dumps(matches,indent=2)+'\n')
    print('All',len(matches),'prepared tensors match; loading official FP32 generator',flush=True)
    torch.cuda.reset_peak_memory_stats()
    dtypes=sorted({str(v.dtype) for v in pipeline.mulacover.parameters() if v.is_floating_point()})
    if dtypes!=['torch.float32']:raise ValueError('Generator is not entirely FP32')
    capacity=pipeline.mulacover.backbone.max_seq_len
    calls=[0];start=time.perf_counter()
    def progress(*unused):
        calls[0]+=1
        if calls[0]%50==0:print('Official backbone calls',calls[0],'elapsed',round(time.perf_counter()-start,1),'seconds',flush=True)
    hook=pipeline.mulacover.backbone.layers[0].register_forward_hook(progress)
    torch.manual_seed(a.seed);torch.cuda.synchronize();start=time.perf_counter()
    try:outputs=pipeline._forward(prepared,max_audio_length_ms=frames*80,temperature=1.,topk=250,cfg_scale=guidance,disable_progress=True)
    finally:hook.remove()
    torch.cuda.synchronize();generation_seconds=time.perf_counter()-start;generation_peak=torch.cuda.max_memory_allocated()
    if pipeline._mulacover is not None:raise ValueError('Generator was not released before decode')
    codes=np.ascontiguousarray(outputs['frames'].cpu().numpy().T,dtype='<i4')
    if codes.ndim!=2 or codes.shape[1]!=8 or np.any(codes<0) or np.any(codes>=8192):raise ValueError('Invalid upstream codes')
    codes.tofile(a.output/'codes.i32')
    (a.output/'generation.json').write_text(json.dumps(dict(frames=len(codes),eos=len(codes)<frames,seconds=generation_seconds,peak_allocated_bytes=generation_peak),indent=2)+'\n')
    print('Official generation finished:',len(codes),'frames; loading official FP32 codec',flush=True)
    torch.cuda.reset_peak_memory_stats();codec=pipeline.codec
    if {v.dtype for v in codec.parameters() if v.is_floating_point()}!={torch.float32}:raise ValueError('Codec is not entirely FP32')
    original_decode=codec.detokenize;audio_info={}
    def capture(*args,**kwargs):
        waveform=original_decode(*args,**kwargs)
        array=np.ascontiguousarray(waveform.detach().float().cpu().numpy().T,dtype='<f4')
        if array.shape!=(len(codes)*3840,2) or not np.isfinite(array).all():raise ValueError('Invalid official decoded audio')
        data=array.tobytes();rate=codec.sample_rate
        header=struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(data),b'WAVE',b'fmt ',16,3,2,rate,rate*8,8,32,b'data',len(data))
        (a.output/'audio-f32.wav').write_bytes(header+data)
        audio_info.update(samples_per_channel=len(array),sample_rate=rate,channels=2,peak=float(np.abs(array).max()),rms=float(np.sqrt(np.mean(array.astype(np.float64)**2))),samples_above_full_scale=int(np.count_nonzero(np.abs(array)>1)))
        return waveform
    codec.detokenize=capture
    torch.manual_seed(a.decoder_seed);torch.cuda.synchronize();start=time.perf_counter()
    try:pipeline.postprocess(outputs,save_path=a.output/'audio-official.wav',disable_progress=True)
    finally:del codec.detokenize
    torch.cuda.synchronize();decode_seconds=time.perf_counter()-start;decode_peak=torch.cuda.max_memory_allocated()
    del original_decode,codec;gc.collect();torch.cuda.empty_cache()
    report=dict(harness_sha256=harness_sha256,upstream_revision=revision,upstream_worktree_clean=True,models=frozen['models'],codec_source=codec_source,
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','torchtune','tokenizers','numpy','mido','soundfile')},
        request=request,request_sha256=digest(a.request),input_midi_hashes=input_files,prepared_manifest_sha256=digest(a.prepared/'manifest.json'),
        fresh_preprocessing_bit_exact=matches,generator_dtype='float32',codec_dtype='float32',requested_frames=frames,frames=len(codes),eos=len(codes)<frames,
        generation_seed=a.seed,decoder_seed=a.decoder_seed,guidance=guidance,temperature=1.,top_k=250,upstream_cache_capacity=capacity,
        decoder=dict(duration=29.76,segment_codes=372,steps=10,guidance=1.25),generation_seconds=generation_seconds,decode_seconds=decode_seconds,
        generation_peak_allocated_bytes=generation_peak,decode_peak_allocated_bytes=decode_peak,audio=audio_info,
        files={name:digest(a.output/name) for name in ('codes.i32','audio-f32.wav','audio-official.wav')},
        sampler='Unmodified upstream Torch exponential sampler; seed is not native SplitMix64-equivalent',
        codec_noise='Unmodified upstream Torch codec noise; not native C++ RNG-equivalent',
        instrumentation='Backbone progress hook and decoder-output capture only; original upstream generator/sampler/codec/postprocess execute',
        output_formats=dict(original_upstream_postprocess='audio-official.wav (soundfile default PCM16)',unclipped_decoder_capture='audio-f32.wav (float32 stereo)'),
        execution_passed=True,instrumental_control_accepted=False,listening_assessment='pending',ggml_bump_check_started=False)
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Official pipeline finished:',len(codes)*.08,'seconds; peak',audio_info['peak'],flush=True)


if __name__=='__main__':main()
