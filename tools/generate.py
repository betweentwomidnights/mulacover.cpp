#!/usr/bin/env python3
"""Prepared-input native generation and staged HeartCodec decode with provenance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
for name in ('reference','model','cli','output','ggml-source'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--flow-model',type=Path);p.add_argument('--sq-model',type=Path)
p.add_argument('--backend',choices=('cpu','cuda'),default='cuda');p.add_argument('--frames',type=int,default=100)
p.add_argument('--capacity',type=int,default=256);p.add_argument('--guidance',type=float,default=1.5)
p.add_argument('--temperature',type=float,default=1);p.add_argument('--top-k',type=int,default=250);p.add_argument('--seed',type=int,default=1701)
p.add_argument('--decoder-seed',type=int,default=1701);p.add_argument('--decoder-steps',type=int,default=10);p.add_argument('--segment-codes',type=int,default=372)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
profile=json.loads((root/'dependencies/ggml-development.json').read_text())
revision=subprocess.check_output(['git','-C',str(a.ggml_source),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(a.ggml_source),'status','--porcelain'],text=True).strip()
if revision!=profile['revision'] or dirty:raise ValueError('Generation requires the clean isolated development GGML profile')
if (a.output/'run.json').exists() or (a.output/'provenance.json').exists():raise ValueError('Use a fresh output directory')
a.output.mkdir(parents=True,exist_ok=True);inp=a.output/'input';inp.mkdir(exist_ok=True)
manifest=json.loads((a.reference/'manifest.json').read_text())
def load(key):
    item=manifest['preprocessing'][key];data=(a.reference/item['file']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Prepared input hash mismatch')
    return np.frombuffer(data,np.dtype(item['stored_dtype'])).reshape(item['shape'])
mapping={'tokens':('input.tokens',np.int32),'mask':('input.tokens_mask',np.uint8),'qwen':('input.qwen_embedding',np.float32),
    'muq':('input.muq_embedding',np.float32),'piano':('input.pianoroll',np.float32),'drum':('input.drum_pianoroll',np.float32),
    'chord':('input.chord',np.float32),'context-mask':('input.context_mask',np.uint8)}
for name,(key,dtype) in mapping.items():
    extension={np.int32:'i32',np.uint8:'u8',np.float32:'f32'}[dtype]
    np.ascontiguousarray(load(key)[0],dtype=dtype).tofile(inp/(name+'.'+extension))
command=[str(a.cli.resolve()),'--model',str(a.model.resolve()),'--input',str(inp.resolve()),'--output',str(a.output.resolve()),
    '--prompt-frames',str(load('input.tokens').shape[1]),'--symbolic-frames',str(load('input.pianoroll').shape[1]),
    '--qwen-index',str(load('input.qwen_indices')[0]),'--muq-index',str(load('input.muq_indices')[0]),
    '--backend',a.backend,'--frames',str(a.frames),'--capacity',str(a.capacity),'--guidance',str(a.guidance),
    '--temperature',str(a.temperature),'--top-k',str(a.top_k),'--seed',str(a.seed),
    '--decoder-seed',str(a.decoder_seed),'--decoder-steps',str(a.decoder_steps),'--segment-codes',str(a.segment_codes)]
if bool(a.flow_model)!=bool(a.sq_model):raise ValueError('Supply both codec models or neither')
if a.flow_model:command+=['--flow-model',str(a.flow_model.resolve()),'--sq-model',str(a.sq_model.resolve())]
spec=importlib.util.spec_from_file_location('native_parity',root/'tools/run-parity.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
with (a.output/'stdout.txt').open('w') as out,(a.output/'stderr.txt').open('w') as err:
    run,memory=helper.run_native(command,out,err,dict(os.environ,GGML_CUDA_TF32='0',GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto',CUBLAS_WORKSPACE_CONFIG=':4096:8'),a.backend=='cuda')
if run.returncode:raise RuntimeError(f'Native generation failed: {(a.output/"stderr.txt").read_text()[-3000:]}')
result=json.loads((a.output/'run.json').read_text())
if result['ggml_commit']!=revision[:8]:raise ValueError('Linked GGML revision mismatch')
codes=np.fromfile(a.output/'codes.i32',np.int32)
if codes.size!=result['frames']*8 or np.any(codes<0) or np.any(codes>=8192):raise ValueError('Invalid RVQ output')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(16*1024*1024),b''):h.update(data)
    return h.hexdigest()
report=dict(ggml_revision=revision,shared_baseline=profile['shared_baseline'],ggml_bump_check_started=False,
    command=command,decoder=dict(seed=a.decoder_seed,steps=a.decoder_steps,segment_codes=a.segment_codes,guidance=1.25),
    prepared_manifest_sha256=digest(a.reference/'manifest.json'),upstream_revision=manifest['upstream_revision'],
    model_sha256=digest(a.model),codes_sha256=digest(a.output/'codes.i32'),execution_passed=True,
    audio_quality_accepted=False,device_memory_sampling=memory,run=result,
    rng='Native SplitMix64 indexed by seed/frame/head/category, exponential race; not a PyTorch RNG seed')
if a.flow_model:
    if (a.output/'audio.wav').stat().st_size!=44+result['audio_samples_per_channel']*8:raise ValueError('WAV length mismatch')
    report.update(flow_model_sha256=digest(a.flow_model),sq_model_sha256=digest(a.sq_model),audio_sha256=digest(a.output/'audio.wav'))
(a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['run'],indent=2))
