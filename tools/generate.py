#!/usr/bin/env python3
"""Reference input/provenance harness around the standalone native continuation CLI."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import numpy as np
from continuation import select_guide,plan_frames,frames_for_seconds

p=argparse.ArgumentParser(description=__doc__)
for name in ('reference','model','cli','output','ggml-source'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--flow-model',type=Path);p.add_argument('--sq-model',type=Path)
p.add_argument('--backend',choices=('cpu','cuda','vulkan','metal'),default='cuda');p.add_argument('--frames',type=int,help='Total guide plus new frames; default 100')
p.add_argument('--capacity',type=int,help='Cache positions; default rounds required positions up to 256');p.add_argument('--guidance',type=float,default=1.5)
p.add_argument('--temperature',type=float,default=1);p.add_argument('--top-k',type=int,default=250);p.add_argument('--seed',type=int,default=1701)
p.add_argument('--decoder-seed',type=int,default=1701);p.add_argument('--decoder-steps',type=int,default=10);p.add_argument('--segment-codes',type=int,default=372)
p.add_argument('--rng',choices=('native','torch-cuda'),default='native');p.add_argument('--rng-cuda-blocks',type=int,default=0);p.add_argument('--rng-device',type=int,default=0)
p.add_argument('--experimental-bpm',type=float,default=None,help='Experimental moving score-attention prior; not a guaranteed tempo')
p.add_argument('--tempo-width-beats',type=float,default=1);p.add_argument('--tempo-offset-beats',type=float,default=0)
p.add_argument('--tempo-local-window',action='store_true',help='Also restrict symbolic adaptor and score-attention keys')
p.add_argument('--audio-prefix',type=Path,help='Experimental frame-major int32[frames,8] guide. Output includes this teacher-forced prefix; measure continuation separately.')
p.add_argument('--guide-seconds',help='Use this duration of encoded input; rounds down to 80-ms frames, default all available')
p.add_argument('--guide-start-seconds',default='0',help='Start within encoded input; rounds down to 80-ms frames')
p.add_argument('--continuation-seconds',help='Requested NEW audio after the guide; mutually exclusive with --frames. EOS may finish earlier.')
p.add_argument('--sampler-offset',type=int,default=0,help='Explicit initial Torch CUDA Philox offset; teacher-forced guide frames do not draw random numbers')
a=p.parse_args();root=Path(__file__).resolve().parents[1]
profile=json.loads((root/'dependencies/ggml-development.json').read_text())
revision=subprocess.check_output(['git','-C',str(a.ggml_source),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(a.ggml_source),'status','--porcelain'],text=True).strip()
if revision!=profile['revision'] or dirty:raise ValueError('Generation requires the clean isolated development GGML profile')
if (a.output/'run.json').exists() or (a.output/'provenance.json').exists():raise ValueError('Use a fresh output directory')
manifest=json.loads((a.reference/'manifest.json').read_text())
prefix=None;selection=None
if a.audio_prefix:prefix,selection=select_guide(a.audio_prefix,a.guide_seconds,a.guide_start_seconds)
elif a.guide_seconds is not None or frames_for_seconds(a.guide_start_seconds,allow_zero=True):raise ValueError('Guide selection needs --audio-prefix')
requested_frames,requested_capacity=a.frames,a.capacity
a.frames,a.capacity=plan_frames(manifest['preprocessing']['input.tokens']['shape'][1],len(prefix) if prefix is not None else 0,
    a.frames,a.continuation_seconds,a.capacity)
a.output.mkdir(parents=True,exist_ok=True);inp=a.output/'input';inp.mkdir(exist_ok=True)
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
    '--backend',a.backend,'--guidance',str(a.guidance),
    '--temperature',str(a.temperature),'--top-k',str(a.top_k),'--seed',str(a.seed),
    '--decoder-seed',str(a.decoder_seed),'--decoder-steps',str(a.decoder_steps),'--segment-codes',str(a.segment_codes)]
command+=['--rng',a.rng]
if a.continuation_seconds is not None:command+=['--continuation-seconds',a.continuation_seconds]
elif requested_frames is not None:command+=['--frames',str(requested_frames)]
if requested_capacity is not None:command+=['--capacity',str(requested_capacity)]
if prefix is not None:
    command+=['--audio-prefix',str(a.audio_prefix.resolve()),'--guide-start-seconds',a.guide_start_seconds]
    if a.guide_seconds is not None:command+=['--guide-seconds',a.guide_seconds]
if a.experimental_bpm is not None:
    command+=['--experimental-bpm',str(a.experimental_bpm),'--tempo-width-beats',str(a.tempo_width_beats),'--tempo-offset-beats',str(a.tempo_offset_beats)]
    if a.tempo_local_window:command+=['--tempo-local-window','1']
elif a.tempo_width_beats!=1 or a.tempo_offset_beats!=0 or a.tempo_local_window:raise ValueError('Tempo options require --experimental-bpm')
if a.rng=='torch-cuda':command+=['--rng-cuda-blocks',str(a.rng_cuda_blocks),'--rng-device',str(a.rng_device)]
elif a.rng_cuda_blocks or a.rng_device or a.sampler_offset:raise ValueError('CUDA RNG options require torch-cuda mode')
if a.sampler_offset:command+=['--sampler-offset',str(a.sampler_offset)]
if bool(a.flow_model)!=bool(a.sq_model):raise ValueError('Supply both codec models or neither')
if a.flow_model:command+=['--flow-model',str(a.flow_model.resolve()),'--sq-model',str(a.sq_model.resolve())]
spec=importlib.util.spec_from_file_location('native_parity',root/'tools/run-parity.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
with (a.output/'stdout.txt').open('w') as out,(a.output/'stderr.txt').open('w') as err:
    run,memory=helper.run_native(command,out,err,dict(os.environ,GGML_CUDA_TF32='0',GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto',CUBLAS_WORKSPACE_CONFIG=':4096:8'),a.backend=='cuda')
if run.returncode:raise RuntimeError(f'Native generation failed: {(a.output/"stderr.txt").read_text()[-3000:]}')
result=json.loads((a.output/'run.json').read_text())
if result['ggml_commit']!=revision[:8]:raise ValueError('Linked GGML revision mismatch')
if result['cache_capacity']!=a.capacity or result['requested_new_frames']!=a.frames-(len(prefix) if prefix is not None else 0):raise ValueError('Native and reference timeline plans differ')
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
    model_sha256=digest(a.model),cli_sha256=digest(a.cli),codes_sha256=digest(a.output/'codes.i32'),execution_passed=True,
    audio_quality_accepted=False,device_memory_sampling=memory,run=result,
    rng=('Torch CUDA Philox4x32-10, F32 exponential per head and consecutive F32 normal per codec window; explicit target grid cap'
        if a.rng=='torch-cuda' else 'Native SplitMix64 indexed by seed/frame/head/category, exponential race; not a PyTorch RNG seed'))
if a.audio_prefix:
    if result['newly_generated_frames']<1:raise ValueError('No newly generated audio after the prefix')
    if prefix.size!=result['audio_prefix_frames']*8 or not np.array_equal(codes[:prefix.size],prefix.ravel()):raise ValueError('Output prefix does not equal the forced guide')
    if result['guide_source_start_frame']!=selection['source_start_frame']:raise ValueError('Native and reference guide selections differ')
    report.update(audio_prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),audio_prefix_selection=selection,audio_prefix_included_in_output=True,
        continuation_start_seconds=result['audio_prefix_frames']*.08,continuation_seconds=result['newly_generated_frames']*.08)
if a.flow_model:
    if (a.output/'audio.wav').stat().st_size!=44+result['audio_samples_per_channel']*8:raise ValueError('WAV length mismatch')
    report.update(flow_model_sha256=digest(a.flow_model),sq_model_sha256=digest(a.sq_model),audio_sha256=digest(a.output/'audio.wav'))
    if prefix is not None:
        suffix=(a.output/'continuation.wav').read_bytes();full=(a.output/'audio.wav').read_bytes()
        if len(suffix)!=44+result['newly_generated_frames']*3840*8 or suffix[44:]!=full[44+result['audio_prefix_frames']*3840*8:]:raise ValueError('Native continuation export differs from the waveform suffix')
        report.update(continuation_audio_sha256=digest(a.output/'continuation.wav'),continuation_audio_samples_per_channel=result['newly_generated_frames']*3840)
report['requested_timeline']=dict(total_frames=a.frames,guide_frames=len(prefix) if prefix is not None else 0,
    new_frames=a.frames-(len(prefix) if prefix is not None else 0),continuation_seconds=a.continuation_seconds)
report['driver_role']='Reference preprocessing, provenance and independent validation only; the native CLI plans, selects and exports the continuation'
(a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['run'],indent=2))
