#!/usr/bin/env python3
"""Freeze actual upstream prompt/MIDI/Qwen preparation for a concrete request."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','request','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('refdata/upstream-f32'))
    p.add_argument('--frames',type=int,default=100);p.add_argument('--guidance',type=float,default=1.5)
    a=p.parse_args()
    if (a.output/'manifest.json').exists():raise ValueError('Use a fresh conditioning output')
    baseline=json.loads((a.baseline/'manifest.json').read_text())
    revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=baseline['upstream_revision']:raise ValueError('Upstream revision does not match the frozen baseline')
    sources={name:json.loads(Path('models',name+'.json').read_text()) for name in ('generator','style')}
    for name,value in sources.items():
        if value['revision']!=baseline['models'][name]['revision']:raise ValueError('Model revision mismatch')
    request=json.loads(a.request.read_text());inputs={k:request[k] for k in ('lyrics','tags')}
    normalized_tags=request['tags'].strip().lower()
    if normalized_tags.startswith('<tag>') and normalized_tags.endswith('</tag>'):normalized_tags=normalized_tags[5:-6].strip()
    input_files={}
    for key in ('melody_midi','chord_midi','drum_midi'):
        if request.get(key) is None:continue
        path=(a.request.parent/request[key]).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        inputs[key]=path;input_files[key]=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    sys.path.insert(0,str(a.upstream/'src'))
    from mulacover import MuLaCoverGenPipeline,MuLaCoverGenConfig,MuLaCoverConfig
    from mulacover.symbolic import SymbolicCondition
    from tokenizers import Tokenizer
    torch.set_num_threads(4);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    paths={k:Path(v['snapshot']) for k,v in sources.items()};paths['mulacover']=paths.pop('generator');paths['qwen']=paths.pop('style')
    devices=dict.fromkeys(('mulacover','qwen','codec','transcriptor'),torch.device('cuda'))
    pipeline=MuLaCoverGenPipeline(paths,devices,dict.fromkeys(devices,torch.float32),
        Tokenizer.from_file(str(paths['mulacover']/'tokenizer.json')),MuLaCoverGenConfig.from_file(paths['mulacover']/'gen_config.json'),
        MuLaCoverConfig.from_pretrained(paths['mulacover'],local_files_only=True),lazy_load=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        prepared=pipeline.preprocess(inputs,cfg_scale=a.guidance,max_audio_length_ms=a.frames*80,symbolic_save_dir=a.output/'condition-midi')
    if pipeline._qwen is not None:raise ValueError('Qwen model was not released after preparation')
    condition=SymbolicCondition.from_midi(a.output/'condition-midi/melody.mid',a.output/'condition-midi/chord.mid',a.output/'condition-midi/drums.mid')
    replay=condition.to_tensors()
    if not all(torch.equal(prepared[k][0],v) for k,v in replay.items()):raise ValueError('Symbolic MIDI roundtrip changed conditioning')
    files={}
    for key,value in prepared.items():
        array=np.ascontiguousarray(value.detach().cpu().numpy());ext={'float32':'f32','int64':'i64','bool':'bool'}[str(array.dtype)]
        path=a.output/('input.'+key+'.'+ext);data=array.tobytes();path.write_bytes(data)
        files['input.'+key]=dict(file=path.name,shape=list(array.shape),source_dtype=str(value.dtype),stored_dtype=str(array.dtype),sha256=hashlib.sha256(data).hexdigest())
    (a.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    manifest=dict(upstream_revision=revision,models=baseline['models'],generator_dtype='float32',style_dtype='float32',
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','torchtune','tokenizers','numpy','mido')},
        preparation_only=True,preprocessing=files,cases=[],request_sha256=hashlib.sha256(a.request.read_bytes()).hexdigest(),
        request=request,input_files=input_files,requested_frames=a.frames,guidance=a.guidance,
        normalized_lyrics=request['lyrics'].lower(),normalized_tags=normalized_tags,
        style_regenerated=True,qwen_released=True,symbolic_midi_roundtrip_exact=True,
        symbolic_frames=int(prepared['pianoroll'].shape[1]),drum_condition_zero=bool(torch.count_nonzero(prepared['drum_pianoroll'])==0))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared',a.output,'prompt',prepared['tokens'].shape[1],'symbolic',manifest['symbolic_frames'],'fresh style',files['input.qwen_embedding']['sha256'],flush=True)


if __name__=='__main__':main()
