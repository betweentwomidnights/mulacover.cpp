#!/usr/bin/env python3
"""Encode audio for continuation guides with the pinned official HeartCodec encoder.

Only this input guide may be tempo-scaled to create a counterfactual conditioning
control. Generated continuations are never stretched. Run in Spark Docker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import types
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--upstream',type=Path);p.add_argument('--codec-snapshot',type=Path)
    p.add_argument('--audio',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--factors',type=float,nargs='+',default=[1.],help='Guide speed factors; default source speed only. Extra factors are research controls.')
    a=p.parse_args();source_path=Path('refdata/bpm-prefix/encoder-source.json')
    if a.download:
        from huggingface_hub import snapshot_download
        revision='e6020510612d6653b73d6cd54393ef80929e3c81'
        path=Path(snapshot_download('HeartMuLa/HeartCodec-oss-encoder',revision=revision,
            allow_patterns=['encoder.safetensors','config.json','encoder_config.json','SHA256SUMS','LICENSE','README.md']))
        checksum=digest(path/'encoder.safetensors')
        if checksum!=(path/'SHA256SUMS').read_text().split()[0]:raise ValueError('Encoder checksum mismatch')
        source_path.parent.mkdir(parents=True,exist_ok=True)
        source_path.write_text(json.dumps(dict(repository='HeartMuLa/HeartCodec-oss-encoder',revision=revision,snapshot=str(path),weights_sha256=checksum),indent=2)+'\n')
        print(revision,checksum,flush=True);return
    if any(v is None for v in (a.upstream,a.codec_snapshot,a.audio,a.output)):p.error('Encoding needs upstream, codec snapshot, audio and output')
    if a.output.exists():raise ValueError('Use a fresh prefix directory')
    revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip()
    if revision!='5858ca8f1ffe58d62be7c007ea55f1033c5ca456' or dirty:raise ValueError('Expected clean pinned heartlib encoder source')
    if a.codec_snapshot.name!='f889dab0532cfa4bf459f2a3367eb6d346b8eeda':raise ValueError('Expected existing decoder revision')
    import torch
    import soundfile as sf
    # Import the codec directly, without loading unrelated generation pipelines.
    package=types.ModuleType('heartlib');package.__path__=[str(a.upstream/'src/heartlib')];sys.modules['heartlib']=package
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    encoder=json.loads(source_path.read_text())
    if digest(Path(encoder['snapshot'])/'encoder.safetensors')!=encoder['weights_sha256']:raise ValueError('Encoder weights changed')
    a.output.mkdir(parents=True)
    waveform,rate=sf.read(a.audio,dtype='float32',always_2d=True)
    if not len(waveform) or waveform.shape[1] not in (1,2):raise ValueError('Input audio must be nonempty mono or stereo')
    # Model arithmetic remains FP32, including the shared decoder RVQ.
    model=HeartCodec.from_encoder_decoder_pretrained(a.codec_snapshot,encoder['snapshot'],local_files_only=True,dtype=torch.float32).to(a.device).eval()
    cases=[]
    for factor in a.factors:
        if not .5<=factor<=2:raise ValueError('Invalid guide speed factor')
        name=format(factor,'g');guide=a.output/('guide-'+name+'.wav')
        if factor==1:shutil.copyfile(a.audio,guide)
        else:subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(a.audio),'-af','atempo='+name,'-c:a','pcm_f32le',str(guide)],check=True)
        audio,rate=sf.read(guide,dtype='float32',always_2d=True)
        with torch.inference_mode():codes=model.tokenize(torch.from_numpy(audio.T.copy()),rate)
        # Discard encoder's extra partial frame; feed only complete guide frames.
        frames=int(len(audio)/rate/.08);array=np.ascontiguousarray(codes[:,:frames].T.numpy(),dtype='<i4')
        if array.shape!=(frames,8) or np.any(array<0) or np.any(array>=8192):raise ValueError('Invalid prefix RVQ codes')
        path=a.output/('guide-'+name+'.i32');array.tofile(path)
        cases.append(dict(speed_factor=factor,frames=frames,guide_seconds=len(audio)/rate,
            complete_token_seconds=frames*.08,codes_file=path.name,codes_sha256=digest(path),guide_sha256=digest(guide)))
        print('Encoded',name,frames,'frames',flush=True)
    report=dict(heartlib_revision=revision,encoder=encoder,codec_revision=a.codec_snapshot.name,source_excerpt_sha256=digest(a.audio),
        harness_sha256=digest(__file__),torch=torch.__version__,cuda=torch.version.cuda,device=a.device,cases=cases,
        intervention='Pitch-preserving ffmpeg atempo on input guide only, then original FP32 encoder and decoder RVQ; generated continuations must be measured excluding the guide')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
