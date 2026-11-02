#!/usr/bin/env python3
"""Freeze actual upstream top-k probabilities with explicit exponential noise."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.upstream/'src'));import mulacover.modeling as modeling
a.output.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(1701)
cases=[('ordinary',250,1.0),('warm',25,1.7),('full',8197,0.7),('ties',2,1.0),('flat',1,1.0),('extreme',16,0.1),('eos',1,1.0),('padding',1,1.0)]
manifest=dict(upstream_revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip(),torch_version=torch.__version__,noise_contract='Supplied positive exponential noise, not RNG seed equivalence',cases=[])
for name,k,t in cases:
    logits=rng.normal(size=8197).astype(np.float32)*3;noise=np.maximum(rng.exponential(size=8197).astype(np.float32),np.float32(1e-7))
    if name=='ties':logits.fill(-20);logits[3]=2;logits[17]=1;logits[23]=1;noise[23]=0.001
    if name=='flat':logits.fill(1)
    if name=='extreme':logits=np.linspace(-1000,1000,8197,dtype=np.float32)
    if name in ('eos','padding'):logits.fill(-10);logits[8193 if name=='eos' else 8192]=10
    captured=[]
    def pick(probs):
        captured.append(probs.cpu().numpy().copy());return torch.argmax(probs/torch.from_numpy(noise),dim=-1,keepdim=True).to(torch.int32)
    old=modeling._multinomial_sample_one_no_sync;modeling._multinomial_sample_one_no_sync=pick
    try:chosen=modeling.sample_topk(torch.from_numpy(logits)[None,:],k,t).item()
    finally:modeling._multinomial_sample_one_no_sync=old
    files={}
    for key,value in [('logits',logits),('noise',noise),('probabilities',captured[0].reshape(-1))]:
        data=value.tobytes();file=name+'.'+key+'.f32';(a.output/file).write_bytes(data);files[key]=dict(file=file,sha256=hashlib.sha256(data).hexdigest())
    manifest['cases'].append(dict(name=name,top_k=k,temperature=t,chosen=chosen,files=files))
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
