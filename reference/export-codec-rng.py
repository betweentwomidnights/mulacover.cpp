#!/usr/bin/env python3
"""Exercise the real upstream codec random-call graph with deterministic stubs."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Use fresh codec RNG fixtures')
a.output.mkdir(parents=True);sys.path.insert(0,str(a.upstream/'src'))
from mulacover._codec.modeling import HeartCodec
from mulacover._codec.models.flow_matching import FlowMatching
captured=[]
class VQ:
    def eval(self):return self
    def get_output_from_indices(self,codes):return torch.zeros((*codes.shape[:2],512),device='cuda')
flow=SimpleNamespace(vq_embed=VQ(),cond_feature_emb=torch.nn.Identity(),latent_dim=256,zero_cond_embedding1=torch.zeros(512,device='cuda'))
def solve(x,context,prefix,*unused):
    captured.append(x.clone());return x
flow.solve_euler=solve
flow.inference_codes=lambda *args,**kwargs:FlowMatching.inference_codes(flow,*args,**kwargs)
scalar=SimpleNamespace(decode=lambda x:torch.zeros((x.shape[0],1,x.shape[-1]*1920),device='cuda'))
codec=SimpleNamespace(dtype=torch.float32,device=torch.device('cuda'),flow_matching=flow,scalar_model=scalar,sample_rate=48000)
torch.manual_seed(1701);before=torch.get_rng_state().clone()
audio=HeartCodec.detokenize(codec,torch.zeros((8,861),device='cuda',dtype=torch.long),disable_progress=True)
assert len(captured)==3 and audio.shape==(2,861*3840)
cpu_changed=not torch.equal(before,torch.get_rng_state());assert cpu_changed
gen=torch.Generator(device='cuda').manual_seed(1701)
expected=torch.stack([torch.randn((1,744,256),device='cuda',generator=gen) for _ in range(3)])
actual=torch.stack(captured);assert torch.equal(actual,expected)
data=np.ascontiguousarray(actual.cpu().numpy()).tobytes();file='codec-flow-normal.f32';(a.output/file).write_bytes(data)
report=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),upstream_revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip(),
    real_upstream_detokenize_and_inference_codes=True,model_arithmetic_stubbed=True,cpu_fillers_advance_cpu_rng_only=cpu_changed,
    effective_flow_noise_equals_three_consecutive_cuda_randn_calls=True,cases=[dict(kind='normal',seed=1701,count=744*256,draws=3,offset=0,next_offset=gen.get_offset(),file=file,sha256=hashlib.sha256(data).hexdigest())])
(a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
