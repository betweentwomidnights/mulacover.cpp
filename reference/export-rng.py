#!/usr/bin/env python3
"""Freeze actual contiguous Torch CUDA RNG calls without loading any model."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use fresh RNG fixtures')
    a.output.mkdir(parents=True)
    prop=torch.cuda.get_device_properties(0)
    report=dict(torch=torch.__version__,cuda=torch.version.cuda,device=prop.name,sm=prop.multi_processor_count,cases=[])
    cases=[]
    for kind in ('normal','exponential'):
        for seed in (0,1701,0x1234567887654321):
            for count,draws,offset in ((1,3,0),(257,3,4),(8197,8,0),(190464,3,32),(2000000,2,1<<34)):
                cases.append((kind,seed,count,draws,offset))
    # Exact full-song RVQ-head stream, not one flattened RNG call.
    cases.append(('exponential',1701,8197,875*8,0))
    for index,(kind,seed,count,draws,offset) in enumerate(cases):
        gen=torch.Generator(device='cuda').manual_seed(seed);gen.set_offset(offset)
        values=[]
        for _ in range(draws):
            if kind=='normal':value=torch.randn(count,generator=gen,device='cuda',dtype=torch.float32)
            else:value=torch.empty(count,device='cuda',dtype=torch.float32).exponential_(1.,generator=gen)
            values.append(value)
        array=np.ascontiguousarray(torch.stack(values).cpu().numpy());file=f'{index:02d}-{kind}.f32';data=array.tobytes();(a.output/file).write_bytes(data)
        report['cases'].append(dict(kind=kind,seed=seed,count=count,draws=draws,offset=offset,next_offset=gen.get_offset(),file=file,sha256=hashlib.sha256(data).hexdigest()))
        del values,array,value,data
        print('Frozen RNG',index,kind,count,draws,'offset',offset,flush=True)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
