#!/usr/bin/env python3
"""Require bit-exact native/Torch RNG values and generator offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for name in ('reference','cli','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--cuda-blocks',type=int,default=0);a=p.parse_args()
if a.output.exists():raise ValueError('Use fresh RNG comparison outputs')
a.output.mkdir(parents=True);manifest=json.loads((a.reference/'manifest.json').read_text());results=[]
for case in manifest['cases']:
    source=a.reference/case['file'];data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!=case['sha256']:raise ValueError('RNG oracle hash changed')
    target=a.output/case['file']
    command=[str(a.cli.resolve()),case['kind'],str(case['seed']),str(case['offset']),str(case['count']),str(case['draws']),str(a.cuda_blocks),'0',str(target.resolve())]
    run=subprocess.run(command,text=True,capture_output=True)
    if run.returncode:raise RuntimeError(run.stderr)
    state=json.loads(run.stdout);native=target.read_bytes();x=np.frombuffer(data,np.float32);y=np.frombuffer(native,np.float32)
    exact=len(native)==len(data) and native==data;offset=state['next_offset']==case['next_offset']
    results.append(dict(file=case['file'],kind=case['kind'],seed=case['seed'],count=case['count'],draws=case['draws'],offset=case['offset'],
        values=x.size,bit_exact=exact,offset_exact=offset,changed_values=int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))),max_abs=float(np.abs(x-y).max()),state=state))
    print(case['file'],'bit-exact',exact,'offset',offset,flush=True)
report=dict(reference_device=manifest['device'],torch=manifest['torch'],cuda=manifest['cuda'],cuda_blocks_override=a.cuda_blocks,cases=results,passed=all(c['bit_exact'] and c['offset_exact'] for c in results))
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['passed']:raise SystemExit(1)
