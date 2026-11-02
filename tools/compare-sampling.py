#!/usr/bin/env python3
"""Compare native sampler probabilities and selections to frozen upstream tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--cli',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads((a.reference/'manifest.json').read_text());report=dict(upstream_revision=m['upstream_revision'],gates=dict(max_abs=1e-6,relative_l2=1e-5,selection='exact with supplied exponential noise'),cases=[]);a.output.mkdir(parents=True,exist_ok=True)
for c in m['cases']:
    for item in c['files'].values():
        if hashlib.sha256((a.reference/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Sampler oracle hash mismatch')
    path=a.output/(c['name']+'.f32')
    chosen=int(subprocess.check_output([str(a.cli.resolve()),'--logits',str((a.reference/c['files']['logits']['file']).resolve()),
        '--noise',str((a.reference/c['files']['noise']['file']).resolve()),'--temperature',str(c['temperature']),
        '--top-k',str(c['top_k']),'--probabilities',str(path.resolve())],text=True).strip())
    expected=np.fromfile(a.reference/c['files']['probabilities']['file'],np.float32);actual=np.fromfile(path,np.float32)
    finite=expected.shape==actual.shape and np.isfinite(actual).all()
    maximum=float(np.abs(expected.astype(np.float64)-actual).max()) if finite else float('inf')
    relative=float(np.linalg.norm(expected.astype(np.float64)-actual)/np.linalg.norm(expected.astype(np.float64))) if finite else float('inf')
    result=dict(name=c['name'],chosen=chosen,expected=c['chosen'],max_abs=maximum,relative_l2=relative,passed=bool(finite and maximum<=1e-6 and relative<=1e-5 and chosen==c['chosen']))
    report['cases'].append(result);print(c['name'],result['passed'])
report['passed']=all(c['passed'] for c in report['cases']);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(0 if report['passed'] else 1)
