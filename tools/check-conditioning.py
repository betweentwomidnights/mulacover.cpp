#!/usr/bin/env python3
"""Offline-only exact discrete and strict F32 frontend comparison; no gate changes."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
a=p.parse_args();manifest=json.loads((a.reference/'manifest.json').read_text());results=[]
for case in manifest['cases']:
    name=case['name'];ref=a.reference/name;native=a.native/name;run=json.loads((native/'run.json').read_text())
    discrete={file:(ref/file).read_bytes()==(native/file).read_bytes() for file in ['tokens.i32','mask.u8','style-ids.i32','normalized-lyrics.txt','normalized-tags.txt','muq.f32']}
    discrete.update({key:run[key]==case[key] for key in ['prompt_frames','qwen_index','muq_index']})
    floats={}
    for file in ['qwen.f32']+([] if case['style_skipped'] else ['style-hidden.f32','style-layer-0.f32','style-layer-1.f32','style-layer-14.f32','style-layer-28.f32']):
        left=np.fromfile(ref/file,np.float32).astype(np.float64);right=np.fromfile(native/file if file=='qwen.f32' else native/'stages'/file,np.float32).astype(np.float64)
        assert left.shape==right.shape
        maximum=float(np.max(np.abs(left-right)));relative=float(np.linalg.norm(left-right)/max(float(np.linalg.norm(left)),1e-30))
        floats[file]={'max_abs':maximum,'relative_l2':relative,'passed':bool(np.isfinite(right).all() and maximum<=1e-3 and relative<=1e-4)}
    if (ref/'qwen-stock.f32').exists():
        left=np.fromfile(ref/'qwen-stock.f32',np.float32).astype(np.float64);right=np.fromfile(native/'qwen.f32',np.float32).astype(np.float64)
        maximum=float(np.max(abs(left-right)));relative=float(np.linalg.norm(left-right)/max(float(np.linalg.norm(left)),1e-30))
        floats['stock_upstream_style']={'max_abs':maximum,'relative_l2':relative,'passed':bool(np.isfinite(right).all() and maximum<=1e-3 and relative<=1e-4)}
    results.append({'name':name,'discrete':discrete,'floating':floats,'passed':all(discrete.values()) and all(v['passed'] for v in floats.values())})
report={'reference':str(a.reference),'native':str(a.native),'upstream_revision':manifest['upstream_revision'],
        'style_revision':manifest['style_revision'],'versions':manifest['versions'],'reference_device':manifest.get('device','cuda'),
        'attention_oracle':manifest['attention'],'stock_control':manifest.get('stock_control'),
        'gates':{'max_abs':1e-3,'relative_l2':1e-4,'discrete':'exact bytes'},'cases':results,'passed':all(x['passed'] for x in results)}
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
