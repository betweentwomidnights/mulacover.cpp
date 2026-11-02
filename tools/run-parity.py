#!/usr/bin/env python3
"""Drive native teacher forcing and compare stages/logits/KV against frozen tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def metric(expected,actual,activation=False):
    if expected.shape!=actual.shape or not np.isfinite(actual).all():return dict(passed=False,reason="shape mismatch or nonfinite native output")
    delta=actual.astype(np.float64)-expected.astype(np.float64)
    maximum=float(np.abs(delta).max(initial=0))
    relative=float(np.linalg.norm(delta)/max(np.linalg.norm(expected.astype(np.float64)),1e-30))
    flat=maximum<=1e-3 and relative<=1e-4
    mixed=bool(np.all(np.abs(delta)<=1e-3+1e-6*np.abs(expected.astype(np.float64))) and relative<=1e-4)
    worst=int(np.argmax(np.abs(delta))) if delta.size else 0
    return dict(max_abs=maximum,relative_l2=relative,flat_absolute_gate_passed=flat,
                activation_mixed_tolerance_passed=mixed,
                worst_reference=float(expected[worst]) if delta.size else 0,
                worst_native=float(actual[worst]) if delta.size else 0,
                worst_reference_float32_ulp=float(abs(np.spacing(expected[worst]))) if delta.size else 0,
                passed=bool(mixed if activation else flat))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--cli",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--backend",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--cases",default="conditional,cfg-1.5")
    p.add_argument("--reuse",action="store_true")
    p.add_argument("--ggml-source",type=Path,required=True)
    a=p.parse_args()
    m=json.loads((a.reference/"manifest.json").read_text())
    if m['generator_dtype']!='float32':raise ValueError("This first native baseline requires FP32 oracle tensors")
    pin='07b77b0112cc30861b49116b057bac33e819ee06'
    git_env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    revision=subprocess.check_output(['git','-C',str(a.ggml_source),'rev-parse','HEAD'],text=True,env=git_env).strip()
    dirty=subprocess.check_output(['git','-C',str(a.ggml_source),'status','--porcelain'],text=True,env=git_env).strip()
    if revision!=pin or dirty:raise ValueError('Parity requires the unchanged HeartCodec baseline GGML pin')
    report=dict(reference_manifest_sha256=hashlib.sha256((a.reference/"manifest.json").read_bytes()).hexdigest(),
        upstream_revision=m['upstream_revision'],checkpoint_revision=m['models']['generator']['revision'],
        ggml_revision=revision,ggml_worktree_clean=True,
        backend=a.backend,gates=dict(logits_and_cache_max_abs=1e-3,relative_l2=1e-4,
            activation_atol=1e-3,activation_rtol=1e-6,
            activation_policy='elementwise abs(delta)<=atol+rtol*abs(reference), plus relative L2; flat absolute diagnostic retained'),cases=[])
    def load(root,item):
        data=(root/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError("Oracle hash mismatch: "+item['file'])
        return np.frombuffer(data,np.dtype(item['stored_dtype'])).reshape(item['shape'])
    prepared={key:load(a.reference,item) for key,item in m['preprocessing'].items() if key.startswith('input.')}
    for case in m['cases']:
        if case['name'] not in a.cases.split(','):continue
        root=a.output/case['name'];native=root/'native';inp=root/'input';inp.mkdir(parents=True,exist_ok=True)
        expected_root=a.reference/case['name']
        teacher=load(expected_root,case['files']['teacher_frames'])
        mapping={'tokens':('input.tokens','i32'), 'mask':('input.tokens_mask','u8'),
                 'qwen':('input.qwen_embedding','f32'),'muq':('input.muq_embedding','f32'),
                 'piano':('input.pianoroll','f32'),'drum':('input.drum_pianoroll','f32'),
                 'chord':('input.chord','f32'),'context-mask':('input.context_mask','u8')}
        dtypes={'i32':np.int32,'u8':np.uint8,'f32':np.float32}
        for name,(key,extension) in mapping.items():np.ascontiguousarray(prepared[key][0],dtype=dtypes[extension]).tofile(inp/(name+'.'+extension))
        np.ascontiguousarray(teacher,dtype=np.int32).tofile(inp/'teacher.i32')
        command=[str(a.cli.resolve()),'--model',str(a.model.resolve()),'--input',str(inp.resolve()),'--output',str(native.resolve()),
            '--prompt-frames',str(prepared['input.tokens'].shape[1]),'--symbolic-frames',str(prepared['input.pianoroll'].shape[1]),
            '--qwen-index',str(prepared['input.qwen_indices'][0]),'--muq-index',str(prepared['input.muq_indices'][0]),
            '--backend',a.backend,'--guidance',str(case['guidance'])]
        if not a.reuse:
            env=dict(os.environ,GGML_CUDA_TF32='0')
            with (root/'stdout.txt').open('w') as out,(root/'stderr.txt').open('w') as err:
                run=subprocess.run(command,stdout=out,stderr=err,env=env)
            if run.returncode:raise RuntimeError(f"Native CLI failed ({run.returncode}): {(root/'stderr.txt').read_text()[-4000:]}")
        results={};exact_replay=True
        for name,item in case['files'].items():
            if name=='teacher_frames':continue
            expected=load(expected_root,item).reshape(-1)
            path=native/(name+'.f32')
            actual=np.fromfile(path,np.float32)
            replay=np.fromfile(native/'replay'/(name+'.f32'),np.float32)
            exact_replay &= np.array_equal(actual,replay)
            results[name]=metric(expected,actual,activation=('.stage.' in name or '.local.' in name))
            if '.logits.' in name or '.raw_logits.' in name:
                results[name]['top1_equal']=bool(np.array_equal(expected.reshape(-1,8197).argmax(-1),actual.reshape(-1,8197).argmax(-1)))
        stats=json.loads((native/'stats.json').read_text())
        if stats['ggml_commit']!=pin[:8]:raise ValueError('Linked GGML binary does not report the clean HeartCodec baseline pin')
        result=dict(name=case['name'],frames=case['frames'],branches=case['batch'],tensors=results,
                    native_reset_replay_bit_exact=bool(exact_replay),stats=stats,
                    flat_absolute_diagnostic_failures=[n for n,v in results.items() if not v.get('flat_absolute_gate_passed',False)],
                    passed=bool(exact_replay and stats['cache_tail_zero'] and all(v['passed'] for v in results.values())))
        report['cases'].append(result)
        failures=[(n,v) for n,v in results.items() if not v['passed']]
        print(case['name'],'PASS' if result['passed'] else 'FAIL',f"{len(results)} tensors; {len(failures)} failures",flush=True)
        for name,value in failures[:12]:print(name,value,flush=True)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=bool(report['cases'] and all(c['passed'] for c in report['cases']))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
