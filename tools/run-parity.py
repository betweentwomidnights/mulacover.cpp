#!/usr/bin/env python3
"""Drive native teacher forcing and compare stages/logits/KV against frozen tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from ggml_profile import PIN, validate_checkout


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


def logit_drift(expected,actual,teacher):
    """Distribution diagnostics, not a quantization acceptance gate."""
    a=expected.astype(np.float64).reshape(-1,8197)
    b=actual.astype(np.float64).reshape(-1,8197)
    def logsoftmax(x):
        shifted=x-x.max(axis=-1,keepdims=True)
        return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
    la,lb=logsoftmax(a),logsoftmax(b)
    top_a,top_b=a.argmax(-1),b.argmax(-1)
    k=32
    sets_a=np.argpartition(a,-k,axis=-1)[:,-k:]
    sets_b=np.argpartition(b,-k,axis=-1)[:,-k:]
    overlap=[len(set(x).intersection(y))/k for x,y in zip(sets_a,sets_b)]
    teacher_delta=lb[:,teacher]-la[:,teacher]
    return dict(top1_agreement=float(np.mean(top_a==top_b)),top32_overlap=float(np.mean(overlap)),
                kl_reference_to_native=float(np.mean(np.sum(np.exp(la)*(la-lb),axis=-1))),
                teacher_log_probability_delta_mean=float(teacher_delta.mean()),
                teacher_log_probability_delta_max_abs=float(np.abs(teacher_delta).max()))


def run_native(command,out,err,env,sample_memory):
    samples=[];stop=threading.Event()
    def sample():
        while not stop.is_set():
            try:
                run=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                if run.returncode==0:
                    used,total=map(int,run.stdout.strip().split(','));samples.append((used,total))
            except (OSError,ValueError,subprocess.TimeoutExpired):pass
            stop.wait(0.2)
    monitor=threading.Thread(target=sample,daemon=True)
    if sample_memory:monitor.start()
    started=time.monotonic()
    try:run=subprocess.run(command,stdout=out,stderr=err,env=env)
    finally:
        stop.set()
        if sample_memory:monitor.join(timeout=3)
    memory=dict(samples=len(samples),interval_seconds=0.2,scope='nvidia-smi physical GPU 0 total device usage, includes desktop/other processes')
    if samples:memory.update(first_used_mib=samples[0][0],peak_used_mib=max(s[0] for s in samples),total_mib=samples[0][1])
    memory['process_wall_seconds']=time.monotonic()-started
    return run,memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--cli",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--backend",choices=("cpu","cuda","vulkan","metal"),default="cuda")
    p.add_argument("--cases",default="conditional,cfg-1.5")
    p.add_argument("--reuse",action="store_true")
    p.add_argument("--ggml-source",type=Path,required=True)
    p.add_argument("--capacity",type=int,default=8192)
    p.add_argument("--measure-quantization",action="store_true",help="Measure drift; FP32 gates remain diagnostics and are not relaxed")
    p.add_argument("--sample-device-memory",action="store_true")
    a=p.parse_args()
    m=json.loads((a.reference/"manifest.json").read_text())
    if m['generator_dtype']!='float32':raise ValueError("This first native baseline requires FP32 oracle tensors")
    pin=PIN
    revision=validate_checkout(a.ggml_source)
    report=dict(reference_manifest_sha256=hashlib.sha256((a.reference/"manifest.json").read_bytes()).hexdigest(),
        upstream_revision=m['upstream_revision'],checkpoint_revision=m['models']['generator']['revision'],
        ggml_revision=revision,ggml_worktree_clean=True,shared_ggml_baseline=pin,
        unadopted_review_candidate=False,
        backend=a.backend,cache_capacity=a.capacity,evaluation_mode='quantization_measurement' if a.measure_quantization else 'fp32_parity',
        quantization_accuracy_accepted=False if a.measure_quantization else None,
        gates=dict(logits_and_cache_max_abs=1e-3,relative_l2=1e-4,
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
            '--backend',a.backend,'--guidance',str(case['guidance']),'--capacity',str(a.capacity),'--frames',str(case['frames'])]
        if 'checkpoints' in case:command+=['--dump-frames',','.join(map(str,case['checkpoints']))]
        if 'experimental_tempo' in case:
            clock=case['experimental_tempo']
            command+=['--experimental-bpm',str(clock['bpm']),'--tempo-width-beats',str(clock['width_beats']),'--tempo-offset-beats',str(clock['offset_beats'])]
            if clock.get('local_window'):command+=['--tempo-local-window','1']
        if 'sampling_noise' in case['files']:
            noise=load(expected_root,case['files']['sampling_noise'])
            np.ascontiguousarray(noise,np.float32).tofile(inp/'sampling-noise.f32')
            command+=['--sampling-noise',str((inp/'sampling-noise.f32').resolve())]
        memory_sampling=None
        if not a.reuse:
            env=dict(os.environ,GGML_CUDA_TF32='0',GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto')
            with (root/'stdout.txt').open('w') as out,(root/'stderr.txt').open('w') as err:
                run,memory_sampling=run_native(command,out,err,env,a.sample_device_memory and a.backend=='cuda')
            if run.returncode:raise RuntimeError(f"Native CLI failed ({run.returncode}): {(root/'stderr.txt').read_text()[-4000:]}")
        results={};exact_replay=True
        for name,item in case['files'].items():
            if name in ('teacher_frames','sampling_noise'):continue
            expected=load(expected_root,item).reshape(-1)
            path=native/(name+'.f32')
            actual=np.fromfile(path,np.float32)
            replay=np.fromfile(native/'replay'/(name+'.f32'),np.float32)
            exact_replay &= np.array_equal(actual,replay)
            results[name]=metric(expected,actual,activation=('.stage.' in name or '.local.' in name))
            if '.logits.' in name or '.raw_logits.' in name:
                results[name]['top1_equal']=bool(np.array_equal(expected.reshape(-1,8197).argmax(-1),actual.reshape(-1,8197).argmax(-1)))
                if expected.shape==actual.shape and np.isfinite(actual).all():
                    frame=int(name.split('.')[0][5:]);head=int(name.rsplit('.',1)[1])
                    results[name]['distribution_drift']=logit_drift(expected,actual,int(teacher[frame,head]))
        stats=json.loads((native/'stats.json').read_text())
        if stats['ggml_commit']!=revision[:8]:raise ValueError('Linked GGML binary does not report the requested clean GGML revision')
        if stats.get('cache_capacity',8192)!=a.capacity:raise ValueError('Native output cache capacity does not match the requested run')
        execution_ok=bool(exact_replay and stats['cache_tail_zero'] and all('reason' not in v for v in results.values()))
        result=dict(name=case['name'],frames=case['frames'],branches=case['batch'],tensors=results,execution_passed=execution_ok,
                    sampling_with_injected_noise='sampling_noise' in case['files'],
                    device_memory_sampling=memory_sampling,
                    native_reset_replay_bit_exact=bool(exact_replay),stats=stats,
                    flat_absolute_diagnostic_failures=[n for n,v in results.items() if not v.get('flat_absolute_gate_passed',False)],
                    passed=bool(exact_replay and stats['cache_tail_zero'] and all(v['passed'] for v in results.values())))
        report['cases'].append(result)
        failures=[(n,v) for n,v in results.items() if not v['passed']]
        print(case['name'],'MEASURED' if a.measure_quantization and execution_ok else 'PASS' if result['passed'] else 'FAIL',f"{len(results)} tensors; {len(failures)} FP32 gate failures",flush=True)
        if not a.measure_quantization:
            for name,value in failures[:12]:print(name,value,flush=True)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=bool(report['cases'] and all(c['passed'] for c in report['cases']))
    report['execution_passed']=bool(report['cases'] and all(c['execution_passed'] for c in report['cases']))
    report['fp32_parity_passed']=report['passed']
    report['model_file_bytes']=a.model.stat().st_size
    digest=hashlib.sha256()
    with a.model.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):digest.update(chunk)
    report['model_sha256']=digest.hexdigest()
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if (report['execution_passed'] if a.measure_quantization else report['passed']) else 1)


if __name__=='__main__':main()
