"""Isolated restart-v2 metric timings. Run only in a granted serial slot.

Example: python -m restart_v2.metrics_candidate.benchmark --n 2000 --d 64 \
  --k 15 --threads 1 --repeats 3 --output results/metrics_v2_2k_t1.jsonl
"""
import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=2000);p.add_argument('--d',type=int,default=64)
    p.add_argument('--dataset',choices=['random','digits'],default='random')
    p.add_argument('--k',type=int,nargs='+',default=[15]);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--dtype',choices=['float32','float64'],default='float64')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--seed',type=int,default=20261001)
    p.add_argument('--methods',nargs='+',default=['sklearn','numpy','numba','sqrt_numba','sqrt_numpy'])
    p.add_argument('--block-rows',type=int,default=256)
    p.add_argument('--max-scratch-bytes',type=int,default=32*2**20)
    p.add_argument('--max-distance-bytes',type=int,default=2*2**30)
    p.add_argument('--output',required=True);p.add_argument('--worker')
    return p.parse_args()


def worker(a):
    # The orchestrator sets affinity/env before exec, hence before package imports.
    import gc
    import numpy as np
    from sklearn.manifold import trustworthiness
    from threadpoolctl import threadpool_limits,threadpool_info
    from . import joint_quality
    started_utc = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=a.threads):
        def invoke(X,Y,ks):
            if a.worker=='sklearn':
                return [(trustworthiness(X,Y,n_neighbors=k),trustworthiness(Y,X,n_neighbors=k)) for k in ks]
            backend,method=a.worker,None
            if a.worker in ('numpy_broadcast','numpy_sortsearch'):
                backend='numpy';method=a.worker.removeprefix('numpy_')
            return joint_quality(X,Y,ks,backend=backend,threads=a.threads,
                block_rows=a.block_rows,rank_method=method,max_scratch_bytes=a.max_scratch_bytes,
                max_distance_bytes=a.max_distance_bytes)
        rng=np.random.default_rng(a.seed)
        tiny=rng.normal(size=(17,5)).astype(a.dtype)
        start=time.perf_counter();invoke(tiny,tiny[:,:2],[2]);warmup=time.perf_counter()-start
        rng=np.random.default_rng(a.seed)
        if a.dataset == 'digits':
            from sklearn.datasets import load_digits
            data=load_digits().data
            if a.n > len(data):raise ValueError('digits contains only 1797 rows; set --n <= 1797')
            X=np.array(data[:a.n],dtype=a.dtype)
            del data
        else:
            X=rng.normal(size=(a.n,a.d)).astype(a.dtype)
        W=rng.normal(size=(X.shape[1],2)).astype(a.dtype)
        # Fix generator arithmetic across 1/9-thread campaigns; only timed work changes resources.
        with threadpool_limits(limits=1):
            Y=(X@W/np.sqrt(X.shape[1])).astype(a.dtype)
        fn=lambda:invoke(X,Y,a.k)
        gc.collect();start=time.perf_counter();result=fn();cold=time.perf_counter()-start
        times=[]
        for repeat in range(a.repeats):
            gc.collect();start=time.perf_counter();result=fn();times.append(time.perf_counter()-start)
        completed_utc = datetime.now(timezone.utc).isoformat()
        if a.worker=='sklearn':scores=result;penalties=None
        else:
            scores=[(q.trustworthiness,q.continuity) for q in result]
            penalties=[(q.trustworthiness_penalty,q.continuity_penalty) for q in result]
        source=Path(__file__).parent
        version={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','threadpoolctl']}
        for name in ['numba','llvmlite']:
            try:version[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:version[name]=None
        med=statistics.median(times)
        print(json.dumps({'campaign':'restart_v2','started_utc':started_utc,'timed_trials_completed_utc':completed_utc,'method':a.worker,'dataset':a.dataset,'n':len(X),'d':X.shape[1],'k':a.k,'seed':a.seed,
            'dtype_X':str(X.dtype),'dtype_Y':str(Y.dtype),'threads':a.threads,
            'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'rank_block_rows':a.block_rows,'max_rank_scratch_bytes':a.max_scratch_bytes,
            'warmup_s':warmup,'cold_size_s':cold,'times_s':times,'median_s':med,
            'mad_s':statistics.median(abs(x-med) for x in times),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'scores':scores,'penalties':penalties,'versions':version,
            'numba_imported':any(k=='numba' or k.startswith('numba.') for k in sys.modules),
            'input_sha256':{'X':hashlib.sha256(X.tobytes()).hexdigest(),'Y':hashlib.sha256(Y.tobytes()).hexdigest()},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
            'threadpools':threadpool_info()}))


def main():
    a=parse()
    if a.worker:return worker(a)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    available=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    affinity=available[:a.threads] if available else None
    def prepare_child():
        if affinity is not None:os.sched_setaffinity(0,affinity)
    rows=[]
    for method in a.methods:
        env={**os.environ,'OPENBLAS_NUM_THREADS':str(a.threads),'OMP_NUM_THREADS':str(a.threads),
             'NUMBA_NUM_THREADS':str(a.threads),'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0','LOKY_MAX_CPU_COUNT':str(a.threads)}
        proc=subprocess.run([sys.executable,'-m','restart_v2.metrics_candidate.benchmark',*sys.argv[1:],'--worker',method],
            capture_output=True,text=True,env=env,preexec_fn=prepare_child if affinity is not None else None)
        if proc.returncode:
            print(proc.stdout);print(proc.stderr,file=sys.stderr);raise SystemExit(proc.returncode)
        row=json.loads(proc.stdout.strip().splitlines()[-1]);rows.append(row)
        with output.open('a') as f:f.write(json.dumps(row)+'\n')
        print(f"{method:18s} {row['median_s']:.6f}s ± {row['mad_s']:.6f}  RSS {row['peak_rss_bytes']/2**20:.1f}MiB",flush=True)
    baseline=next((r for r in rows if r['method']=='sklearn'),None)
    reference=next((r for r in rows if r['method']=='numpy'),next((r for r in rows if r['penalties'] is not None),None))
    summary=[]
    for row in rows:
        s={**row,'speedup_vs_sklearn':baseline['median_s']/row['median_s'] if baseline else None,
           'exact_scores_match_sklearn':row['scores']==baseline['scores'] if baseline else None,
           'exact_penalties_match_numpy':row['penalties']==reference['penalties'] if reference and row['penalties'] is not None else None,
           'inputs_match_baseline':row['input_sha256']==baseline['input_sha256'] if baseline else None}
        summary.append(s)
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    output.with_suffix('.run.json').write_text(json.dumps({'summary_completed_utc':datetime.now(timezone.utc).isoformat(),
        'raw_output':str(output),'methods':[r['method'] for r in rows],
        'summary_checks_pass':not any(r['exact_scores_match_sklearn'] is False or r['exact_penalties_match_numpy'] is False or r['inputs_match_baseline'] is False for r in summary)},indent=2)+'\n')
    for row in summary:
        print(row['method'],row['speedup_vs_sklearn'],'scores',row['exact_scores_match_sklearn'],'penalties',row['exact_penalties_match_numpy'])
    if baseline and any(r['exact_scores_match_sklearn'] is False or r['exact_penalties_match_numpy'] is False or r['inputs_match_baseline'] is False for r in summary):
        raise SystemExit(2)

if __name__=='__main__':main()
