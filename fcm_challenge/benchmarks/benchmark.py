#!/usr/bin/env python3
"""Reproducible FCM API comparisons; never run alongside other benchmarks."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_info, threadpool_limits
from ubukit_fcm import fit_fcm

CASES = {
    'tiny_separated': (300, 2, 3, 0.35, 5.0),
    'small_overlap': (2000, 8, 5, 2.5, 1.0),
    'large_n': (100000, 8, 8, 0.7, 4.0),
    'high_d': (10000, 128, 8, 0.7, 4.0),
    'many_clusters': (10000, 16, 64, 0.5, 4.0),
    'large_overlap': (20000, 16, 8, 2.5, 1.0),
}
BACKENDS = ['numpy', 'scipy', 'blas', 'numba', 'numba_parallel', 'scikit-fuzzy']


def data(name):
    n, d, k, noise, separation = CASES[name]
    seed = 718 + list(CASES).index(name)
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * separation
    labels = np.arange(n) % k
    rng.shuffle(labels)
    x = np.ascontiguousarray(centers[labels] + rng.normal(size=(n, d)) * noise)
    u = rng.random((n, k))
    u /= u.sum(axis=1, keepdims=True)
    return x, u, labels


def run(backend, x, u, *, max_iter, tol, threads, m):
    if backend != 'scikit-fuzzy':
        return fit_fcm(x, init=u, max_iter=max_iter, tol=tol, backend=backend, threads=threads, m=m)
    from skfuzzy.cluster import cmeans
    with threadpool_limits(limits=threads, user_api='blas'):
        c, uu, u0, dist, history, n_iter, fpc = cmeans(x.T, u.shape[1], m, tol, max_iter, init=u.T)
        objective = 0.
        for start in range(0, len(x), 8192):
            block = uu[:, start:start+8192].T
            objective += float(np.einsum('ij,ij->', block**m, cdist(x[start:start+8192], c, 'sqeuclidean')))
        uu = uu.T.copy()
        return dict(centers=c, membership=uu, labels=uu.argmax(axis=1), objective=objective,
                    fpc=float(fpc), n_iter=n_iter, converged=(False if tol == 0 else True if n_iter < max_iter else None), delta=None,
                    reported_previous_membership_objective=float(history[-1]))


def summary(result, truth, reference):
    return dict(objective=result['objective'], fpc=result['fpc'], n_iter=result['n_iter'],
                converged=result['converged'], delta=result['delta'],
                ari_truth=float(adjusted_rand_score(truth, result['labels'])),
                ari_reference=float(adjusted_rand_score(reference['labels'], result['labels'])),
                objective_relative_error=abs(result['objective']-reference['objective']) / max(abs(reference['objective']), 1e-300),
                membership_max_abs_error=float(np.max(np.abs(result['membership'] - reference['membership']))),
                centers_max_abs_error=float(np.max(np.abs(result['centers'] - reference['centers']))))


def metadata():
    versions = {}
    for package in ['numpy', 'scipy', 'numba', 'llvmlite', 'scikit-fuzzy', 'scikit-learn', 'threadpoolctl', 'psutil']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), None) if Path('/proc/cpuinfo').exists() else None
    return dict(python=sys.version, platform=platform.platform(), versions=versions,
                cpu_model=cpu_model, cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
                threadpools=threadpool_info(), command=sys.argv,
                source_sha256={str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).parent.parent/'ubukit_fcm').glob('*.py')})


def benchmark(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(metadata=metadata(), protocol=dict(threads=args.threads, repeats=args.repeats, fixed_iterations=args.fixed, max_iterations=args.max_iter, tol=args.tol, m=args.m,
                  order='One warmup per method/case/mode, then seeded shuffled interleaving per repeat; perf_counter wall time; median',
                  timing_scope='Full fit plus common-output adapter incl validation/copies, supplied U, final objective and labels; scikit-fuzzy is cmeans plus adapter, not bare cmeans; data generation/quality checks excluded',
                  stop='absolute Frobenius membership delta < tol; tol=0 runs exact iteration count',
                  baseline='Straightforward NumPy broadcast distance tensor, O(NK) inverse-power membership normalization',
                  finalizer='All methods return final centers/membership pair objective, recomputed by identical chunked direct SciPy distances'), cases=[])
    # Pre-import all dependencies; import/JIT startup is a separate experiment.
    import numba
    import skfuzzy
    for name in args.cases:
        x, u, truth = data(name)
        dimensions = CASES[name][:3]
        for mode in args.modes:
            params = dict(max_iter=args.fixed if mode=='fixed' else args.max_iter,
                          tol=0 if mode=='fixed' else args.tol, threads=args.threads, m=args.m)
            item = dict(name=name, mode=mode, shape=dict(zip(['n','d','k'],dimensions)),
                        X_sha256=hashlib.sha256(x.tobytes()).hexdigest(), init_sha256=hashlib.sha256(u.tobytes()).hexdigest(),
                        methods={})
            reference = None
            for backend in args.backends:
                warm = run(backend, x, u, **params)
                if reference is None:
                    reference = warm
                item['methods'][backend] = dict(seconds=[], quality=summary(warm, truth, reference))
            ordering = random.Random(734 + list(CASES).index(name) * 10 + (1 if mode=='fixed' else 0))
            for repeat in range(args.repeats):
                order = list(args.backends)
                ordering.shuffle(order)
                for backend in order:
                    start = time.perf_counter()
                    answer = run(backend, x, u, **params)
                    elapsed = time.perf_counter() - start
                    item['methods'][backend]['seconds'].append(elapsed)
                    if answer['n_iter'] != item['methods'][backend]['quality']['n_iter']:
                        raise RuntimeError('iteration count changed between repetitions')
                print(f'{name} {mode} repeat {repeat+1}/{args.repeats}', flush=True)
            baseline = statistics.median(item['methods'][args.backends[0]]['seconds'])
            for backend, values in item['methods'].items():
                values['median_seconds'] = statistics.median(values['seconds'])
                values['min_seconds'] = min(values['seconds'])
                values['max_seconds'] = max(values['seconds'])
                values['speedup_vs_first'] = baseline / values['median_seconds']
            result['cases'].append(item)
            output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'case':name,'mode':mode,'seconds':{b:round(v['median_seconds'],5) for b,v in item['methods'].items()}}), flush=True)
    return result


def auxiliary(args):
    # Isolated fresh interpreters: captures JIT/cache/import startup and high-water RSS.
    root = Path(__file__).resolve().parent.parent
    script = r'''
import json, os, resource, time
s=time.perf_counter()
import numpy as np
from benchmarks.benchmark import data, run
imports=time.perf_counter()-s
x,u,truth=data(os.environ['FCM_CASE'])
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
s=time.perf_counter(); r=run(os.environ['FCM_BACKEND'],x,u,max_iter=10,tol=0,threads=int(os.environ['FCM_THREADS']),m=2.); first=time.perf_counter()-s
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
s=time.perf_counter(); r=run(os.environ['FCM_BACKEND'],x,u,max_iter=10,tol=0,threads=int(os.environ['FCM_THREADS']),m=2.); second=time.perf_counter()-s
print(json.dumps(dict(import_seconds=imports,first_fit_seconds=first,second_fit_seconds=second,peak_rss_mib=peak/1024,peak_minus_pre_fit_highwater_mib=(peak-before)/1024,objective=r['objective'])))
'''
    records=[]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        for backend in args.backends:
            with tempfile.TemporaryDirectory(prefix='fcm_numba_') as cache:
                env=os.environ.copy()
                env.update(FCM_CASE=name,FCM_BACKEND=backend,FCM_THREADS=str(args.threads),NUMBA_CACHE_DIR=cache)
                begin=time.perf_counter()
                proc=subprocess.run([sys.executable,'-c',script],cwd=root,env=env,check=True,text=True,capture_output=True)
                record=json.loads(proc.stdout.strip().splitlines()[-1])
                record.update(case=name,backend=backend,process_wall_seconds=time.perf_counter()-begin)
                records.append(record)
                print(json.dumps(record),flush=True)
                Path(args.output).write_text(json.dumps(dict(metadata=metadata(),threads=args.threads,note='Each cell is a new process with a new empty Numba cache; RSS is Linux ru_maxrss high-water, not live allocation; 10 fixed iterations; imports include benchmark/scikit-learn helper imports',records=records),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/benchmark_t1.json')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--fixed',type=int,default=30)
    p.add_argument('--max-iter',type=int,default=300)
    p.add_argument('--tol',type=float,default=1e-5)
    p.add_argument('--m',type=float,default=2.)
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--modes',nargs='+',choices=['fixed','converged'],default=['fixed','converged'])
    p.add_argument('--backends',nargs='+',choices=BACKENDS,default=BACKENDS)
    p.add_argument('--auxiliary',action='store_true')
    args=p.parse_args()
    if args.auxiliary:
        auxiliary(args)
    else:
        benchmark(args)

if __name__=='__main__':
    main()
