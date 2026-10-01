"""Fair paired fixed-init benchmarks. Run when the shared CPU timing lease is free."""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
from rough_cmeans import fit, _step


def timed(f):
    t = time.perf_counter(); result = f(); return time.perf_counter() - t, result


def fixture(n, k, d, seed):
    rng = np.random.default_rng(seed)
    roots = rng.normal(0, 2.0, size=(k, d))
    X = np.ascontiguousarray(roots[np.arange(n) % k] + rng.normal(size=(n, d)))
    # Identical explicit initial centers for every implementation.
    init = X[rng.choice(n, k, replace=False)].copy()
    return X, init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--output', default='results/benchmark.json')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
        limit = threadpool_limits(limits=args.threads)
        pools = threadpool_info()
    except ImportError:
        limit = contextlib.nullcontext(); pools = []
    backends = ['naive', 'numpy'] + [b for b in ['scipy', 'numba'] if importlib.util.find_spec(b)]
    config = [(1024, 4, 8), (12000, 8, 16), (4096, 16, 64), (2048, 16, 256)]
    if args.quick: config = config[:2]
    rows = []
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ['rough_cmeans.py', '_numba_kernel.py', 'benchmark.py']}
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'numba'] if importlib.util.find_spec(name)}
    meta = dict(source_sha256=source_hashes, versions=versions, python=sys.version, numpy=np.__version__, platform=platform.platform(), machine=platform.machine(), processor=platform.processor(), threads=args.threads, threadpools=pools,
                repeats=args.repeats, backends=backends,
                semantics='d Euclidean; M=1{d^p <= alpha^p*dmin^p+beta^p}; U=M/sum_c M; center=sum_i U*x / sum_i U',
                distance_note='naive full broadcast + np.sum; others direct-difference squared distance (no dropped ||x||^2)',
                timing_note='Warm timings; imports/JIT excluded. Same X/init, max_iter, membership output, thread limit. Each backend warmed once. Paired rounds rotate backend order. Full fit includes fresh final assignment and cycle checks.',
                memory_note='naive broadcast O(NKD); block NumPy/SciPy O(B(K+D)+KD+NK/8) plus cycle masks; fused Numba O(KD+K+NK/8) plus cycle masks; requested final memberships O(NK) additional.')
    with limit:
        for index, (n, k, d) in enumerate(config):
            X, init = fixture(n, k, d, 104 + index)
            for p in [1., 2., 3.]:
                kw = dict(alpha=1.15, beta=.5, p=p, init=init, max_iter=60, block_size=2048, return_memberships=True)
                initial = {}
                # Warm the actual shape/path and verify before collecting timings.
                for backend in backends:
                    _, initial[backend] = timed(lambda b=backend: fit(X, k, backend=b, **kw))
                oracle = initial['naive']
                raw = {b: [] for b in backends}
                step_raw = {b: [] for b in backends}
                for rep in range(args.repeats):
                    order = backends[rep % len(backends):] + backends[:rep % len(backends)]
                    for backend in order:
                        dt, _ = timed(lambda b=backend: fit(X, k, backend=b, **kw))
                        raw[backend].append(dt)
                        dt, _ = timed(lambda b=backend: _step(X, init, 1.15, .5, p, b, 2048, False))
                        step_raw[backend].append(dt)
                for backend in backends:
                    q = initial[backend]
                    center_error = float(np.max(np.abs(q.centers - oracle.centers)))
                    parity = bool(np.array_equal(q.upper_memberships, oracle.upper_memberships) and np.array_equal(q.memberships, oracle.memberships) and np.allclose(q.centers, oracle.centers, rtol=2e-12, atol=2e-12))
                    row = dict(n=n,k=k,d=d,p=p,backend=backend,alpha=1.15,beta=.5,seed=104+index,block_size=2048,max_iter=60,return_memberships=True,
                               seconds=raw[backend],median_seconds=statistics.median(raw[backend]),step_seconds=step_raw[backend],step_median_seconds=statistics.median(step_raw[backend]),
                               speedup_vs_naive=statistics.median(raw['naive'])/statistics.median(raw[backend]),
                               step_speedup_vs_naive=statistics.median(step_raw['naive'])/statistics.median(step_raw[backend]),
                               correctness_pass=parity,center_max_abs_error=center_error,mask_mismatches=int(np.count_nonzero(q.upper_memberships != oracle.upper_memberships)),
                               n_iter=q.n_iter,oracle_n_iter=oracle.n_iter,converged=q.converged,stop_reason=q.stop_reason)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                output.write_text(json.dumps(dict(metadata=meta,results=rows),indent=2))
    if not all(row['correctness_pass'] for row in rows):
        raise SystemExit('Correctness mismatch; failed timings must not be promoted.')


if __name__ == '__main__': main()
