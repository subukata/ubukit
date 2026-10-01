"""Paired, interleaved, single-thread benchmark with graph-inclusive timings."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits, threadpool_info
from ubukit_rmcm import fit_rmcm, prepare_rmcm
from ubukit_rmcm.core import PreparedRMCM

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(name='small_sparse', n=1024, d=4, k=4, delta=.8, seed=41),
    dict(name='many_points', n=8000, d=3, k=12, delta=.45, seed=42),
    dict(name='high_dimension', n=2048, d=64, k=8, delta=5.5, seed=43),
    dict(name='many_clusters', n=3072, d=8, k=48, delta=1.7, seed=44),
    dict(name='isolated', n=4096, d=16, k=8, delta=0., seed=45),
    dict(name='dense_partial', n=4096, d=2, k=4, delta=1.8, seed=47),
    dict(name='full_graph', n=512, d=8, k=8, delta=100., seed=46),
]


def fixture(case):
    rng = np.random.default_rng(case['seed'])
    k, n, d = case['k'], case['n'], case['d']
    roots = rng.normal(0, 2, (k, d))
    X = roots[np.arange(n) % k] + rng.normal(0, .5, (n, d))
    # The explicit initial data indices, arrays, and random seed are shared.
    ids = rng.choice(n, k, replace=False)
    return np.ascontiguousarray(X), X[ids].copy(), ids


def timed(fn):
    t = time.perf_counter()
    value = fn()
    return time.perf_counter() - t, value


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list((ROOT / 'ubukit_rmcm').glob('*.py')) + [Path(__file__)])}


def summary(t):
    return {'raw_seconds': t, 'median_seconds': statistics.median(t),
            'min_seconds': min(t), 'max_seconds': max(t)}


def run(args):
    methods = ['numpy', 'csr', 'adjoint', 'numba'] if args.numba else ['numpy', 'csr', 'adjoint']
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes(),
                  versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'threadpoolctl']},
                  python=platform.python_version(), platform=platform.platform(),
                  cpu_model=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), 'unknown'),
                  threads=1, repeats=args.repeats, max_iter=args.max_iter, warmup=True,
                  threadpools=threadpool_info(), cases=[],
                  notes=['Graph construction, backend precomputation, validation/copying, iteration, and requested final dense memberships are included in end_to_end.',
                         'reused_graph_fit excludes graph/precompute, includes final dense memberships; step includes one hard assignment and weighted update, excludes final membership materialization for adjoint/numba.',
                         'All methods use the same cKDTree constructor, fixed initialization, stopping policy, float64 direct distances, full-neighborhood shortcut, thread count, and output flag.',
                         'graph_build includes input snapshot validation/copy and CSR graph construction; backend precompute is reported separately.',
                         'Randomized method order within each repeat; setup/data generation and numerical comparison outside timer. Numba JIT warmed; cold cost measured separately.',
                         'The NumPy baseline uses O(E+NK) bincount occupancy and BLAS weighted centers, not an artificially quadratic membership routine.',
                         'There is no third-party comparator claiming the identical user-specified algorithm. CSR is a strong materialized-R implementation.'])
    if args.numba:
        result['versions']['numba'] = importlib.metadata.version('numba')
    chooser = random.Random(12345)
    for case in CASES:
        if args.cases and case['name'] not in args.cases:
            continue
        X, init, ids = fixture(case)
        prepared = {}
        preparation = {m: [] for m in methods}
        # One common graph snapshot used by all amortized-iteration methods.
        graph_times = []
        for _ in range(args.repeats):
            dt, graph = timed(lambda: prepare_rmcm(X, case['delta'], backend='csr'))
            graph_times.append(dt)
        for m in methods:
            for _ in range(args.repeats):
                dt, p = timed(lambda m=m: PreparedRMCM(graph._X, graph._P, graph._degrees,
                                      graph.candidate_edges, graph.delta, m, graph.block_size))
                preparation[m].append(dt)
            prepared[m] = p
            p.fit(case['k'], init=init, max_iter=args.max_iter, return_memberships=True)
            fit_rmcm(X, case['k'], case['delta'], init=init, backend=m,
                     max_iter=args.max_iter, return_memberships=True)
        times = {m: {key: [] for key in ['end_to_end', 'reused_graph_fit', 'step', 'final_memberships']} for m in methods}
        outputs = {}
        reused_outputs = {}
        materialized_outputs = {}
        for repeat in range(args.repeats):
            order = methods.copy()
            chooser.shuffle(order)
            for m in order:
                p = prepared[m]
                dt, r = timed(lambda: fit_rmcm(X, case['k'], case['delta'], init=init, backend=m,
                                      max_iter=args.max_iter, return_memberships=True))
                times[m]['end_to_end'].append(dt)
                outputs[m] = r
                dt, reused = timed(lambda: p.fit(case['k'], init=init, max_iter=args.max_iter, return_memberships=True))
                times[m]['reused_graph_fit'].append(dt)
                reused_outputs[m] = reused
                dt, step = timed(lambda: p._step(init))
                times[m]['step'].append(dt)
                def materialize():
                    R = p._memberships(r.labels, case['k'])
                    return R.toarray() if sparse.issparse(R) else R
                dt, R = timed(materialize)
                times[m]['final_memberships'].append(dt)
                materialized_outputs[m] = R
        oracle = outputs['numpy']
        rows = []
        for m in methods:
            r = outputs[m]
            row = {'backend': m, 'n_iter': r.n_iter, 'stop_reason': r.stop_reason,
                   'cycle_length': r.cycle_length, 'center_max_abs_error': float(np.max(np.abs(r.centers - oracle.centers))),
                   'membership_max_abs_error': float(np.max(np.abs(r.memberships - oracle.memberships))),
                   'label_mismatches': int(np.count_nonzero(r.labels != oracle.labels)),
                   'correctness_pass': bool(np.array_equal(r.labels, oracle.labels) and
                       np.allclose(r.centers, oracle.centers, rtol=1e-11, atol=1e-12) and
                       np.allclose(r.memberships, oracle.memberships, rtol=1e-11, atol=1e-12) and
                       r.n_iter == oracle.n_iter and r.stop_reason == oracle.stop_reason and r.cycle_length == oracle.cycle_length and
                       np.array_equal(reused_outputs[m].labels, r.labels) and
                       np.array_equal(reused_outputs[m].centers, r.centers) and
                       np.array_equal(materialized_outputs[m], r.memberships)),
                   'precompute': summary(preparation[m])}
            for key, samples in times[m].items():
                row[key] = summary(samples)
                row[key]['speedup_vs_numpy'] = statistics.median(times['numpy'][key]) / statistics.median(samples)
                row[key]['speedup_vs_csr'] = statistics.median(times['csr'][key]) / statistics.median(samples)
            rows.append(row)
        entry = dict(**case, graph_build=summary(graph_times), n_edges=graph.n_edges,
                     mean_degree=graph.n_edges / len(X), min_degree=int(graph.degrees.min()), max_degree=int(graph.degrees.max()),
                     graph_csr_bytes=int(graph._P.data.nbytes + graph._P.indices.nbytes + graph._P.indptr.nbytes),
                     adjoint_extra_bytes=0 if graph._full else int(X.nbytes + len(X)*8),
                     data_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                     init_sha256=hashlib.sha256(init.tobytes()).hexdigest(), initial_indices=ids.tolist(),
                     output_memberships=True, results=rows)
        result['cases'].append(entry)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        for row in rows:
            print(json.dumps(dict(case=case['name'], backend=row['backend'], edges=graph.n_edges,
                                  iterations=row['n_iter'], reason=row['stop_reason'], correctness=row['correctness_pass'],
                                  total_ms=1000*row['end_to_end']['median_seconds'],
                                  total_speedup=row['end_to_end']['speedup_vs_numpy'],
                                  step_speedup=row['step']['speedup_vs_numpy'])), flush=True)
        if not all(row['correctness_pass'] for row in rows):
            raise RuntimeError('Cross-backend correctness mismatch: inspect JSON before making timing claims')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/benchmark.json')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--numba', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
