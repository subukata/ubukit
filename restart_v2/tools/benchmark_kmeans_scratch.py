"""NEW v2 scratch-NumPy versus accelerated matched-output fits; exclusive timing required.

A fresh process owns each fixture. It interleaves actual public portable fits
(sklean/Numba finalizers) and actual sklearn fits, retaining every sample.
No label finalization or output materialization occurs outside a fit timer.
Validation occurs after timing; a mismatch aborts and is never silently skipped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument('--case', choices=['low_d', 'high_d'])
p.add_argument('--threads', type=int, default=9)
p.add_argument('--repeats', type=int, default=7)
p.add_argument('--max-iter', type=int, default=20)
p.add_argument('--seed', type=int, default=20261001)
p.add_argument('--package-dir')
p.add_argument('--output', default='evidence/kmeans_scratch_matched')
p.add_argument('--budget-seconds', type=float, default=120.)
a = p.parse_args()

if a.case is None:
    start = time.monotonic()
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    records = []
    for case in ['low_d', 'high_d']:
        remaining = a.budget_seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError('matched benchmark budget exhausted')
        output = root / (case + '.json')
        cmd = [sys.executable, '-m', 'tools.benchmark_kmeans_scratch',
               '--case', case, '--threads', str(a.threads), '--repeats', str(a.repeats),
               '--max-iter', str(a.max_iter), '--seed', str(a.seed), '--output', str(output)]
        if a.package_dir:
            cmd += ['--package-dir', a.package_dir]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=remaining)
        print(r.stdout, end='', flush=True)
        if r.stderr:
            print(r.stderr, end='', file=sys.stderr, flush=True)
        if output.exists():
            records.append(json.loads(output.read_text()))
        summary = {'status': 'running', 'scope': 'NEW restart-v2 scratch NumPy and accelerated public matched-output end-to-end fits',
                   'wall_seconds': time.monotonic() - start, 'cases': records}
        if r.returncode or not records or records[-1]['status'] != 'matched':
            summary['status'] = 'failed_or_mismatched'
            (root / 'summary.json').write_text(json.dumps(summary, indent=2))
            raise SystemExit(1)
    summary['status'] = 'matched'; summary['wall_seconds'] = time.monotonic() - start
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    raise SystemExit(0)

# Fresh worker: set limits and path before importing any numerical runtime.
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = str(a.threads)
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['GOMP_SPINCOUNT'] = '0'
if hasattr(os, 'sched_setaffinity'):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < a.threads:
        raise RuntimeError('fewer affinity CPUs than requested thread budget')
    os.sched_setaffinity(0, available[:a.threads])
if a.package_dir:
    sys.path.insert(0, str(Path(a.package_dir).resolve()))
record = {'status': 'running', 'case': a.case, 'threads': a.threads, 'repeats': a.repeats,
          'max_iter': a.max_iter, 'seed': a.seed,
          'protocol': 'fresh process per case; 7 interleaved warm repetitions; all finalization inside fit',
          'wait_policy': 'PASSIVE', 'gomp_spincount': '0'}
try:
    import gc
    import platform
    import numpy as np
    import scipy
    import sklearn
    import numba
    import portable_accel
    from portable_accel import fit_kmeans, ExecutionPolicy
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits, threadpool_info
    package = Path(portable_accel.__file__).resolve().parent
    def sources():
        return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(package.rglob('*.py'))}
    before_sources = sources()
    policy = ExecutionPolicy(threads=a.threads, block_rows=256)
    rng = np.random.default_rng(a.seed)
    if a.case == 'low_d':
        n, d, k = 20000, 8, 16
        means = rng.normal(size=(k, d)) * 6.
        truth = np.arange(n) % k; rng.shuffle(truth)
        X = np.ascontiguousarray(means[truth] + rng.normal(size=(n, d)) * .7)
        backend = 'numba'
    else:
        n, d, k = 10000, 128, 64
        X = np.ascontiguousarray(rng.normal(size=(n, d)))
        backend = 'numba_blas_vector'
    init = X[rng.choice(n, k, replace=False)].copy()
    def fingerprint():
        return hashlib.sha256(memoryview(X).cast('B')).hexdigest() + ':' + hashlib.sha256(memoryview(init).cast('B')).hexdigest()
    input_before = fingerprint()
    def portable(finalizer, selected_backend=None):
        return fit_kmeans(X, init, max_iter=a.max_iter, backend=selected_backend or backend,
                          policy=policy, finalize=True, finalizer=finalizer)
    def sklearn_fit():
        with threadpool_limits(limits=a.threads):
            model = KMeans(n_clusters=k, init=init.copy(), n_init=1, max_iter=a.max_iter,
                           tol=0., algorithm='lloyd', copy_x=True).fit(X)
            return {'centers': model.cluster_centers_, 'labels': model.labels_,
                    'inertia': float(model.inertia_), 'n_iter': int(model.n_iter_),
                    'finalized_labels': True}
    methods = {'scratch_numpy_sklearn_finalizer': lambda: portable('sklearn', 'numpy'),
               'portable_numba_finalizer': lambda: portable('numba'),
               'sklearn_lloyd': sklearn_fit}
    first_seconds, first_outputs = {}, {}
    # Each first call is recorded independently; it is not a clean-install JIT claim.
    for name in ['sklearn_lloyd', 'scratch_numpy_sklearn_finalizer', 'portable_numba_finalizer']:
        t = time.perf_counter(); first_outputs[name] = methods[name](); first_seconds[name] = time.perf_counter() - t
    reference = first_outputs['sklearn_lloyd']
    def validate(out):
        labels_equal = bool(np.array_equal(out['labels'], reference['labels']))
        centers_close = bool(np.allclose(out['centers'], reference['centers'], rtol=1e-10, atol=1e-10))
        inertia_close = bool(np.isclose(out['inertia'], reference['inertia'], rtol=1e-10, atol=1e-10))
        count = np.bincount(np.asarray(out['labels']), minlength=k)
        nonempty = bool(np.all(count > 0))
        core_nonempty = bool(np.all(np.bincount(np.asarray(out.get('core_labels', out['labels'])), minlength=k) > 0))
        independent_inertia = float(np.sum((X - out['centers'][out['labels']]) ** 2))
        inertia_direct_close = bool(np.isclose(out['inertia'], independent_inertia, rtol=1e-10, atol=1e-10))
        fields = {'labels_equal': labels_equal, 'label_mismatches': int(np.count_nonzero(out['labels'] != reference['labels'])),
                  'centers_allclose': centers_close, 'center_max_abs': float(np.max(np.abs(out['centers'] - reference['centers']))),
                  'inertia_allclose': inertia_close, 'inertia_direct_allclose': inertia_direct_close,
                  'inertia': float(out['inertia']), 'independent_direct_inertia': independent_inertia,
                  'n_iter': int(out['n_iter']), 'n_iter_equal': int(out['n_iter']) == int(reference['n_iter']),
                  'all_final_clusters_nonempty': nonempty, 'all_core_clusters_nonempty': core_nonempty,
                  'finalized_labels': bool(out.get('finalized_labels', False))}
        fields['matched'] = all([labels_equal, centers_close, inertia_close, inertia_direct_close,
                                 fields['n_iter_equal'], nonempty, core_nonempty, fields['finalized_labels']])
        return fields
    validation = {name: validate(out) for name, out in first_outputs.items()}
    record.update(scratch_contract='Public numpy backend: feature-ordered direct squared distances, bincount update, B256; standard-dependency sklearn finalizer included. Not a globally optimal NumPy baseline.', driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), shape=[n, d], k=k, backend=backend, input_fingerprint=input_before,
                  initializer='same distinct random data rows for every method',
                  first_call_seconds=first_seconds, initial_validation=validation,
                  package_path=str(package), source_sha256=before_sources,
                  versions={'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'numba': numba.__version__})
    if not all(item['matched'] for item in validation.values()):
        record['status'] = 'initial_output_mismatch'
        raise RuntimeError('initial matched-output check failed; no warm ranking collected')
    samples = {name: [] for name in methods}; cpu_samples = {name: [] for name in methods}
    repetition_validation = {name: [] for name in methods}; order = list(methods)
    bitwise = {name: True for name in methods}
    rng_order = np.random.default_rng(a.seed + 71)
    for repetition in range(a.repeats):
        rng_order.shuffle(order)
        for name in order:
            gc.collect(); gc.disable(); tc = time.process_time(); t = time.perf_counter()
            try:
                out = methods[name]()
                elapsed = time.perf_counter() - t; cpu = time.process_time() - tc
            finally:
                gc.enable()
            samples[name].append(elapsed); cpu_samples[name].append(cpu)
            valid = validate(out); repetition_validation[name].append(valid)
            bitwise[name] &= bool(np.array_equal(out['labels'], first_outputs[name]['labels']) and
                                  np.array_equal(out['centers'], first_outputs[name]['centers']) and
                                  out['inertia'] == first_outputs[name]['inertia'])
            if not valid['matched']:
                record.update(status='warm_output_mismatch', seconds=samples,
                              repetition_validation=repetition_validation)
                raise RuntimeError('warm matched-output check failed; no mismatch skipped')
    record.update(status='matched', seconds=samples, cpu_seconds=cpu_samples,
                  median_seconds={name: float(np.median(values)) for name, values in samples.items()},
                  mad_seconds={name: float(np.median(np.abs(np.array(values) - np.median(values)))) for name, values in samples.items()},
                  repeat_bitwise_consistent=bitwise, repetition_validation=repetition_validation,
                  input_unchanged=input_before == fingerprint(), source_unchanged=before_sources == sources(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), threadpools=threadpool_info(),
                  time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if not record['input_unchanged'] or not record['source_unchanged']:
        record['status'] = 'mutation_or_source_change'
except Exception as exc:
    if record['status'] == 'running':
        record['status'] = 'error'
    record.update(error=repr(exc), traceback=traceback.format_exc())
output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2))
print(json.dumps({key: record.get(key) for key in ['case', 'status', 'median_seconds', 'initial_validation', 'error']}, indent=2), flush=True)
if record['status'] != 'matched':
    raise SystemExit(1)
