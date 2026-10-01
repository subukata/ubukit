"""Fresh-process import/JIT/startup and whole-process peak memory evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CHILD = r'''
import json, resource, time
start = time.perf_counter()
import numpy as np
from ubukit_rmcm import fit_rmcm
imported = time.perf_counter()
rng = np.random.default_rng(123)
X = rng.normal(size=(2048, 8))
init = X[rng.choice(len(X), 8, replace=False)].copy()
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = fit_rmcm(X, 8, 1.5, init=init, backend=BACKEND, max_iter=10)
first = time.perf_counter()
result2 = fit_rmcm(X, 8, 1.5, init=init, backend=BACKEND, max_iter=10)
end = time.perf_counter()
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(dict(backend=BACKEND, import_seconds=imported-start,
 first_fit_seconds=first-imported, second_fit_seconds=end-first,
 total_seconds=end-start, max_rss_kib=after, incremental_max_rss_kib=after-before,
 n_iter=result.n_iter, stop_reason=result.stop_reason, edges=result.n_edges)))
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/startup.json')
    parser.add_argument('--numba', action='store_true')
    args = parser.parse_args()
    rows = []
    for m in ['numpy', 'csr', 'adjoint'] + (['numba'] if args.numba else []):
        with tempfile.TemporaryDirectory(prefix='rmcm-empty-jit-cache-') as cache:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, '-c', 'BACKEND='+repr(m)+'\n'+CHILD],
                               text=True, capture_output=True, check=True, env=env)
            row = json.loads(p.stdout)
            rows.append(row)
            print(json.dumps(row), flush=True)
    output = dict(note='Linux fresh interpreters, empty separate Numba caches; OS filesystem cache not flushed. Whole-process peak RSS includes imports/JIT. First fit includes data generation (small) and graph/precompute; second fit includes graph/precompute again. max_iter=10 is a bounded startup workload, not a convergence claim.', results=rows)
    Path(args.output).write_text(json.dumps(output, indent=2)+'\n')
