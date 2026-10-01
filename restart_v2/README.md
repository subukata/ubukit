# Restart-v2 portable acceleration

This is newly reconstructed and freshly tested research code. It is not the
lost final package, and no earlier performance numbers are reused as v2 results.
Current validation covers Linux x86-64/Python3.12 and the recorded dependency
versions. Production reliability and other operating systems are not verified.

## Install and use

```
python -m pip install .
python -m pip install '.[numba]'       # optional JIT kernels/finalizer
python -m unittest discover -s tests -v
```

NumPy, SciPy, sklearn and threadpoolctl are standard dependencies. No compiler
or Numba is needed for the NumPy/SOM paths. `requirements-tested.txt` records the
complete measured environment, including optional Cython used by standalone
native controls. Root package import loads no numerical/JIT runtime.

```python
import numpy as np
from portable_accel import ExecutionPolicy, fit_kmeans, fit_som_olp, joint_quality
X = np.random.default_rng(4).normal(size=(300, 12))
R = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
policy = ExecutionPolicy(threads=4, block_rows=256, max_scratch_bytes=64*2**20)
km = fit_kmeans(X, X[:8], policy=policy)  # final labels and inertia included
som = fit_som_olp(X, R, gamma=.5, lam=1., backend='threadpool',
                  initializer='svd_lowrank', policy=policy)
quality = joint_quality(X, som['V'], ks=[5, 10], policy=policy)
```

## Explicit methods and contracts

- K-means: `numpy`, `numba` (register8), `numba_blas`, `numba_blas_vector`.
  `auto` chooses available Numba register8, otherwise NumPy. Initial centers
  are explicit, empty centers are retained, and core labels precede the final
  center update. NEW default `finalize=True` computes nearest-final-center
  labels and inertia inside the public fit. `core_labels` preserves the old
  assignment and `finalized_labels` reports the choice. `finalize=False` gives
  inertia=None. `finalizer='sklearn'` is default; explicit `'numba'` uses strict
  direct squared distances and first computed ties. The finalizers can differ
  near floating ties/large offsets; neither makes the whole algorithm a general
  sklearn KMeans substitute. BLAS near-tie repair remains a heuristic.
- SOM: original reference `cdist` remains default. Explicit `gemm_guarded`,
  `cdist_optimized`, `gemm_centered` and `threadpool` are available. Centered
  arithmetic/objective identities change rounding; guarded mode can use direct
  fallback. ThreadPool uses disjoint row work and BLAS=1 inside the executor.
  Original full-SVD/direct-P0 initialization is default. `initializer='svd_lowrank'`
  keeps the same full SVD/W0 and uses guarded low-rank P0 evaluation, with direct
  fallback possible. Requested initialization is reported. Final W/V preserve
  the original lagged update order. Optimized SOM options require explicit
  thread counts; original cdist supports inherited limits.
- Joint trustworthiness/continuity: `numpy`, `numba`, `sqrt_numpy`, `sqrt_numba`
  (alias `numba_sqrt`). `auto` chooses available Numba scan, otherwise NumPy
  broadcast. `rank_method` stays explicit. Every mode preserves full sklearn
  distance-call shape, float32/64, independent per-k neighbors and NumPy's actual
  argsort tie behavior. Only rank work is blocked. Results always form a list
  including scores and integer penalties. `max_distance_bytes` refuses an
  oversized full distance matrix; distance memory is still O(N²).

Availability defaults are not optimal-backend predictors. Native SOM controls
are separately included under `som_candidate`; their Linux/glibc build flags,
compiler requirement and measured scope do not apply to standard-dependency APIs.

## Shared processing, memory and threads

`prepare(X)` owns an immutable bytes-backed snapshot. Caller mutation cannot
stale its explicitly keyed dtype/preprocessing/numerical-contract caches for
norms, bounds and centered data. Raw inputs have no implicit identity cache.
`PreparedSOM` is a separate explicit full-SVD reuse API in 0.2.0a2. It owns
float64 X and leading factors from the same complete mean-centered SVD, pinned
to the preparation thread count. Every fit creates fresh W0/P0; gamma/lambda
changes do not cause reuse of fit state. Shape, original dtype or input changes
require a new object (the optional `assert_same_input` scans and checks them).
It does not cache neighbors or reuse SOM-centered norms for strict metrics.

```python
from portable_accel import PreparedSOM
prepared = PreparedSOM(X, max_rank=R.shape[1], threads=policy.threads)
a = prepared.fit(R, gamma=.5, lam=1., policy=policy)
b = prepared.fit(R, gamma=.8, lam=1.2, policy=policy)
```

Preparation is outside each `.fit` call, so charge it once when comparing a
whole parameter sweep. The measured MNIST70k three-fit workload improved from
40.0082s to33.0890s (1.2091x), including a NEW snapshot and full SVD in every
prepared workload. Setup median was4.1890s; owned snapshot/factors419.787MiB.
This is an amortized three-fit result, not a faster cold first-fit claim.
See `som_candidate/results/prepared_sweep70k/` and its report. The integrated
module adapts only imports/documentation/description metadata from the measured
candidate; tiny equality and single-SVD instrumentation verify the integration.
Sharing validation/policy code alone is not a performance gain.

`ExecutionPolicy` scopes/restores thread limits. BLAS/OpenMP controls are
process-wide; separate processes are needed for independent concurrent policies.
The default uses one thread. Scratch caps cover selected primary work buffers,
not total RSS. ThreadPool accounts for private and merged numerator/denominator
arrays plus per-worker B×M buffers. Output arrays, input/centering copies, SVD,
library work, direct-fallback temporaries, linear norms and rank/query outputs
are separate. The measured full-fit public SOM configuration uses9threads,
B256 and64MiB; this is a named-workspace budget, not an RSS promise.

## Evidence and checkpoints

The a1 ThreadPool integration passed12 test methods; a2 adds2 PreparedSOM methods
covering18 parameterized fits, immutable ownership, input identity and one-SVD reuse.
Fresh raw comparisons are in the candidate result folders, with their exact
scope: matched-output public k-means; strict metric candidates; SOM kernel-only
screens; and the later true-full-fit pair. `evidence/public_metrics/` is the
completed actual-public metrics comparison. `evidence/kmeans_scratch_matched/`
adds the same-window scratch NumPy baseline with final labels/inertia included. Candidate timing must not be
represented as an unmeasured installed-wrapper timing. Imports/JIT/first calls
and warm trials remain distinct. Sensitive differences are retained explicitly.

`evidence/` records environment, source provenance, integration tests and freezes.
Source checkpoints exclude datasets, venvs, native binaries and JIT caches.
Dataset generators/source metadata and hashes remain for reproduction. The
checkpoint3 packaging omission was corrected and superseded, then the corrected
baseline and integrated package were revalidated; the warning is retained.

Original SOM source: https://github.com/subukata/som-olp at
4361175b776987d65c348d0132b31d43505e1069. Its notice is in LICENSE-SOM.txt.
No external publication or GitHub push is performed.

`requirements-foundation.txt` pins the standard-dependency-only environment;
`requirements-tested.txt` additionally pins optional Numba/llvmlite/Cython.
Declared minimum package versions are resolver metadata, not a tested version
matrix. No project-wide publication license has been selected; retained
third-party notices are included in NOTICE.txt and LICENSE-SOM.txt.

The completed true-full-fit SOM comparison is documented in
som_candidate/FULL_FIT_REPORT.md and RESULTS_SUMMARY.json. It includes original
full fit versus the actual public ThreadPool/same-SVD-lowrank call,9threads,
B256/64MiB,3warm repetitions, both36iterations and atol=rtol=1e-8 comparisons.
The earlier candidate README sections preserve chronological kernel screens;
use the newer full-fit report for the final end-to-end scope.

The prior32-file measured public a1 source is preserved in
`evidence/measured_public_a1/`; a2 retains31 numeric modules byte-for-byte, adds
`som_prepared.py`, and changes only version/lazy-export metadata in `__init__.py`.
`evidence/final_a2_source_manifest.json` records this boundary. The stable a1
wheel/QA remains in `evidence/distribution/`; final a2 distribution QA is kept
separately, so historical wheel results are not silently attributed to a2.
