# UbuKit: portable fuzzy c-means candidate

A portable standard fuzzy c-means implementation. See the [repository quickstart](../README.md#fcm) for a complete runnable example.

## Install

```sh
python -m pip install .
# Optional JIT backend (no compiler toolchain needed where Numba wheels exist):
python -m pip install '.[numba]'
# Tests and reproducible comparisons:
python -m pip install '.[numba,test,benchmark]'
```

This is a pure-Python package with NumPy, SciPy, and threadpoolctl dependencies. Its default path needs no Numba, Cython, local C/C++ compiler, or platform-specific extension authored by UbuKit. It is intended to integrate into the existing portable UbuKit package after review. Only Linux/x86-64 has been tested; macOS/Windows are not verified.

## Usage

```python
from ubukit_fcm import fit_fcm, fit_fcm_numpy

result = fit_fcm(X, n_clusters=5, m=2, random_state=42)
centers = result['centers']       # (K, D)
u = result['membership']         # (N, K)
labels = result['labels']        # membership argmax
objective = result['objective']  # final returned pair, sum(u**m * squared_distance)

# For fair comparisons, explicitly share the same initial membership matrix.
reference = fit_fcm_numpy(X, init=U0, m=2, max_iter=30, tol=0)
fast = fit_fcm(X, init=U0, m=2, max_iter=30, tol=0, backend='blas')
jit = fit_fcm(X, init=U0, m=2, max_iter=30, tol=0, backend='numba')
```

## Algorithms and contracts

- Standard squared-Euclidean FCM: `C_j = sum_i U_ij**m X_i / sum_i U_ij**m`, then memberships proportional to `D²_ij**(-1/(m-1))`
- Float64 throughout; `m` may be any finite real value greater than 1
- Membership normalization is O(NK), including the NumPy scratch baseline. An artificially expensive O(NK²) ratio implementation is not used to inflate speedups
- `m=2` uses square/reciprocal specializations
- Distances retain the `||x||²` term; dropping it changes fuzzy memberships
- Exactly zero distances split a point's membership equally among exactly coincident centers
- Each iteration computes centers from the old memberships, then updates memberships. The returned centers are from the last center step, and returned memberships from its subsequent membership step. There is no hidden final center update
- Stop at `||U_new-U_old||_F < tol`, an absolute Frobenius criterion. `tol=0` runs exactly `max_iter` iterations, including at a stationary point
- The returned objective is evaluated at the returned `(centers, memberships)` pair. This differs from scikit-fuzzy's reported previous-membership objective, so the comparator recomputes it
- Empty clusters retain their previous centers, initially the arithmetic data mean
- Input/init arrays are copied as needed and are never modified. Initial membership rows are normalized. Complex/nonfinite/negative membership inputs are rejected
- Internally subtracting the first observation stabilizes weighted centers under large translations. Very small global coordinate scales are normalized temporarily, and distance cancellation in the BLAS path is detected and directly recomputed
- Log-domain fallbacks preserve representable fractional membership powers and objective terms when an intermediate ratio/power would underflow. Near-one m uses log1p on nearby distances; tiny objective terms are aggregated with logsumexp/logaddexp before conversion back to ordinary values
- Huge coordinate ranges or locally unrepresentable distance ranges are rejected with a rescaling error rather than silently inventing coincident points. A genuinely unrepresentable final objective can underflow to zero or raise on overflow
- At huge offsets, converting centered centers back to input float64 coordinates necessarily rounds centers. The published objective uses those published coordinates
- `objective_history`, if requested, uses the internal center-step coordinates before final output rounding. Its final entry can therefore differ slightly from the final published objective at huge offsets

## Backend choices

| Backend | Implementation | Main tradeoff |
|---|---|---|
| `numpy` | Straightforward broadcast `(N,K,D)` differences | Clear scratch reference; large temporary memory |
| `scipy` | Direct SciPy `cdist(..., 'sqeuclidean')` | Portable low-memory default |
| `blas` | Centered Gram identity plus exact cancellation repair | Often best at higher dimension |
| `numba` | Fused direct distances, memberships, delta, objective | Optional JIT startup; low-D hot loop candidate |
| `numba_parallel` | Same fused work with `prange` | Optional JIT and explicit worker count |
| `auto` | Currently selects `scipy` | No import-time JIT, no performance heuristic inferred from one machine |

`threads` caps BLAS threads; it also sets the Numba worker count for `numba_parallel`. The serial `numba` loop remains serial. Settings are restored on exit. Threadpool settings are process-global; simultaneous fits in different threads with different limits need caller coordination.

## Reproduce tests and measurements

```sh
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.benchmark --output results/benchmark_t1.json
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m benchmarks.benchmark --threads 4 --cases large_n high_d many_clusters \
  --output results/benchmark_t4.json
python -m benchmarks.benchmark --auxiliary --cases small_overlap high_d \
  --output results/cold_and_memory.json
```

The harness generates deterministic separated/overlapping data spanning N, D, and K, and hashes both the data and supplied initial memberships. It runs a warmup for each method/case/mode, then repeats all methods in seeded shuffled order. Median wall time is reported along with raw samples, iteration counts, final objective/FPC, membership/center error, and ARI against both known generating labels and the NumPy reference. Fixed-iteration and convergence modes are separate. Data generation and quality scoring are outside the timer. Validation, copying, fitting, and final output construction are timed.

The scikit-fuzzy comparator is **scikit-fuzzy 0.5.0 plus a common-output adapter**, not bare `cmeans`: final-pair objective, membership layout conversion, and labels are included. All implementations use the same supplied memberships and stopping criterion. scikit-fuzzy's epsilon flooring is intentionally retained as library behavior, so extreme-distance/zero-distance edge cases are separately described rather than called bit-for-bit equivalent.

Cold-process/JIT cost is measured in fresh interpreters with empty Numba cache directories. Linux `ru_maxrss` records process high-water memory, and incremental high-water above imports/data is reported; this is not a precise allocator trace. First-fit and second-fit times are separate from import time. The benchmark helpers import scikit-learn, so total process import time is a benchmarking environment measurement, not a minimal runtime import claim.

## Comparators

Primary comparator: [scikit-fuzzy 0.5.0](https://pypi.org/project/scikit-fuzzy/), BSD-3-Clause. It accepts supplied memberships and exposes iterations, objective history, and FPC. Its source is inspected as a dependency, not copied into this implementation.

[fuzzy-c-means 2.0.3](https://pypi.org/project/fuzzy-c-means/) was also examined. It provides no explicit initial-membership argument, depends on NumPy <2, and its inverse-distance path has no exact-zero protection. It was not installed into this NumPy 2 benchmark environment; no runtime performance claim is made about it. Details are recorded in [COMPARATORS.md](COMPARATORS.md).

## Status

See [numeric stability](THEORY.md), [comparator contracts](COMPARATORS.md), and the [performance summary](../docs/PERFORMANCE_JA.md). Benchmark commands above generate new output files; archived raw measurements and older source copies are not required or included. The summarized primary FCM measurements used three repeats; the harness default is five.

Additional theoretical ablations: `python -m benchmarks.ablation` and `python -m benchmarks.m2_ablation`. The former swaps a private membership function only in an isolated research process; do not use it concurrently with application fits. Numba objective history requests add a direct-distance objective pass for robust extreme-value summation.
