# ExRCM / RCM: independent NumPy research implementation

This implements the normalized binary-overlap ExRCM update below. See the [repository quickstart](../README.md#exrcm) for complete runnable examples.

## Exact requested model

For Euclidean distance `d[c,i] = ||x[i] - center[c]||₂`:

```
M[c,i] = 1{d[c,i]^p <= alpha^p * min_j(d[j,i])^p + beta^p}
U[c,i] = M[c,i] / sum_j M[j,i]
center[c] = sum_i U[c,i] * x[i] / sum_i U[c,i]
```

- `p > 0`, finite; `alpha >= 1`, finite; `beta >= 0`, finite
- `p=1` is the requested RCM rule; `p=2` can use squared distances directly
- `beta` has Euclidean-distance units for **every** p. Increasing p is not the same as reinterpreting beta in squared units
- Every qualifying cluster receives the same membership for a sample. No fuzzy exponent is applied to memberships or the center update
- All equal nearest distances qualify, including multiple zero-distance centers
- No conventional rough-c-means lower/upper-region weights have been substituted

## Quick use

Requires Python 3.10+ and NumPy. SciPy and Numba are optional.

```python
from rough_cmeans import fit_exrcm, fit_rcm

result = fit_exrcm(X, 8, p=2, alpha=1.15, beta=0.5, seed=42,
                   backend="auto", return_memberships=True)
centers = result.centers            # (clusters, features)
U = result.memberships             # (clusters, samples)
M = result.upper_memberships        # (clusters, samples), bool
print(result.stop_reason, result.n_iter)
```

`fit`, `fit_exrcm` are aliases. `fit_rcm` enforces `p=1`. `assign(X, centers, ...)` returns `(U, M)` without fitting.

For fair comparisons, supply the **same explicit** `init=(clusters, features)` array to each backend. A seed is reproducible within NumPy's `default_rng` sampling contract; it is not a promise of matching another language's random generator.

## Initialization, empty clusters and stopping

- Default initialization samples distinct data **indices** without replacement. Duplicate data rows may still produce identical initial centers
- An empty cluster retains its previous center; `empty_cluster_updates` counts how often that policy was used
- A repeated consecutive binary mask or identical centers gives a fixed point. There is no unproven objective-decrease test and no arbitrary center-tolerance shortcut
- Exact repeated mask-and-center states within `cycle_window=16` stop with `stop_reason="cycle"` and `converged=False`. Set `cycle_window=0` to disable cycle detection. Longer cycles are still bounded by `max_iter`
- At the iteration cap, `stop_reason="max_iter"`, unless the final assignment verifies a fixed point
- Exported memberships are recomputed for the exported centers, including when a run did not converge. Nonconverged centers need not equal the mean implied by these final memberships
- `n_iter` counts center-update iterations, not the extra final output-alignment assignment

## Implementations

| Backend | Distance and update | Temporary storage |
|---|---|---|
| `naive` | Simple full broadcast `(X[:,None]-C[None,:])²`, NumPy sum, full U and weighted center sum | O(N K D) |
| `numpy` | Blocked direct differences, no broadcast across both K and D, BLAS weighted update | O(B(K+D)+K D) plus packed masks |
| `scipy` | Blocked `scipy.spatial.distance.cdist(..., sqeuclidean)`, BLAS weighted update | O(B(K+D)+K D) plus packed masks |
| `numba` | Optional serial fused distance/threshold/count/accumulation, `fastmath=False` | O(K D+K) plus packed masks |
| `auto` | SciPy if installed, otherwise NumPy | As above |

All implementations currently retain packed binary masks (N K/8 bytes, rounded up per sample) for exact fixed-point/cycle checks. The history keeps at most `cycle_window` packed masks and their updated centers. Centers are part of cycle state because empty clusters retain their previous location. `return_memberships=False` avoids exporting dense N×K floating memberships and dense boolean masks; the Numba kernel also avoids their allocation during every update. Requested final output adds O(NK) storage for every backend.

Numba's first invocation includes import/JIT overhead; warm timings exclude it. No architecture-specific source or C++ compiler is required. The serial kernel avoids race-dependent reductions. Compiled availability and timings on x86 Linux are not a claim of measured performance on every platform.

## Numerical semantics and limits

1. Direct differences are used, not `||x||² + ||c||² - 2x·c`. A sample-dependent squared-norm term must not be dropped from this relative-plus-additive threshold: doing so changes the model
2. `p=1` evaluates square roots plus the linear threshold; `p=2` uses the squared-distance comparison. For general p, let `a=alpha*dmin`, `s=max(a,beta)`. The overflow-safe reference comparison is `(d/s)^p <= 1 + (min(a,beta)/s)^p`. `s=0` selects exactly zero distances. With beta=0, the power cancels algebraically, so a tiny p cannot round all positive ratios to one. Extreme p, underflowed ratios, or overflowed distance ratios use the equivalent log-domain inequality, with `log1p` near ratio one and `logaddexp` for the right-hand sum
3. A generic-p optimized block path computes one scalar root threshold per sample, then compares squared distances. Values near its floating threshold are recomputed using the powered reference comparison. The guard selects **recomputation only**; it never widens `<=`. Extreme p uses the scaled comparison directly
4. These formulas agree in real arithmetic. Different IEEE-754 evaluation orders, powers, distance sums or center-reduction orders can differ at an exact boundary. Tests include ULP-near boundaries; exact universal bitwise equivalence to every possible evaluation order is not claimed. The reference backend and explicit initialization are supplied to investigate such cases
5. Center sums use `X-X[0]` before adding the common offset back, reducing loss from large offsets. This is algebraically the same weighted mean
6. Input and initial centers must be finite float64 arrays. Squared-distance overflow, subnormal squared distances, or a nonzero distance rounded to squared zero raise `FloatingPointError` with a rescaling request. This deliberately avoids silently interpreting tiny nonzero distances as zero ties. Rescale coordinates **and beta by the same factor**; alpha and p do not change

## Verification and benchmark reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark.py --repeats 5 --output results/benchmark_final.json
```

The suite covers the literal formula, hand-computed overlap/center updates, duplicate and zero ties, deterministic initialization, empty clusters, final-output alignment at max iterations, large offsets, p up to 1e6, near-zero range handling, near-threshold ULPs, and same-init backend parity. Optional backends are tested only if installed.

The benchmark fixes data, initial centers, thread count, max iterations and output requirements. It warms all backends, rotates execution order between paired repetitions, records all raw timings and checks masks, memberships and centers. It records SHA-256 source hashes and package versions and reports full fit and one update separately. A failed correctness case must not be presented as an acceleration win.

[JavaScript parity fixtures](../javascript/fixtures/exrcm-python.json) contain the fixed-init Python reference results used by the JavaScript tests. Python memberships are clusters×samples; transpose when comparing a samples×clusters API.

[Performance summary](../docs/PERFORMANCE_JA.md). The commands above generate new measurement files; historical raw logs and intermediate revisions are not included.
