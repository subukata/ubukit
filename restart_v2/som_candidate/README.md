# SOM candidate: newly tested reconstruction

This isolated package is a new tested implementation for the restarted speed
challenge. It is not represented as a byte-identical recovered old package.
The oracle is copied from the retained `harness/reference.py`, which attributes
upstream SOM-OLP commit `4361175b776987d65c348d0132b31d43505e1069` (MIT):
https://github.com/subukata/som-olp/tree/4361175b776987d65c348d0132b31d43505e1069
The original upstream MIT license was downloaded from that pinned commit and
is included as LICENSE.upstream; the example grid source is retained in upstream/.

## Dependencies and API

Required: NumPy, SciPy, threadpoolctl. No JIT/compiler/native extension is required.
All package imports are relative and the package does not mutate `sys.path`.

- `initialize(X,R,lam,pca_scale=2.,threads=1,method='direct') -> (W0,P0)`
- `initialize_lowrank(...) -> (W0,P0)` explicitly selects the same-SVD identity
- `run(X,R,W0,P0,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,distance='guarded',block_rows=None)`
- `fit(X,R,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,pca_scale=2.,initialization='direct',distance='guarded',block_rows=None)`

Run returns `W/P/V/history/n_iter` plus descriptive diagnostics. Inputs are
unmodified; run expects finite C-contiguous float64 arrays with compatible
shapes, positive lambda, nonnegative gamma, and a row-stochastic initial P.
Fixed iterations use tol=-1. Zero iterations return copies of W0/P0, V=None,
and empty history. For positive iteration counts W and V precede the final P
update, matching the original lagged-state return. Zero-mass prototype weights
retain their prior W values.

`distance='direct'` uses squared-difference cdist throughout and a stable
log-partition objective. `distance='guarded'` centers feature/grid coordinates,
uses BLAS for softmax scores, adds row constants back to every objective and
uses direct-distance fallback for translated or cancellation-risk inputs.
`distance='centered'` is an explicitly unguarded algebraic comparison.
The guard is conservative engineering, not a universal proof of floating
trajectory equivalence. All variants preserve the exact-real equations, with
floating accumulation/order differences explicitly acknowledged.

The optional initializer uses the **same full SVD** and exactly the original
W0 formula. It evaluates initial scores in the leading SVD coordinate space;
it does not substitute randomized, truncated, covariance, or approximate SVD.
Large translation or cost-error scales relative to lambda fall back to original
cdist initialization. Default initialization remains the original method.

Per-call setup/centering/allocation is inside the callable. Threadpoolctl limits
BLAS threads for the duration of each call. `block_rows=None` handles the full
row dimension in one block; explicit blocking reduces N×M scratch memory. There
is no hidden persistent data cache or unmeasured precomputation.

## Tests completed before benchmarking

From the parent directory of this package, with BLAS threads controlled:

```
OPENBLAS_NUM_THREADS=1 python -m som_candidate.test_tiny
OPENBLAS_NUM_THREADS=1 python -m som_candidate.test_initialization
```

- 144 tiny kernel configurations passed, including zero masses, duplicated and
  zero data, feature/grid translations, lambda .001/.2/10, and row blocking
- Worst normal absolute errors: P 3.4e-15, V 7.7e-15, history 2.5e-13
- Offset stress tests permitted 2e-5 tolerance; observed P/W/V differences were
  zero in this particular test, with direct fallback exercised
- Stopping and zero-iteration checks passed
- 72 initialization/full-fit configurations passed; W0 was bitwise identical,
  maximum P0 difference 5.1e-14 and five-update P difference 2.0e-14

Machine-readable results are `tiny_result.json` and `initialization_result.json`.
These finite tests are not universal error guarantees. No heavy performance
measurement has been run for this package at this checkpoint.

## Additional explicit backends

`som_candidate.threaded.run` is a dependency-free ThreadPoolExecutor experiment.
BLAS is limited to one thread once around the executor; statistics finish before
any old P is overwritten, assignments write disjoint row ranges, and numerator,
denominator and objective reductions follow fixed worker order. Translation
risk triggers an explicit single-thread direct fallback. It remains an option,
not the default. Its 108 tiny configurations and repeat-stability/stopping
checks pass, with worst P error 2.5e-14.

`som_candidate.native.run` is an optional rebuilt native control adapted from
retained v2 Cython/header source into a unique module name. Build from restart_v2:
`.venv/bin/python som_candidate/setup_native.py build_ext --inplace`.
It requires Cython/GCC/OpenMP/Linux glibc libmvec and is compiled with
`-march=native` without fastmath. Guarded mode conservatively selects direct
rather than raw GEMM distances at risky scales. Raw `distance='gemm'` remains an
explicit comparison mode. The new build passed 162 tiny configurations,
zero-mass/nonmutation/stopping checks; worst observed P difference was 1.9e-15.
This rebuilt binary is not represented as the original pre-restart binary.

## Reproducible MNIST preparation

`prepare_mnist.py` downloads the official Keras archive only if missing and
checks SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
It uses original training then test order, deterministic prefixes, float64 /255,
no shuffle or StandardScaler, gamma 2.767, lambda 2.439, and PCA scale 2.
The 16×16 [-1,1] grid exactly matches the pinned upstream example's meshgrid
and C-ravel order; its bytes hash to
`423f053767168a424a50c145b4bd0ba2614c566818943eee177b2462b976061a`.

`inspect` verifies only archive/configuration. `fixture` explicitly performs
initialization and an optional original-reference run. Generated NPZ fixtures
under runtime/ are reproducible data, excluded from source checkpoints.
Benchmark workers run one method/scope/thread choice per fresh process and
record input/source hashes, affinity, warm samples, CPU time, memory, validation,
and scope. Initializer/full-fit measurement is separate from kernel timing.

## First fresh representative screen

MNIST2k, 784 features, 256 prototypes, 30 fixed updates, three warmed repetitions,
common original-SVD initial state. All eight runs passed W/P/V/history allclose
1e-8 and input nonmutation; candidate P maximum error was at most 5.66e-11.
Median kernel seconds, including per-call setup but excluding initialization:

| Method | 1 thread | 9 threads |
|---|---:|---:|
| Original oracle | 4.4327 | 4.3705 |
| Guarded NumPy | 1.1140 | 0.4611 |
| ThreadPool NumPy | 1.1197 | 0.3625 |
| Rebuilt native | 1.1240 | 0.2723 |

Original shared initialization took 0.4400 s in the fixture-generation call;
that is one setup observation, not a warmed full-fit benchmark. The native
nine-thread kernel is 16.05× faster than the same-thread oracle in this screen.
No earlier-run measurements were substituted for these new results.

## Fresh full-MNIST fixed-10 candidate screen

All 70,000 official images, common original full-SVD initial state, 10 fixed
updates, nine threads, three warmed repetitions per candidate:

| Candidate | Median kernel seconds | Max absolute P error | Peak process RSS |
|---|---:|---:|---:|
| Rebuilt native | 2.58144 | 3.69e-11 | 1328.5 MiB |
| ThreadPool NumPy | 2.85783 | 3.66e-11 | 1485.8 MiB |
| Guarded NumPy | 4.30279 | 4.81e-12 | 1804.1 MiB |

All W/P/V/history comparisons passed allclose1e-8 and inputs were unchanged.
Kernel times include candidate centering, guards, allocation and executor setup,
but exclude common initialization. Peak process RSS includes fixture and
validation arrays, not just algorithm workspace.

Fixture construction observed original initialization at 12.9988 s and one
one-thread oracle fixed-10 run at 51.0961 s. Those are single preparation
observations, **not** warmed same-thread baselines and not a fair speedup ratio
against the nine-thread candidate medians. Actual full-fit and real-data
same-SVD-lowrank initializer timings have not yet been measured at this stage.

Complete raw records and source hashes are in results/mnist70k_i10. The official
archive and large uncompressed fixtures are reproducible local data and should
be excluded from source-checkpoint ZIPs; retain evidence/*.metadata.json and the
preparation driver. All numbers here are from this restarted challenge.
