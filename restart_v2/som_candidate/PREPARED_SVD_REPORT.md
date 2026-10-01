# Same-X prepared-SVD reuse: a separate amortized workload

## Result and scope

For three fresh-state full fits on the same 70,000-image MNIST matrix, the
existing public API took **40.008182 s median**. Creating an owned immutable
snapshot, computing the same full SVD once, and then running those three fits
took **33.088952 s median**: **1.209110× throughput improvement**, or **17.2945%
less total wall time** in this experiment.

Preparation is included in every warmed workload. This does not replace or
increase the separately measured **12.8956× single-full-fit** result. In
particular, cached-fit latency alone must not be compared with a cold full fit
while hiding preparation cost.

| Workload | Three warm samples (s) | Median (s) | MAD (s) |
|---|---|---:|---:|
| Three fresh public full fits | 39.338523, 40.147115, 40.008182 | 40.008182 | 0.138933 |
| New snapshot + full SVD + three prepared fits | 33.298969, 33.088952, 32.486246 | 33.088952 | 0.210017 |

The parameter pairs (gamma, lambda) were (2.5,2.2), (2.767,2.439), and (3.0,2.7),
in that order. They stopped at 38, 36 and 41 updates respectively in both paths.
Each process used nine CPUs, B=256, the explicit 64 MiB named scratch allowance,
tol=1e-4 and max_iters=1000. The grid, /255 data preprocessing and full SVD were
the same as the saved baseline. No warm-started W or P was shared between fits.

## Preparation and observed amortization

Median externally timed preparation was **4.189039 s**. It includes the owned
copy/snapshot, input validation/hash, centering, full SVD and retained factors.
Every prepared warm workload pays that cost again, followed by its three fits.

Prefixes computed from the timed phase components of the same K=3 workloads:

| Prefix | Fresh median phase total (s) | Prepare + prefix median phase total (s) |
|---|---:|---:|
| First fit only | 13.105776 | 13.511531 |
| First two fits | 25.754864 | 22.494271 |
| All three fits | 40.008163 | 33.088919 |

The first-fit prefix was slightly slower; preparation paid off by the second
fit for these parameter cases. These prefix figures are derived from the
measured phase components, not independent K=1/K=2 benchmark campaigns. Use the
outer workload medians above for the K=3 headline; their tiny differences from
phase sums include loop/materialization overhead.

## Memory and ownership

The retained snapshot and leading factors occupy **440,178,832 bytes
(419.787 MiB)**. The caller's original X can remain alive separately. Preparation
still incurs full-SVD peak workspace; retaining only leading factors does not
turn the solver into a truncated or randomized SVD.

Measured full-process peak RSS was 2478.113 MiB fresh versus 2607.457 MiB prepared.
These peaks include inputs, retained workload outputs, SVD and validation arrays.
The difference between peaks is not the cache's steady owned size, because the
peaks can occur in different phases. The 64 MiB policy remains a named scratch
allowance, not a total-RSS bound.

Data and retained factors are backed by immutable bytes. Ordinary NumPy write
flags cannot reopen them, and prepared object fields are sealed. Changing the
original caller array does not change the owned snapshot. `fit` has no X
argument and always uses that snapshot. `assert_same_input` is an optional
explicit dtype/shape/raw-value identity check; its scan cost is not hidden in
cached fit times. Different X requires a new prepared object.

## Exact semantics and cache contract

- Working dtype: float64, with public-compatible real numeric conversion
- Preprocessing: original owned X; mean-centering only for the original SVD
- Factorization: numpy.linalg.svd(X-mean(X), full_matrices=False), computed fully
- Retained state: X, mean, leading singular values/basis, and U[:, :rank]*s[:rank]
- Initial W/P: newly constructed for each grid, lambda and PCA scale
- Probability scoring: same numerical low-rank identity and direct-distance
  risk fallback as the already validated initializer
- Kernel: the frozen public run_som_olp ThreadPool path; no warm starts
- Factorization is pinned to preparation_threads. A later fit policy with a
  different thread count reuses those original factors; it does not silently
  recompute an SVD under the new thread count. To request a fresh SVD under a
  different policy, construct a new prepared object

The measured comparison used nine threads consistently for preparation and fits.
It does not claim bitwise equality against fresh SVDs under arbitrary different
thread counts, libraries or versions.

## Validation and reproducibility

All cold and warmed W/P/V/history arrays had **maximum absolute difference 0**
against the corresponding fresh public references, with matching iteration
counts. Repeated results within each mode were bitwise stable. Inputs and public
source hashes remained unchanged. The measured candidate source hashes were
checked after the campaign and still matched.

Eighteen tiny parameterized fits also pass, including float64/float32/uint8,
zero/duplicate/translated data, changed lambda/PCA scale, shape/dtype/value
identity rejection and immutable ownership. A separate untimed post-benchmark
70k initialization check at lambda 2.439 and nine threads found **SHA256-identical
W0 and P0** between prepared and fresh public initialization.

Completed timing records:
- `results/prepared_sweep70k/fresh.json` (completed 2026-10-01T00:56:34Z)
- `results/prepared_sweep70k/prepared.json` (completed 2026-10-01T00:58:51Z)
- `prepared_svd_real_initialization_check.json` (correctness-only check,
  completed 2026-10-01T01:00:41Z)

Use `prepared_sweep_worker.py` first with mode=fresh, then mode=prepared, the
same fixture/cases/reference directory, threads=9 and repeats=3. The first timed
fresh workload supplies the per-case references; no long original-class oracle
is added. Large reference NPZ files remain reproducible local data, excluded
from source checkpoints. `PREPARED_SVD_MANIFEST.json` records this extension
separately; the saved single-full-fit source/evidence manifest stays unchanged.

## Measured implementation versus public integration

The large timing campaign measured the isolated `som_candidate.prepared_svd.PreparedSOM`
wrapper, which delegates each kernel to the frozen public `run_som_olp`. The
fresh comparator called the frozen public `fit_som_olp` directly. After the
campaign, core integrated an explicit lazy PreparedSOM API into version 0.2.0a2
with relative-import/docstring/metadata adaptation and reported 14 passing shared
test methods, including 18 exact-equality reuse cases and a one-SVD-call check.
Those post-integration tests are distinct from this candidate's large timing
record. The saved single-fit default and its 12.8956× result are not replaced.
