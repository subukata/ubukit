# SOM-OLP and joint neighborhood quality

These are dependency-free ES modules using Float64Array row-major matrices.
They are CPU implementations with resumable generators and Web Worker support;
there is no GPU/WASM acceleration or blanket speedup claim.

## Sources and license

SOM-OLP is a mathematical JavaScript port of Seiki Ubukata's implementation:

- https://github.com/subukata/som-olp
- Pinned commit: `4361175b776987d65c348d0132b31d43505e1069`
- Recovered local source: `restart_v2/som_candidate/upstream/somolp.py`
- Source SHA-256: `7761f76770fa846f106ea031141bb953cda9d5967dc6b88d169649ee6cc2057d`
- The original MIT notice is retained in `LICENSES/SOM-OLP-MIT.txt`
- The recovered Python oracle is `restart_v2/som_candidate/oracle.py`, SHA-256
  `7d07789ee5e5e6677562a6c1407a0bd510823ad6a5aeaf8ce637a8ecce0a5daf`

Neighborhood normalization and cross-rank penalties follow the recovered
`restart_v2/portable_accel/_backends/metrics_numpy.py` implementation. The JS
selection/ranking implementation is new, with a deliberately explicit portable
floating-point/tie contract described below.

The source files under the repository-root `restart_v2/` remain unchanged. Fixture
provenance and source hashes are embedded in
`fixtures/som-neighborhood.json`. The evidence is tiny correctness work, not a
performance benchmark or a claim of full MNIST Python/JavaScript parity.

## Shared input and execution

A matrix is `{data: Float64Array, nSamples: N, nFeatures: D}`. Other numeric typed
arrays are converted to Float64Array; ordinary JS arrays are intentionally not
accepted. Inputs must be finite and are never mutated. Outputs are typed arrays.

Use `somOlp(input, options)` or `neighborhood(input, options)` synchronously.
`somOlpSteps` and `neighborhoodSteps` return generators which yield bounded-work
progress events. Both call `onProgress(event)` themselves, once per event, and
check `signal`/`shouldCancel` at boundaries. The common async/worker wrappers
consume these generators without doubling callback delivery.

`maxScratchBytes` caps documented **primary numerical scratch allocations**, not
whole-process RSS, input conversion, retained results, JS-engine overhead, sort
implementation scratch, garbage-collection timing, or Worker message copies.
No caller-owned input buffers are transferred unless the caller selects that
behavior in the common worker interface.

## SOM-OLP API

Required option:

- `grid`: matrix of M grid/prototype coordinates in Q dimensions

Main options:

- `gamma=1`: nonnegative map-coherence penalty
- `lambda=0.1` (`lam` alias): strictly positive softmax/entropy temperature
- `maxIterations=100`: nonnegative integer, including a useful initialization-only zero
- `tolerance=1e-4`: relative objective-change stopping threshold
- `blockRows=128`: progress/cancellation work granularity
- `maxScratchBytes=33554432`: primary scratch cap
- `maxMemoryBytes=536870912`: guard on estimated normalized input, state, output, and primary scratch arrays
- `initialPrototypes` (`initCenters` alias): M×D typed array
- `initialMemberships` (`initMembership` alias): N×M typed array; requires prototypes
- `initializer='pca'`: used only when prototypes are not supplied
- `pcaScale=2`, `pcaTolerance=1e-13`, `pcaMaxSweeps=80`, `pcaMaxDimension=128`
- `initializer='sample'`, `seed=0`: explicit alternative initialization

Membership rows provided by the caller must be nonnegative and sum to one
within 1e-8. They are copied without renormalization, preserving the supplied
iteration state. Prototype-only initialization computes memberships by softmax
of squared direct-difference distances.

### The actual SOM-OLP iteration

With the input memberships P of one iteration:

1. Compute V = P R
2. Compute each prototype W_j = sum_i P_ij X_i / sum_i P_ij, retaining its old
   value when its mass is zero
3. Compute C_ij = ||X_i-W_j||² + gamma ||V_i-R_j||²
4. Compute new memberships P_ij proportional to exp(-C_ij/lambda)
5. Compute objective sum(P*C) + lambda sum(P*log(P)), taking 0*log(0)=0
6. After at least two iterations, stop when the absolute objective change divided
   by max(1, absolute previous objective) is at most tolerance

This is the upstream SOM-OLP update, **not** a winner-take-all SOM, a Gaussian
neighborhood SOM, or a k-means surrogate. Softmax subtracts the minimum cost
before exponentiation to prevent overflow. Float64 reduction/exp/log order can
still differ slightly from BLAS, NumPy, and SciPy.

The original return timing is preserved: W and V are calculated from the input
P of the final iteration; the returned P is the newer softmax result. V is not
silently recomputed as final-P times R. At zero iterations V is null, while W/P
contain initialization. `history` contains exactly the executed objectives.

Result fields:

- `prototypes`, `centers`, `W`: aliases of one M×D Float64Array
- `memberships`, `membership`, `P`: aliases of one N×M Float64Array
- `embedding`, `V`: aliases of one N×Q Float64Array, or null at zero iterations
- `labels`: each row's first maximum-membership index
- `history`, `iterations`, `nIter`, `converged`
- `nSamples`, `nFeatures`, `nUnits`, `nClusters`, `nComponents`
- `initialization` and `stats`: explicit initialization and work/memory metadata

### Initialization compatibility boundaries

For direct cross-language **kernel** comparison, pass the same W0 and P0 from
the original Python initializer. The fixtures do this and test the subsequent
iteration sequence against the recovered Python oracle. This does not establish
that JavaScript and NumPy independently initialize the same full fit.

Default JS PCA forms the centered covariance matrix (or dual sample Gram
matrix when N<D), uses cyclic Jacobi diagonalization, sorts eigenvalues in
descending order, and gives each principal loading a positive largest-magnitude
entry. Grid coordinates are centered and normalized per axis by the maximum
absolute centered coordinate, floored at 1e-12. Prototypes use the original
PCA scale rule, pcaScale times each component's population standard deviation.

This reproduces the PCA mathematical construction, but NumPy's LAPACK SVD has
its own sign and degenerate-subspace choices. Therefore `numpySvdParity` is
false for this initializer. Tests compare two well-separated-eigenvalue cases
to **sign-canonicalized** NumPy SVD; they do not imply general full-fit parity.
Covariance formation also squares conditioning relative to direct SVD. Avoid
claiming accurate tiny principal components on severely ill-conditioned data.

PCA is intended for small browser demos: min(N,D) defaults to a cap of 128.
Higher dimensions fail with an actionable message rather than silently doing
an expensive eigensolve. Explicitly raise `pcaMaxDimension`, provide W0/P0, or
choose sample initialization when appropriate. Sample initialization selects
M input rows independently with replacement using JS Mulberry32 and the stated
seed; it is explicitly not NumPy's RNG or the original PCA initializer.

### Cost and memory

Each training iteration is O(N*M*(D+Q)). Membership state costs 8*N*M bytes,
embedding 8*N*Q bytes, and prototypes 8*M*D bytes. No N×M cost matrix or second
membership matrix is allocated: one M-element cost row is reused after the
old memberships have contributed to all V/W accumulations. Primary training
scratch is 8*(M*D+2*M) bytes, independent of blockRows. A separate
maxMemoryBytes guard checks estimated normalized inputs, state, result arrays,
and the larger of training/PCA scratch before allocating SOM state. It is not a
whole-process or garbage-collector peak-memory guarantee.

PCA uses dimension L=min(N,D), about two L×L Float64 matrices, and cubic
Jacobi-sweep work; covariance formation is O(N*D*L). Its primary scratch is
16*L² + 8*(2*D+2*Q) + 4*L bytes, separately reported. Input/output buffers and
engine overhead are outside that number. Timings must include initialization
when labeled full fit and must distinguish supplied-init kernel-only runs.

## Exact joint trustworthiness / continuity API

Options:

- `embedding`: Y matrix, with the same N as X (`coordinates` alias)
- `k=5` or `ks=[...]`: each integer must satisfy 1 <= k < N/2
- Duplicate k values are removed while preserving requested order
- `blockRows=16`, `maxScratchBytes=33554432`
- Shared progress and cancellation options

`qualities` is an array of `{k, trustworthiness, continuity,
trustworthinessPenalty, continuityPenalty}`. `stats` states the exact distance
and tie contract, scratch size, and absence of full distance/rank matrices.

For each sample i, neighbor selection and ranks use the same order:

1. Rooted Float64 Euclidean distance computed by summing squared direct
   coordinate differences in feature order, then applying Math.sqrt
2. Original sample index ascending for equal rounded distances
3. Self excluded even when other samples are identical

Thus rounded-root ties are retained; distinct squared values that round to the
same square root are **not** ordered by their squared values. Top-k is a prefix
of one consistent order, so sharing a maximum-k heap across requested ks is
valid for this contract.

The trustworthiness penalty sums max(r_X(i,j)-k,0) over each of Y's k nearest
neighbors. Continuity reverses X/Y. Both scores are one minus the penalty times
2 / (N*k*(2*N-3*k-1)). All accumulated integer penalties are checked against
Number.MAX_SAFE_INTEGER; an unsupported result throws rather than rounding.

### Exact means exact under the stated contract

This computes the full-dataset formula, with no sampling, approximate neighbor
search, surrogate ranks, or silently substituted k. However it does **not**
promise bitwise equivalence to sklearn for every floating-point input:

- The recovered strict Python implementation uses sklearn's full Gram-based
  distance arithmetic, installed NumPy's unstable argsort for tied rank rows,
  and sklearn's independently selected per-k neighbors
- Those rules vary with dependency version, dimensionality and neighbor backend
- Direct differences can resolve near ties differently from Gram arithmetic
- Exact equal-distance ties use the explicit index policy in JS

The ordinary no-tie Python fixture exactly matches sklearn's integer penalties
and scores. A duplicate-tie fixture intentionally demonstrates differing
sklearn and portable-index results. These differences are visible in tests and
metadata rather than being presented as full strict-Python parity.

### Quadratic work without quadratic storage

Each row reuses two N-distance vectors and small maximum-k heaps. Query ranks
use a histogram over sorted query thresholds, binary-searching each distance
rather than sorting a full N-rank row or scanning the row once per query.

- Distance work remains O(N²*(DX+DY)); exact metrics are expensive at large N
- Neighbor/rank work is O(N²*log(Kmax) + N*sum(ks))
- Primary numerical scratch is 16*N + 36*Kmax + 8 + 16*numberOfKs bytes
- No N×N distance matrix, N×N rank matrix, or N×K neighbor cache is allocated
- blockRows affects yielded work batches, not row-distance arithmetic or results

For a future demo, show progress and cancellation and make the exact quadratic
cost visible. Use a worker to keep the UI responsive. Choosing a smaller
explicit dataset changes the evaluated dataset and must be disclosed; this
implementation does not silently subsample.

## Reproduce correctness checks

From the package directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/generate-som-neighborhood-fixtures.py
node --test tests/som-neighborhood.test.js
```

The generator expects the unchanged recovered Python tree alongside the package;
its dependencies are NumPy, SciPy, sklearn and threadpoolctl. JavaScript runtime
code and checked-in tests need no Python or third-party npm dependencies.

The focused tests cover six prepared-initialization SOM fixtures, both PCA
paths against sign-canonicalized SVD, the final-return lag, deterministic sample
initialization, tile invariance, cancellation, validation, four independent
Python neighborhood fixtures, no-tie sklearn parity, tied-case incompatibility,
identity/reversal properties, and 35 tiny randomized brute-force rank checks.
Browser/Worker integration evidence is maintained by the common package tests;
these focused Node results are not by themselves a browser test or benchmark.
