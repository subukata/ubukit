# Online SOM and true Batch SOM

Private additive candidate; no registry publication. These are separate algorithms
from the existing `fit_som_olp`; that entry point and its numerical code are retained.

## Quick start

```python
import numpy as np
from ubukit import som, som_batch, initialize_som, initialize_som_batch

X = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [2., 2., 1.]])
online = som(X, grid_shape=(16, 16), epochs=10, random_state=7)
batch = som_batch(X, grid_shape=(16, 16), epochs=10, random_state=7)
assert online['centers'].shape == batch['centers'].shape == (256, 3)
assert online['labels'].shape == batch['labels'].shape == (4,)
assert online['embedding'].shape == batch['embedding'].shape == (4, 2)
```

`fit_som` and `fit_som_batch` are aliases of `som` and `som_batch` respectively.
All seven additions, including `SOMState`, are available from both `ubukit` and
`portable_accel`. Importing the facade remains lazy, without loading NumPy/Numba.

## Mathematical definitions

The best-matching unit is the nearest prototype by Euclidean distance. Exact
represented distance ties select the lowest row-major unit index. For lattice
positions r, the Gaussian neighborhood is

    h(j,b,sigma) = exp(-||r_j-r_b||² / (2 sigma²))

For online SOM, each input sample selects its current BMU and commits

    W_j <- W_j + learning_rate * h(j,b,sigma) * (x-W_j)

For Batch SOM, all BMUs are assigned using the prototypes at the start of the
epoch. The new prototypes are committed simultaneously:

    W_j <- sum_i h(j,b_i,sigma) X_i / sum_i h(j,b_i,sigma)

Batch SOM has no learning rate and rejects supplied learning-rate parameters.
This implementation is not an online pass relabeled as a batch update. It has no
SOM-OLP objective and does not claim monotonic objective decrease or convergence
to a global optimum. Training stops at the specified budget or cancellation.

The equations follow Teuvo Kohonen, “Essentials of the self-organizing map,”
Neural Networks 37 (2013), §§4.1–4.4, especially equations (3)–(8):
https://doi.org/10.1016/j.neunet.2012.09.018
Author's paper accessible at:
https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/KohonenSOM2013.pdf
SOM Toolbox algorithm documentation (the original page timed out during this
review; the Kohonen paper above was inspected directly):
https://www.cis.hut.fi/somtoolbox/documentation/somalg.shtml

## Parameters

- X: finite real nonempty `(N,D)` array, arbitrary D>=1; copied to float64
- grid_shape: `(width,height)`, positive integers, default `(16,16)`
- Unit order: `[j % width, j // width]`; rectangular, planar, no wraparound
- initial_prototypes: explicit finite `(width*height,D)` array; copied, and
  takes precedence over generated initialization
- initializer: `'sample'` (default) or `'pca'`
- random_state: unsigned 32-bit integer, default 0; sample initialization uses
  Mulberry32, independently sampling N rows with replacement. It is identical
  to the JavaScript sampling stream for the same row order and seed
- PCA uses a centered, power-of-two-scaled SVD, up to two principal directions,
  population standard deviations, canonical largest-component-positive signs,
  and lattice coordinates normalized to [-1,1]. A length-one axis maps to zero.
  `pca_scale=2` by default. Rank-deficient and single-sample inputs are accepted
- epochs: nonnegative integer, default 100
- max_iterations: optional nonnegative integer overriding total update count;
  **online units are samples, batch units are epochs**
- Online sample order: input order, repeated cyclically; no implicit shuffling
- sigma: nonnegative finite value, default max(width,height)/2
- sigma_end: nonnegative finite value, default min(.5,sigma), not above sigma
- learning_rate: online only, finite in [0,1], default .5
- learning_rate_end: online only, default min(.05,learning_rate), not above start
- schedule: `'geometric'` (default) or `'linear'`
- policy: existing `ExecutionPolicy`, for scoped threads and primary scratch cap

For T total committed updates, the schedule fraction at zero-based t is
`t/(T-1)`. Both endpoints are included. T<=1 uses the starting value. Geometric
interpolation uses log-space; when either endpoint is zero it uses linear
interpolation instead. Interrupted/resumed runs retain the original T and t.

The explicit sigma=0 limit updates only the BMU (online) or its assigned sample
mean (batch). A zero effective neighborhood denominator retains the old unit.
At positive sigma, float64 exponential underflow may also make a neighborhood
empty; this behavior is reported here, not replaced with an arbitrary weight.

## Stateful and display contract

```python
state = initialize_som(X, grid_shape=(4, 3), epochs=5)
frame = state.step()                     # exactly one sample update
assert frame['iterations'] == 1
assert frame['unit'] == 'sample'
assert frame['epochs_completed'] == 0
state.run(max_updates=2)                 # continues the same schedule
state.cancel()                          # no next update will start
assert state.step() is None
current = state.result()                # assignments for current prototypes

state = initialize_som_batch(X, grid_shape=(4, 3), epochs=5)
frame = state.step()                     # exactly one full frozen-BMU epoch
assert frame['unit'] == 'epoch'
assert frame['epochs_completed'] == 1
```

`step()` returns None after exhaustion/cancellation; otherwise it returns a
complete committed prototype snapshot. `snapshot()` does not assign every data
row, so drawing a frame does not hide another all-data assignment. It contains
centers, grid, grid_shape, algorithm, iterations, total_iterations, unit,
epochs_completed, samples_seen, sigma, learning_rate, sample_index, bmu,
done, cancelled, initializer and implementation/scratch diagnostics.

The online bmu/sample_index describe the sample selected **before** that update.
Sigma and learning_rate are the values just used; they are None before update 1.
Batch sample_index/bmu/learning_rate are None. `done` means budget exhausted or
cancelled, not mathematical convergence. `run(max_updates=None)` resumes up to
that many further updates and returns `result()`; a prefix may have done=False.

`result()` additionally computes labels `(N,)` and the selected lattice positions
embedding `(N,2)` against the current published centers. Mathematical aliases
W=centers, R=grid, V=embedding refer to the same detached output arrays. Input
arrays, state properties, and different returned snapshots do not alias state.
Cancellation is cooperative between updates, not an asynchronous mid-epoch abort;
there is no callback or thread-safe concurrent mutation contract. Exceptions
before commit leave prototypes, counters and schedule position unchanged.

## Acceleration, scratch scope, and numerics

Ordinary nearest-unit assignment uses blocked SciPy cdist without a broadcast
N*M*D tensor. Batch computes per-BMU sums and counts, then performs separable
x/y Gaussian smoothing using NumPy matrix products. This costs O(NMD) distance
work, O(ND) grouping, and O((width+height)MD) smoothing; storage for kernels is
width²+height² instead of M². It does not change the frozen-BMU update equation.
No mandatory dependency is added and Numba is not needed for these algorithms.

ExecutionPolicy bounds conservatively budgeted primary NumPy algorithm scratch:
blocked distances, grouped/smoothed arrays, kernels, and cold weight rows. It does
not bound process RSS. Owned inputs/state, returned snapshots, initialization
SVD/LAPACK, validation/library workspaces and Python integer/Fraction temporaries
are excluded. The result reports primary_scratch_budgeted_bytes and scratch_rows.
A budget unable to fit one distance row and fixed scratch is rejected up front.

Finite magnitudes outside ordinary safe ranges use exact binary-integer squared
BMU ordering and protected means. Online updates use convex interpolation, with rational arithmetic for the
cold path and strongly cancelling ordinary coordinates before float64 rounding. Strong signed cancellation and tiny
positive mass in an ordinary batch result trigger local raw-sample weighted-mean
repair. Such repairs can be much slower. An unrepresentable PCA/output update
raises ValueError rather than returning silent NaN or infinity.

These protections are not an arbitrary-precision SOM promise: prototypes and
Gaussian weights are still float64; ordinary reductions and BLAS may differ by
roundoff, and nearly tied BMUs can be sensitive. Inputs with very wide dynamic
ranges may lose small SVD components. PCA bases in repeated eigenspaces are not
unique, so Python/JavaScript PCA parity is not promised. Reuse explicit identical
initial_prototypes for trajectory comparisons. Shared nondegenerate fixtures
compare every update and final labels within stated numerical tolerances.

## Validation boundary

Tests include independent scalar online/batch references, frozen BMUs, tie and
empty-unit rules, schedules, ownership, cancellation/resume, all-zero/duplicate
inputs, arbitrary dimensions, PCA degeneracy, import order, scratch rejection,
subnormal/extreme finite inputs and shared Python/JavaScript fixtures.
See the companion validation and benchmark files for exact commands and results.
Only Linux x86-64 CPython 3.12 is checked here; no minimum-dependency, other-OS,
universal timing, convergence or general cross-language bit-identity claim.
