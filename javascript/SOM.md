# Traditional SOM and BatchSOM

The additive registry names are `som` and `som_batch`. They are separate from
SOM-OLP (`som-olp`) and from neighborhood quality evaluation (`neighborhood`).
All earlier APIs, ARI/AMI and TPE optimization remain available.

```js
import { som, somBatch, createSession, somProject } from 'ubukit-js';
const input = { data: Float64Array.of(0, 0, 1, 1, 3, 2),
  nSamples: 3, nFeatures: 2 };
const options = { gridShape: [16, 16], initializer: 'sample', seed: 0,
  epochs: 10, sigma: 8, sigmaEnd: .5 };
const online = som(input, { ...options, learningRate: .5, learningRateEnd: .05 });
const batch = somBatch(input, options); // also exported as som_batch
const session = createSession('som', input, options);
const status = session.step(1, { timeBudgetMs: 8, maxChunks: 1024 });
const snapshot = session.snapshot(); // latest committed centers, no partial model
const projected = session.snapshot({ project: true }); // explicit O(N*M*D) cost
// Or project arbitrary same-dimensional rows onto any committed model:
if (snapshot.result) somProject(input, snapshot.result);
```

## Mathematical contract

- Input is finite row-major numeric TypedArray data, N > 0, D > 0. D is arbitrary
  within explicit memory/PCA limits. All fitting is in full D dimensions.
- Default rectangular grid is `[width, height] = [16,16]`. Unit j has lattice
  coordinates `(j % width, floor(j / width))`. The map is non-periodic, x-fast,
  in integer lattice units, **not** SOM-OLP's caller-supplied normalized grid.
- BMU minimizes squared Euclidean feature distance. Exact ties choose the lowest
  unit index. Gaussian neighborhood is `exp(-latticeDistanceSquared/(2*sigma^2))`.
  Sigma zero is defined as a BMU-only kernel; Gaussian underflow gives zero.
- Online: one committed iteration consumes one row, in sequential cyclic order,
  and applies `w[j] += eta*h[j,bmu]*(x-w[j])`. A protected equivalent convex
  blend preserves endpoint contributions and repairs cancellation-sensitive sums. `epochs*N` updates by default.
  There is no shuffling in this version.
- Batch: one committed iteration is one complete epoch. Every BMU is determined
  against the same frozen old prototypes. New prototypes are the Gaussian
  weighted means of all data. Empty/zero-mass units retain their old prototype.
  This is a batch update, never sequential online training renamed as batch.
  Learning-rate parameters are rejected for BatchSOM.
- Batch groups samples into mean/count sufficient statistics by frozen BMU and
  applies separable horizontal/vertical Gaussian weighted-mean smoothing.
  Normalized mean merging avoids raw weighted-numerator overflow/underflow;
  exceptional merges use the existing exact-binary accumulation helper.
  High-dynamic-range data and cancellation-sensitive signed coordinates also
  receive a cold original-sample mean repair so
  rounded group means cannot erase cancellation residuals. This cold path costs
  O(N*M*D) per epoch and uses an extra O(N) weight vector.
- There is no convergence test or SOM-OLP objective. `converged` stays false;
  completion means the configured schedule finished.

## Options and schedules

`epochs` defaults to 100 and may be zero. `maxIterations`, when supplied,
overrides the number of committed units: samples for online, epochs for batch.
The online sequence starts at row 0 on every fresh/reconfigured schedule.

Sigma defaults to `max(width,height)/2`, ending at `min(.5,sigma)`. Online eta
(`learningRate`) defaults to .5, ending at `min(.05,learningRate)`. Endpoints must
be finite, nonnegative and nonincreasing; learning rates are in [0,1].
`schedule:'geometric'` (default) interpolates logarithmically, except a zero
endpoint uses linear interpolation. `schedule:'linear'` interpolates directly.
For T > 1, update t uses fraction t/(T-1), so both endpoints are included.
For T = 1, it uses the start value. T = 0 performs initialization/projection only.

`initialPrototypes` (alias `initCenters`) must contain M*D finite values and is
copied. Otherwise `initializer:'sample'` (default) samples rows with replacement
using Mulberry32 and uint32 `seed` (default 0), shared with Python's new SOM API.
`initializer:'pca'` uses a deterministic JS Jacobi eigensolver, population
covariance, sign-normalized loadings and lattice coordinates normalized to [-1,1]
for initialization only; `pcaScale` defaults to 2 and is nonnegative. PCA/NumPy
bit parity or identical bases in degenerate eigenspaces is not promised. Explicit
shared prototypes are the strongest cross-language training parity contract. Near-BMU-ties can choose different winners after
  floating-point summation differences, then diverge across multiple epochs;
  training parity fixtures therefore avoid ambiguous near ties. No epsilon tie
  rule is added to alter nearest-prototype semantics.
`pcaMaxDimension` defaults to 128; use sample initialization or supplied prototypes
for larger min(N,D), or deliberately raise the cap.

## State, realtime controls and ownership

`somSteps`/`somBatchSteps` are cooperative generators. `run`, `steps`, `runAsync`,
`createSession`, one-shot workers and realtime workers accept both registry names.
`iterationUnit`/`unit` are `sample` or `epoch`; `samplesSeen`, `epochsCompleted`
and `maxIterations` disambiguate their counters. Returned `centers`, `prototypes`
and `W` alias the same owned numerical array.

Each complete sample/epoch makes current prototypes available immediately through
session snapshots. Partially built next prototypes remain private across time
budgets, cancellations and parameter/data revisions. `step(count)` requests at
most count committed updates but may return earlier at a chunk budget. Default
`blockRows=128`, `blockUnits=32`. Budgets are **soft**, not hard realtime: one row's
BMU scan and one chunk can exceed the requested time. Use workers for large D/N.

Ordinary intermediate snapshots have `labels:null`, `embedding:null` and
`projectionStatus:'not-computed'`. Final results contain fresh BMU labels and
2-D lattice embeddings. `snapshot({project:true})` or `somProject(input,model)`
explicitly computes current-prototype projection at O(N*M*D) cost, without
advancing training or mutating the session; it reports `current-prototypes`.
This makes a current prototype mesh available every update without silently
recomputing every sample label or presenting stale labels as fresh.

Sessions own input/configuration and return copied snapshots. Direct iterators
copy numeric inputs on their first advancement, so do not mutate arrays before
that first call. Parameter/data changes abandon unfinished work, retain compatible
committed prototypes when warmStart is true, and restart the finite schedule.
A grid-shape change cold-restarts; reset replays configured initialization.

## Complexity and bounds

Online training costs O(updates*M*D), plus O(N*M*D) final projection; state is
O(N*D + M*D + N). Batch costs O(epochs*(N*M*D + M*D*(width+height))), with state
O(N*D + M*D + width^2 + height^2 + N). Neither allocates N*M memberships or an
N*M*D tensor. The shape of very elongated grids matters to kernel memory.
Defaults cap primary arrays at 512 MiB and scratch at 32 MiB; session budgets
also reserve owned inputs and committed snapshot state. Caller-retained snapshots,
JS object/GC overhead, temporary exceptional BigInt allocations and worker copies
are not process-RSS guarantees.

Finite values are validated, but arbitrary extreme feature ranges are not an
unlimited promise: unrepresentable squared distances (overflow, underflow or
subnormal squared norms) raise RangeError with rescaling guidance rather than
silently invent BMU ties. Constant tiny/huge data are supported. Python's exact
extreme-distance fallback is broader than this first JS implementation. Geometric
kernel underflow is expected zero influence. PCA numerical failures are explicit.
Runtime execution is JavaScript Float64 CPU; no GPU/WASM training backend is
claimed for these new variants. Runtime dependencies added: zero.

Algorithm reference: [SOM Toolbox algorithms](https://www.cis.hut.fi/somtoolbox/documentation/somalg.shtml).
The new implementation is independent; existing SOM-OLP notices remain intact.

## BMU route selection

`bmuBackend` accepts `"auto"` (default), `"scalar"`, or `"grouped"`.
`"scalar"` retains the original direct-distance cutoff implementation.
`"grouped"` uses a dedicated 2D loop or four independent center lanes while
preserving increasing-feature summation, first-index ties, and the original
cutoff-sensitive overflow/underflow exceptions. It allocates no extra scratch.
`"auto"` uses specialized BMUs only for at least 16 units and at most 8 features;
small grids and higher-dimensional inputs retain scalar BMUs. Grouped loops can
win for higher-dimensional dense work but lose when early features exclude
most units. No route is claimed to win on every shape, dataset or JS engine.
This option applies to online SOM, BatchSOM, projection and their sessions.

The scalar elite route retains the complete original dev5 SOM module, byte for
byte, in `src/som-elite.js`. Scalar and high-D/small-grid auto calls dispatch to
that separate module rather than an old BMU body inside a rewritten training
loop. This preserves the original V8 compilation unit as well as the arithmetic.
The additional internal file is included in the package manifest; public root
exports and package entrypoints remain unchanged. Benchmarks verify the actual
packed route rather than assuming byte preservation guarantees identical speed.
