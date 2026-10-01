# Rough Membership C-means (RMCM)

RMCM is a separate algorithm from the existing RCM/ExRCM implementations. This
module implements the fixed Euclidean δ-neighborhood algorithm:

- A[i,j] = 1 when the direct Euclidean distance between original rows i and j is
  at most δ. The relation includes self and includes duplicate rows when δ = 0.
- P[i,j] = A[i,j] / degree[i], fixed for the entire fit.
- H is the one-hot nearest-center assignment of the original X, using the
  lowest center index on equal direct squared distances.
- R = P H. The center update is (Rᵀ X) / (Rᵀ 1), without a fuzzifier or exponent.
- Empty centers retain their preceding values.

The default `adjoint` backend precomputes Y = Pᵀ X and s = Pᵀ 1 once, then
aggregates Y and s by H. The `reference` backend explicitly constructs R and
computes its weighted centers every iteration. The transpose is essential on
irregular-degree graphs: P is generally not symmetric even though A is.

## Basic API

```js
import { rmcm, rmcmReference } from './src/index.js';
const input = {
  data: Float64Array.of(0, 1, 3),
  nSamples: 3,
  nFeatures: 1,
};
const options = {
  delta: 2,
  initCenters: Float64Array.of(0, 3),
  maxIterations: 1,
};
const result = rmcm(input, options);
// centers: [1, 2.2]
// labels: [0, 0, 1]
// membership (row-major samples × clusters): [1, 0, 2/3, 1/3, 1/2, 1/2]
const reference = rmcmReference(input, options);
```

`run('rmcm', input, options)`, `steps('rmcm', ...)`, `runAsync('rmcm', ...)`, and
`createWorkerClient().run('rmcm', ...)` use the existing UbuKit JS interfaces.
`rmcmSteps` is also exported. New calls do not affect the RCM/ExRCM behavior.
Do not pass RCM/FCM-specific parameters such as `p`, `alpha`, `beta`, or `m`.

### Preparation and repeated starts

```js
import { prepareRMCM } from './src/index.js';
const prepared = prepareRMCM(input, { delta: 2, backend: 'adjoint' });
const first = prepared.fit({ nClusters: 2, seed: 1 });
const second = prepared.fit({ nClusters: 2, seed: 2, returnMembership: false });
```

Preparation copies the input, constructs CSR, and prepares the selected backend.
The graph and input are fixed and privately held. `prepared.fitSteps(options)`
provides cooperative fit iteration. `prepareRMCMSteps(input, options)` makes
preparation cooperative too. `degrees` and `neighborhoodMatrix()` return copies;
the latter provides CSR `indptr`, `indices`, `data`, and `shape`.

| Option | Default / meaning |
| --- | --- |
| delta | Required finite nonnegative Euclidean radius |
| nClusters | Required unless inferred from initCenters; 1 through N |
| initCenters | Optional numeric TypedArray, row-major K×D |
| seed | 0; distinct sample-index initialization with JS Mulberry32 |
| backend | `'adjoint'` or `'reference'` |
| maxIterations | 100; positive integer |
| cycleWindow | 32 previous complete states; 0 disables cycle detection |
| returnMembership | true; false omits final dense R from the result |
| maxEdges | 10,000,000 directed edges including self, maximum Uint32 range |
| maxMemoryBytes | 512 MiB estimated primary owned numeric arrays |
| blockRows | 128 rows between fit/precompute checkpoints |
| graphBatchPairs | 4096 pair checks between graph checkpoints |
| onProgress / signal / shouldCancel | Existing sync/cooperative progress and cancellation |

For Worker calls use `AbortSignal`, not `shouldCancel` (functions cannot be
serialized). Worker cancellation terminates the active Worker and the client can
be reused. The preparation phase sends progress before clustering begins.
The existing `transferInput` behavior remains available.

JS and NumPy seeds are different random generators. Cross-language checks must
use the same explicit center rows, not merely equal seed numbers.

## Stopping and returned-state contract

A complete state consists of the hard labels **and** updated center values.
Consecutive equal states yield `stopReason: 'fixed_point'` and `converged: true`.
A repeated complete state in the finite history window yields `stopReason:
'cycle'`, `converged: false`, and a positive `cycleLength`. Reaching the iteration
limit yields `stopReason: 'max_iter'`, `converged: false`. A bounded history can
miss longer cycles; maxIterations remains the hard bound. There is no
tolerance-based convergence test or objective-monotonicity claim.

The returned labels and R are the assignment and memberships that **produced**
the returned centers. There is no hidden final nearest-center reassignment. This
matters at the iteration limit and on a cycle. `membership` is an N×K row-major
Float64Array, or null when omitted; `centers` is K×D and `labels` is Int32Array(N).
Results also include `iterations`, `initIndices` (null for explicit centers),
`nEdges`, `emptyClusterUpdates`, and the shape/backend fields.

Real cycles occur even without empty clusters. For X = [[-1,-3], [5,-3], [3,4],
[-1,-2], [1,-3], [2,2]], δ = 7.3 and centers [[1,-3], [2,2]], both JS backends
report a period-2 cycle at iteration 3. They do not call this convergence.

## Numeric and scale limits

Distance comparisons use direct differences, not norm-expansion shortcuts.
Nonfinite, unsafe-magnitude, overflowing, subnormal-squared, or underflowed
nonzero distances fail explicitly; rescale X and δ in those cases. Exact boundary
comparisons do not expand δ by an epsilon. Cross-language reduction differences
can still change boundary inclusion in last-bit cases.

Adjoint and explicit-R updates are equal in real arithmetic; they are not
promised to be bitwise identical in floating point. Near-tied center distances
can turn tiny reduction differences into different labels, stopping reasons, or
trajectories. No tolerance-based tie rule is substituted. If the graph is
entirely complete, every occupied center uses one shared global-mean reduction
on both backends; this avoids artificial tie differences in that special case.

A concrete finite-precision diagnostic is X = [[1,-1], [4,5], [-3,-4], [1,3],
[-5,3], [2,2], [-2,-2], [100,100]], δ = 12, initialized from rows 0, 1 and 7.
The first seven rows form a clique, and the final row is isolated. Tiny rounding
differences around mathematically coincident centers cause the explicit JS
reference to report a fixed point at iteration 3 while the adjoint reports a
period-2 cycle at iteration 4 on the tested runtime. This is a documented
floating-point limitation, not an assertion that both trajectories must match.

The browser graph builder uses two direct pair passes: O(N²D) time and O(ND+E)
storage, with no dense N×N adjacency or distance matrix. It is not a KD-tree.
`maxEdges` and `maxMemoryBytes` guard allocation; they do not bound runtime.
The adjoint fit needs O(NKD+ND) arithmetic per iteration, versus the reference's
O(NKD+E+NKD) assignment/membership/center work. Preparation adds O(ED) for Y.
Requesting final R costs O(E+NK) time and O(NK) output memory for either backend.
Reference needs R internally even when it is omitted from the result.
Memory estimates exclude JavaScript VM, garbage collector, and object overhead.

## Validation

The candidate passes 128 Node tests: 70 RMCM checks plus the existing 58-test
suite. RMCM coverage includes 27 shared explicit-init Python fixtures on both
backends, two exact-cycle fixtures, boundary/duplicate/full/empty cases,
update-producing outputs, memory/numeric guards, CSR properties, snapshots,
cooperative cancellation, Worker parity/progress/cancellation/reuse, and transfer.
The algebra and state contract received a separate focused review.

Run `npm test`, `node examples/rmcm.js`, or regenerate the tiny Python fixtures with
`python tools/generate-python-fixtures.py` (requires the adjacent Python RMCM package and its dependencies).
`examples/rmcm-browser-smoke.html` is a ready browser test. Actual browser execution
has not been verified: the available local preview route was blocked. Node Worker
checks are not represented as browser validation.
