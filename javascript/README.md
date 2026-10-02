# UbuKit for JavaScript

Clustering, self-organizing maps (SOM), evaluation metrics and parameter search for
Node.js and browsers, using ES modules and no runtime npm dependencies.

The package is `private: true` and unpublished. Project contributions use the
[MIT License](LICENSE); [license scope](LICENSE-SCOPE.txt) identifies third-party
terms. Project-developed WebAssembly (WASM) kernels are included. See the
[build instructions](../docs/getting-started.md#javascript) to create a local tarball from source.

## Install locally

```sh
npm install --offline --ignore-scripts --no-audit --no-fund ./ubukit-js-0.1.0-dev.6.tgz
```

There are no runtime npm dependencies or install scripts. Package metadata keeps
the existing Node >=20 requirement. This integration was executed on Linux with
Node 24.19.0; that does not validate all Node >=20 versions or other OSes.

This example uses fuzzy c-means (FCM) for degrees of cluster membership.

```js
import { run, createSession } from 'ubukit-js';
const input = {
  data: Float64Array.of(0, 0.1, 4, 4.1),
  nSamples: 4,
  nFeatures: 1
};
const result = run('fcm', input, { nClusters: 2, seed: 1, maxIterations: 10 });
const session = createSession('fcm', input, { nClusters: 2, seed: 1 });
while (!session.status.done) session.step(1);
const snapshot = session.snapshot();
session.dispose();
```

## Supported package entrypoints

- `ubukit-js`: functions, helpers, prepared rough membership c-means (RMCM)
  graphs/caches for fixed-radius neighborhoods, generators, sessions, Worker
  clients and the metric scheduler exported by `src/index.js`
- `ubukit-js/worker`: `createWorkerClient`
- `ubukit-js/session`: `createSession`, `ClusteringSession`, `sessionAlgorithms`
- `ubukit-js/realtime-worker`: `createRealtimeWorkerClient`
- `ubukit-js/metrics`: `createMetricScheduler`
- `ubukit-js/optimization`: optimizer and search-space helpers; see OPTIMIZATION.md
- `ubukit-js/external-metrics`: adjusted Rand index (ARI), adjusted mutual information
  (AMI) and shared-contingency joint scores for chance-adjusted cluster-label agreement;
  see EXTERNAL_METRICS.md

Rough c-means (RCM, `rcm`) and extended rough c-means (ExRCM, `exrcm`)
allow overlapping cluster assignments. Self-organizing maps with optimized
latent positions (SOM-OLP, `som-olp`) learn continuous sample positions using
a supplied grid.

The root algorithms registry contains `kmeans`, `fcm`, `entropy-fcm`, `rcm`, `exrcm`, `rmcm`,
`som-olp`, `som`, `som_batch`, and `neighborhood`. `run`, `steps` and `runAsync` share that registry.
Neighborhood computes trustworthiness and continuity; it is a one-shot metric,
not an iterative fitting session. ARI and AMI are separate exported scoring
functions: `adjustedRandScore`, `adjustedMutualInfoScore`, and `adjustedScores`;
see [EXTERNAL_METRICS.md](EXTERNAL_METRICS.md).
Online SOM (`som`) commits one sample update; batch SOM (`som_batch`, also called
BatchSOM) commits one epoch with each sample's best-matching unit (BMU) held fixed.
See [SOM.md](SOM.md) for step units and explicit current-model projection.
Data is flat row-major with explicit dimensions. The result layouts below describe the existing API.

## Core inputs and results

Call `run(name, input, options)` with the `{data, nSamples, nFeatures}` input
shown above. `N` is the sample count, `D` the feature count, `K` the cluster
count, `M` the map-unit count and `Q` the grid dimension. Shapes below describe
flat row-major typed arrays: `(N,K)` means `N*K` values, and sample `i`, cluster
`c` is `membership[i*K+c]`. Labels have length `N` and use zero-based indices.

| Algorithm name | Options to start with | Main final result fields and logical shapes |
| --- | --- | --- |
| `kmeans` | `{ nClusters: K }` | `centers` `(K,D)`, `labels` `(N,)` |
| `fcm` | `{ nClusters: K }`; optional `m` defaults to 2 | `centers` `(K,D)`, `membership` `(N,K)`, `labels` `(N,)` |
| `entropy-fcm` | `{ nClusters: K, tau: 1 }` | `centers` `(K,D)`, `membership` `(N,K)`, `labels` `(N,)`; see [ENTROPY_FCM.md](ENTROPY_FCM.md) |
| `rcm`, `exrcm` | `{ nClusters: K }`; `rcm` fixes `p=1` | `centers` `(K,D)`, `membership` and `mask` `(N,K)`, `labels` `(N,)` |
| `rmcm` | `{ nClusters: K, delta: radius }` | `centers` `(K,D)`, `membership` `(N,K)`, `labels` `(N,)`; membership is `null` with `returnMembership: false` |
| `som-olp` | `{ grid }`, where `grid` uses the same input-object format for `M` grid points with `Q` coordinates | `centers` / `W` `(M,D)`, `membership` / `P` `(N,M)`, `embedding` / `V` `(N,Q)`, `labels` `(N,)`; embedding is `null` when `maxIterations: 0` |
| `som`, `som_batch` | Optional `{ gridShape: [width, height], epochs: 10 }`; default grid is 16×16 | `centers` `(M,D)`, `labels` `(N,)`, `embedding` `(N,2)`; see [SOM.md](SOM.md) |
| `neighborhood` | `{ embedding, ks: [1] }`, with `embedding` in the same input-object format and the same `N`; each `k` must satisfy `1 <= k < N/2` | `qualities`: records containing `k`, `trustworthiness` and `continuity` |

FCM and rough-clustering `membership` values are normalized per sample; the
rough `mask` is a separate binary admissibility array. K-means returns nearest
final-center labels. FCM, RCM, ExRCM and SOM-OLP labels select the first maximum
membership; RMCM preserves the hard labels that produced its returned centers,
without a final reassignment. These label meanings are not interchangeable.

ARI/AMI take label arrays directly, outside `run`; see
[external metrics](EXTERNAL_METRICS.md). Tree-structured Parzen Estimator (TPE)
search uses earlier trial results to suggest parameter values; random search
samples without that feedback. Both take an objective and search space; see
[optimization](OPTIMIZATION.md).

## Sessions, updates and Workers

Entropy-regularized FCM supports the one-shot and cooperative APIs and one-shot
Worker. Stateful sessions and the realtime Worker are outside its first scope;
see [ENTROPY_FCM.md](ENTROPY_FCM.md).

Sessions expose `step`, `snapshot`, `updateData`, `updateParameters`, `configure`,
`reset` and `dispose`. Compatible warm state may be retained; changed data,
shape, parameters and point IDs follow existing invalidation rules. Snapshot
arrays are owned copies. Cancellation is available through AbortSignal or the
existing callback hooks. After a cancelled session, update its aborted signal
before resuming/resetting as appropriate. Do not assume a reset un-aborts a signal.

The Worker clients retain their default URLs, resolved relative to their own
installed modules: `src/worker.js` and `src/realtime-worker.js`. Those files are
included in the explicit tarball allowlist. Browser use requires serving the ES
modules from an appropriate origin or a bundler that preserves module Worker URLs.
There are no public `worker-entry` aliases.

In Node, use the existing `workerFactory` option with `node:worker_threads`.
A Worker URL can be derived from the corresponding exported client module:

```js
import { Worker } from 'node:worker_threads';
import { createWorkerClient } from 'ubukit-js/worker';
const workerURL = new URL('./worker.js', import.meta.resolve('ubukit-js/worker'));
const client = createWorkerClient({
  workerFactory: () => new Worker(workerURL, { type: 'module' })
});
try {
  const result = await client.run('fcm', input, { nClusters: 2, seed: 1 });
} finally {
  client.dispose();
}
```

The one-shot Worker permits one job at a time. Cancelling it terminates the Worker
and permits reuse. The realtime Worker permits one in-flight request and one
coalesced latest pending request; superseded promises reject with AbortError.
`transferInput: true` in the one-shot client transfers input buffers and detaches
them from the sender. Default use does not transfer the caller's input buffers.
The metric scheduler supports cancellation, debounce, result-copy caching,
`clearCache` and disposal.

## FCM numerical behavior

FCM keeps the reviewed ordinary Float64 kernels. Large `m` (above 32), near-one `m` (m-1 < 1e-4), extreme
coordinate or membership scales, and numerical range failures use a finite-m
log-domain fallback. Centroid weights are computed directly from stable log
squared-distance ratios; the common `-m log K` is removed analytically, before
rounding memberships or exponentiating. This is not the `m = infinity` limit
and does not clip distances. Exact zero-distance ties still split equally.
Nonzero distances whose squares underflow or overflow retain finite logarithms.

The fallback adds `numericalMode: 'log-domain'`, `objectiveLog`,
`objectiveRepresentation` (`finite`, `underflow`, `overflow`, or `exact_zero`),
`centerRelativeDelta`, `membershipResolutionLostRows`, and
`membershipConvergenceOnly`. An unrepresentable raw
objective is reported as 0 or Infinity together with its explicit status and
logarithm; it is never clipped. At the largest m the logarithm itself may
be unrepresentable, but positive terms still have `underflow` status rather than
`exact_zero`. `membershipResolutionLostRows` counts nonuniform log-membership
rows that round to uniform public memberships; labels still use public-U argmax. The objective is evaluated from the returned
Float64 memberships and exactly the published centers, matching the prior
contract. Restoring a large coordinate offset can round centers, so returned U
need not equal the membership formula evaluated at the published rounded centers. Floating-point centers can themselves round onto a data point; exact
zero rules then apply to that represented state.

The default stopping rule remains the existing absolute membership difference.
At huge m, rounded memberships can stop changing while centers still move.
`membershipConvergenceOnly` flags this situation when relative center movement
exceeds the membership tolerance. Use `tolerance: 0` with an explicit iteration
budget when studying this regime; no claim of center convergence is implied.

Sessions keep private log-membership state across same-data parameter updates,
including changes to m, so warm restarts do not reconstruct weights from rounded
memberships. Public snapshots still own their arrays. Moving/reordering data
uses the existing center-based restart contract. Fallback retry chunks remain
cooperative and do not replay completed iteration progress/checkpoints. The
memory estimate includes the additional log state; time budgets remain soft.

The numerical checks cover Node workers. Browser execution has not been
qualified by these checks.

## Optional kernels and limits

Defaults remain JavaScript/scalar. Existing opt-in options are preserved:

- kmeans and SOM-OLP: `kernelBackend: 'wasm'`
- RMCM graph: `graphBackend: 'wasm-simd'`
- neighborhood distances: `distanceBackend: 'wasm'`

WASM bytes are embedded in the runtime modules; no separate `.wasm` fetch or
build dependency is required. Existing capability, memory and numerical fallback
logic remains unchanged. Requested WASM is not a promise that every operation
executes in WASM. Read existing result diagnostics, where provided.

Other OSes, architectures and actual browser execution are not qualified by
these checks. Local validation does not authorize registry publication.

## SOM-OLP exceptional numerical ranges

SOM-OLP selects an internal JavaScript recovery path for extreme coordinate or gamma
ranges, including when `kernelBackend: 'wasm'` is requested. Its diagnostics report
the actual fallback. The ordinary JavaScript and embedded WASM kernels retain
their previous arithmetic. Exceptional recovery uses original-unit costs,
exact binary cold weighted-product accumulation, gamma-weighted scaled norms, and a centered
principal component analysis (PCA) scratch copy. The latter is included in
memory/scratch budget checks.

This is a bounded numerical extension, not support for every finite input.
Positive composite costs that become zero or overflow raise a RangeError rather
than producing false ties or an approximate hard/uniform assignment. Representable
subnormal costs still have float64 precision. Cold means use bounded BigInt binary product accumulators before final float64
conversion, preserving signed cancellation across extreme exponents. Cold
objectives use the equivalent minimized-row formula min(cost)-lambda*logZ,
with a log1p tail and binary signed accumulation; this retains entropy effects
when the published largest probability rounds to one. Unrepresentable final objectives
still raise. These scalar heap temporaries are outside the primary typed-array
scratch accounting. Lambda and gamma are never clipped or rescaled, and history
plus stopping tests remain in the original objective units. When subtracting
two finite objectives would overflow, the same relative stopping ratio is
evaluated by dividing before subtraction.


## Lightweight optimization

The dependency-free optional TPE optimizer and random baseline are exported from the root and `ubukit-js/optimization`. See OPTIMIZATION.md for examples, numerical contracts, and limitations.

## External clustering agreement

`adjustedRandScore`, `adjustedMutualInfoScore`, and `adjustedScores` are exported
from the package root and `ubukit-js/external-metrics`. The joint API shares one
contingency table. Additional runtime dependencies: none. See
[EXTERNAL_METRICS.md](EXTERNAL_METRICS.md) for input types, normalization choices,
numerical policy, supported limits, and error codes. Definitions follow
scikit-learn's arithmetic-default ARI/AMI conventions; stable high-K answers
intentionally do not reproduce its floating-point artifacts.

The module has no Node-only imports and is designed for modern browsers.
Installed-tarball checks cover Node; actual browser execution remains unverified.

## Online SOM and BatchSOM

The `som` and `som_batch` algorithms support arbitrary feature dimensions,
16×16 rectangular grids, sample/PCA initialization and stateful sample/epoch updates.
See [SOM.md](SOM.md) for exact batch equations, schedules, realtime projection,
complexity and finite-range limits. No runtime dependencies were added.

### k-means WASM center cache

For `kernelBackend: "wasm"`, `wasmCenterCache: true` (default) prepares the center
transpose once per iteration and refreshes it after restoring final centers.
`wasmCenterCache: false` retains the old per-block preparation path. Workspace
size, numerical operations, events, transferability and all fallback rules are
unchanged. This is a copy-reduction change; the embedded WASM bytes are identical.

A checkpoint hook can expose mutable centers. If one is present initially or
installed later, the cache is disabled permanently before it receives centers,
even if the hook is subsequently removed. Sessions therefore retain per-block
refreshing. Only calls with unexposed, internally owned centers reuse transposes.
Small row blocks and wider center matrices benefit most; large row blocks can
be neutral. See `SOM.md` for the separate classic-SOM BMU route selector.
