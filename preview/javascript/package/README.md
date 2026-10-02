# UbuKit JavaScript: dev6 private efficiency candidate

This repository snapshot contains the reviewed JavaScript candidate
`0.1.0-dev.6`, with 27 runtime files, seven package entrypoints and nine
algorithm/metric registry names. It includes
FCM/SOM extreme-range repairs, snapshot ownership and session validation fixes,
optional lightweight TPE/random optimization, ARI/AMI external metrics, and
traditional online SOM plus true BatchSOM. SOURCE_MANIFEST.json binds
the current runtime files to SHA-256 hashes.

The package remains `private: true` and unpublished. No project-wide license
has been selected; retain NOTICE.txt and LICENSE-SOM.txt. See ../REPRODUCE.md in the repository preview to build and
test from source. The earlier research trees
in the repository are historical, not this package's current implementation.

## Install locally

```sh
npm install --offline --ignore-scripts --no-audit --no-fund ./ubukit-js-0.1.0-dev.6.tgz
```

There are no runtime npm dependencies or install scripts. Package metadata keeps
the existing Node >=20 requirement. This integration was executed on Linux with
Node 24.19.0; that does not validate all Node >=20 versions or other OSes.

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

- `ubukit-js`: functions, helpers, prepared RMCM/cache objects, generators, sessions,
  Worker clients and the metric scheduler exported by `src/index.js`
- `ubukit-js/worker`: `createWorkerClient`
- `ubukit-js/session`: `createSession`, `ClusteringSession`, `sessionAlgorithms`
- `ubukit-js/realtime-worker`: `createRealtimeWorkerClient`
- `ubukit-js/metrics`: `createMetricScheduler`
- `ubukit-js/optimization`: optimizer and search-space helpers; see OPTIMIZATION.md
- `ubukit-js/external-metrics`: ARI, AMI and shared-contingency joint scores; see EXTERNAL_METRICS.md

The root algorithms registry contains `kmeans`, `fcm`, `rcm`, `exrcm`, `rmcm`,
`som-olp`, `som`, `som_batch`, and `neighborhood`. `run`, `steps` and `runAsync` share that registry.
Neighborhood computes trustworthiness and continuity; it is a one-shot metric,
not an iterative fitting session. ARI and AMI are separate exported scoring
functions: `adjustedRandScore`, `adjustedMutualInfoScore`, and `adjustedScores`;
see [EXTERNAL_METRICS.md](EXTERNAL_METRICS.md).
Traditional `som` commits one sample update; `som_batch` commits one frozen-BMU
epoch. See [SOM.md](SOM.md) for step units and explicit current-model projection.
Data is flat row-major with explicit dimensions. Algorithm-specific result shapes,
label contracts, error handling and ownership remain unchanged.

## Sessions, updates and Workers

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
There are no new public `worker-entry` aliases.

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

## Numerical correction and compatibility note

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

Other algorithms retain their reviewed numerical policy. Browser execution has
not been reverified by this numerical change; tests cover real Node workers.

## Optional kernels and limits

Defaults remain JavaScript/scalar. Existing opt-in options are preserved:

- kmeans and SOM-OLP: `kernelBackend: 'wasm'`
- RMCM graph: `graphBackend: 'wasm-simd'`
- neighborhood distances: `distanceBackend: 'wasm'`

WASM bytes are embedded in the runtime modules; no separate `.wasm` fetch or
build dependency is required. Existing capability, memory and numerical fallback
logic remains unchanged. Requested WASM is not a promise that every operation
executes in WASM. Read existing result diagnostics, where provided.

SOURCE_MANIFEST.json identifies the immutable input and every packaged runtime
hash. The separate integration verification bundle contains tests, comparison
snapshots, logs and a report; these are intentionally excluded from the npm tarball.
Testing does not imply new performance measurements, Windows/macOS/ARM coverage,
or real-browser verification. Consult the integration report for checks actually
run. Registry publication and deployment still require separate authorization.


## SOM-OLP exceptional numerical ranges

SOM-OLP selects an internal JavaScript recovery path for extreme coordinate or gamma
ranges, including when `kernelBackend: 'wasm'` is requested. Its diagnostics report
the actual fallback. The ordinary JavaScript and embedded WASM kernels retain
their previous arithmetic. Exceptional recovery uses original-unit costs,
exact binary cold weighted-product accumulation, gamma-weighted scaled norms, and a centered
PCA scratch copy. The latter is included in memory/scratch budget checks.

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

## External clustering agreement (introduced in dev4)

`adjustedRandScore`, `adjustedMutualInfoScore`, and `adjustedScores` are exported
from the package root and `ubukit-js/external-metrics`. The joint API shares one
contingency table. Additional runtime dependencies: none. See
[EXTERNAL_METRICS.md](EXTERNAL_METRICS.md) for input types, normalization choices,
numerical policy, supported limits, and error codes. Definitions follow
scikit-learn's arithmetic-default ARI/AMI conventions; stable high-K answers
intentionally do not reproduce its floating-point artifacts.

Node execution is tested from an installed tarball. The module has no Node-only
imports and is designed for modern browsers; an actual browser smoke run could
not be completed because the cloud browser blocked the local test URL.

## Traditional SOM and BatchSOM (dev5)

Additive `som` and `som_batch` algorithms support arbitrary feature dimensions,
16×16 rectangular grids, sample/PCA initialization and stateful sample/epoch updates.
See [SOM.md](SOM.md) for exact batch equations, schedules, realtime projection,
complexity and finite-range limits. No runtime dependencies were added.

### Private efficiency candidate: k-means WASM transpose reuse

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
