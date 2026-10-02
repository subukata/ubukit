# UbuKit JavaScript: private reviewed candidate

This local-only npm package revises the selected integrated acceleration candidate
with scoped review fixes. It is not a published release. The revision fixes
Buffer snapshot ownership, validates execution options before replacing session
state, and rescales endangered FCM weight columns before weighted-coordinate
underflow. SOURCE_MANIFEST.json records the two changed runtime files and their
original and revised hashes; the other 19 runtime files are unchanged.
`private: true`, the existing package name `ubukit-js`, version `0.1.0`, and the
five existing package entrypoints are retained. The name/version are not a new
public naming or release-policy decision. No project-wide license is selected;
see NOTICE.txt and the retained SOM attribution in LICENSE-SOM.txt.

## Install locally

```sh
npm install --offline --ignore-scripts --no-audit --no-fund ./ubukit-js-0.1.0.tgz
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

## Existing supported package entrypoints

- `ubukit-js`: functions, helpers, prepared RMCM/cache objects, generators, sessions,
  Worker clients and the metric scheduler exported by `src/index.js`
- `ubukit-js/worker`: `createWorkerClient`
- `ubukit-js/session`: `createSession`, `ClusteringSession`, `sessionAlgorithms`
- `ubukit-js/realtime-worker`: `createRealtimeWorkerClient`
- `ubukit-js/metrics`: `createMetricScheduler`

The root algorithms registry contains `kmeans`, `fcm`, `rcm`, `exrcm`, `rmcm`,
`som-olp`, and `neighborhood`. `run`, `steps` and `runAsync` share that registry.
Neighborhood computes trustworthiness and continuity; it is a one-shot metric,
not an iterative fitting session. No JavaScript ARI or AMI API is included.
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

The reviewed FCM fallback handles both uniformly and unevenly tiny membership
weights, including weighted-coordinate products that underflow before division.
It retains the existing strict squared-distance underflow errors. In one extreme
`m: 1000` regression fixture, recovering a previously truncated center displacement
now exposes that error instead of returning the historical finite result. This
intentional numerical-domain change is recorded in the separate review/test bundle.
No universal floating-point range or translation-rounding workaround is promised.

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
