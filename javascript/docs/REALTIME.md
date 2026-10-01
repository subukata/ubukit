# Stateful / realtime-oriented JavaScript API

This API adds retained, interruptible fitting state to the library. It does not
change the historical `run`, `runAsync`, `steps`, `createWorkerClient`, or their
result layouts. It is a cooperative CPU implementation, not a hard-real-time
scheduler, automatic cluster-count selector, GPU backend, or approximation.

## Sessions

```js
import { createSession } from '../src/index.js';
const input = { data: Float64Array.of(0, 0, 1, 1, 9, 9, 10, 10),
                nSamples: 4, nFeatures: 2 };
const session = createSession('fcm', input, {
  nClusters: 2, m: 2, seed: 7, maxIterations: 20, blockRows: 64
});
while (!session.status.done) {
  session.step(1, { timeBudgetMs: 4, maxChunks: 64 });
  // Yield the event loop, or use a Worker, rather than this synchronous loop
  // on the rendering thread. snapshot() is optional and copies output arrays.
}
const first = session.snapshot();
session.updateParameters({ m: 2.5 }); // compatible U/centers warm start
session.updateData(newInput);         // compatible centers, new row memberships
session.dispose();
```

Supported fitting algorithms: `kmeans`, `fcm`, `rcm`, `exrcm`, `rmcm`, `som-olp`.
The constructor copies numerical input/options. Later caller mutations cannot
silently alter the active revision. `step()` is synchronous and must not be
called concurrently/reentrantly; a progress callback may inspect `snapshot()`
but cannot mutate/step/dispose the session until the callback returns.

### Method contract

- `step(count=1, {timeBudgetMs=8, maxChunks=1024})`: advances **at most** `count`
  complete iterations and at most `maxChunks` actual generator work chunks. It
  may return partway through an iteration. Count zero performs no work. The
  return value is lightweight metadata, including `chunks`,
  `completedIterations`, `elapsedMs`, `phase`, `iteration`, and `done`
- `status`: metadata only. `iteration` counts complete iterations in the current
  revision. `totalIterations` counts all completed iterations across revisions
- `snapshot()`: `{...status, result}`. Every numerical output is copied; the
  caller may modify it. Alias relationships such as SOM `W === centers` remain
  true within one snapshot. `result` is `null` before the first complete
  iteration, and the previous complete iteration during partially processed
  work. `done` becomes true only after finalization, not just at an iteration
  boundary. Finalization may require extra `step()` calls
- `updateParameters(patch, {warmStart=true})`: merges a parameter patch and
  starts a new revision. Compatible FCM/SOM retain U and centers; other fitters
  retain centers. Explicit initialization/seed changes and K/grid shape changes
  cold-reset. In particular, changing FCM `m` retains the valid previous U but
  clears objective history, convergence, and iteration numbering
- `updateData(input, {warmStart=true})`: starts a new revision for changed
  values, row order, or shape. Retains centers only when D is unchanged. It
  **never assumes row identity** or carries old U rows into moved/reordered/new
  points, even before the first completed iteration. Births/deaths may change N.
  D changes cold-reset; K remains the configured model parameter. Identical
  content with default warmStart is a no-op
- `configure(input, fullOptions, {warmStart=true})`: atomic replacement of the
  full data and option set, used by the Worker below. Unlike a parameter patch,
  omitted options revert to defaults. It retains compatible U only when X is
  exactly unchanged and the configured initialization is unchanged
- `reset()`: cold restart of the **current data and configured initialization**,
  not a rollback of prior data/parameter updates. Reuses a still-valid prepared
  RMCM graph. `dispose()` releases owned state and is idempotent; subsequent
  fitting/snapshot/update calls fail

All updates validate arguments before replacing live state. They abandon any
partial iteration and preserve only the last complete compatible state. A
compatible warm anchor remains available when several revisions arrive before
any new iteration completes. Ordinary argument errors leave the previous state
unchanged. Data-dependent distance overflow, PCA convergence failures, and graph
edge/memory limits discovered during computation can still fail in `step()`;
these mark that revision failed. Fix parameters/data or reset to continue.

### Boundaries, cost and memory

A time budget is **soft**, checked after each generator chunk. One row, the
initial membership/translation pass, PCA work, a complete-state copy, or final
output work can exceed the requested time. Cold JIT and garbage collection also
contribute. `maxChunks` bounds work-unit count; neither bound is a wall-clock
latency guarantee. The session keeps the actual generator/numerical state while
inputs are unchanged, so splitting a fit does not recompute prior iterations.
On an update the old generator is closed and rebuilt from an explicit compatible
checkpoint; it is never kept running against new mutable input arrays.

Completed iterations make an owned stable checkpoint, including necessary array
copies. `snapshot()` makes another copy on demand. These copies are measured in
`bench/realtime.js`; sessions do not claim allocation-free fitting. Primary
memory guards reserve owned X/options and checkpoint storage in addition to
kernel arrays. Caller-retained snapshots, Worker structured cloning, runtime/JIT,
and GC are outside that primary-array budget. Histories and cycle windows are
bounded by the configured finite iteration/window limits. Use modest
`blockRows`, history/window sizes and output frequency for live work.

## Result shapes and timing

The session envelope is uniform; algorithm-specific historical results remain
compatible. All JS input/centers/memberships are flat row-major typed arrays.
N = samples, D = features, K = clusters/SOM units, Q = SOM grid dimensions.

| Value | Shape | Iteration checkpoint contract |
|---|---|---|
| `centers` | K×D | copied complete update |
| `labels` | N | algorithm's existing assignment semantics below |
| FCM/RCM/ExRCM `membership` | N×K | samples-clusters, not Python ExRCM's C×N |
| RMCM `membership` | N×K or null | final-only; checkpoint is null, including the reference backend |
| SOM `membership`, `memberships`, `P` | N×K | same copied array within a snapshot |
| SOM `centers`, `prototypes`, `W` | K×D | same copied array within a snapshot |
| SOM `embedding`, `V` | N×Q | from the pre-membership-update P |

- k-means checkpoint labels are the assignment that produced its updated
  centers. Final labels are recomputed at published final centers. Inertia and
  objective are null at intermediate checkpoints
- FCM centers come from U_old and membership is U_new. A checkpoint objective
  is the iteration objective on translated internal coordinates; the final
  objective is recomputed using published rounded centers. There is no hidden
  extra center update
- RCM/ExRCM checkpoint membership produced the updated centers; final membership
  is re-evaluated at returned centers. The snapshot's membership contract says
  which is present
- RMCM returns the hard assignment that produced centers, with R = P H, and
  does not add a final nearest-center assignment. A cycle is **not convergence**.
  Fixed point, detected cycle and iteration limit remain distinct
- SOM W/V use pre-update P and returned P is post-softmax. Existing `history`
  semantics and initialization diagnostics are retained. Do not relabel V as
  the result of multiplying the final P by the grid

### RMCM invalidation

The immutable CSR graph and its adjoint preparation are reusable only for the
same X values/shape, delta, backend, and compatible construction limits. Changing
X (including movement), delta, backend, memory/edge limits invalidates the graph.
Changing K/fit limits can reuse the graph but not old cluster-assignment history.
Motion therefore still incurs the exact O(N²D) graph preparation cost; a live
view should use a smaller N, lower compute rate, or explicitly disclose an
alternative model rather than silently reusing stale neighbors.

## Progressive Worker: newest input wins

```js
import { createRealtimeWorkerClient } from '../src/index.js';
const client = createRealtimeWorkerClient();
const result = await client.run('fcm', input, {
  nClusters: 2, m: 2, maxIterations: 20, blockRows: 64,
  onProgress: snapshot => render(snapshot.result, snapshot.requestId)
}, { timeBudgetMs: 4, maxChunks: 64, progressIntervalMs: 30 });
// Calling run() again replaces the previous request; use .catch() for AbortError.
client.cancel(); // cooperatively cancel; worker and valid checkpoint stay reusable
client.dispose();
```

`run(algorithm,input,fullOptions,controls)` returns the final snapshot envelope
with `requestId`. Options are full replacements, not patches. Controls are
`timeBudgetMs`, `maxChunks`, `iterationsPerSlice`, `progressIntervalMs`, and
`warmStart`. The Worker reuses its session for the same algorithm via atomic
`configure()`; changing algorithms replaces it. Input snapshots are owned; this
API intentionally does not support `transferInput:true` or serialized
`shouldCancel` callbacks. Use AbortSignal. The older Worker API is unchanged.

There is at most **one dispatched request and one latest pending request** per
client. A burst coalesces intermediate pending inputs before posting them, rather
than building an unbounded Worker message queue. Superseded/cancelled promises
reject AbortError immediately. The Worker acknowledges cancellation between
chunks; stale progress, errors and results are not delivered to newer callbacks.
A worker error terminates it; the client can create a fresh worker next time.
Progress includes complete copied snapshots, throttled by interval; final output
is always delivered for an un-superseded successful request. Cancellation is
cooperative and can wait for the current atomic chunk.

Strict newest-only output cannot guarantee a result if a producer continuously
supersedes every input before even one iteration completes. For animation,
separate display FPS from compute Hz: admit one finite fitting slice, keep one
latest pending frame, let the active slice finish, and then admit the pending
frame. Keep a data/request revision and point IDs **outside** the numeric API,
and render an output only against its matching captured point order, or map
retained point IDs explicitly. Do not overlay old labels on a new row layout.
Simulator group count and fitted K are different quantities; this API does not
perform automatic model selection.

## One-shot neighborhood metrics

```js
import { createMetricScheduler } from '../src/index.js';
const metrics = createMetricScheduler({ debounceMs: 80, timeBudgetMs: 4 });
const { result, cacheHit } = await metrics.run(input, { embedding, ks: [3, 5] });
metrics.cancel();
metrics.clearCache();
metrics.dispose();
```

Trustworthiness/continuity are exact **one-shot metrics**, not fitting sessions.
The scheduler snapshots input, debounces a burst, cancels superseded computation
between actual row chunks, and caches at most one completed result. A cache hit
requires identical X, embedding and numerical options by content, not merely a
reused object/key; results are copied. Changed inputs/options invalidate a hit.
This scheduler runs cooperatively on its calling thread; the historical
`createWorkerClient().run('neighborhood', ...)` is available for off-thread work.
As with sessions, soft row-chunk bounds do not guarantee a hard latency limit.

## Reproduce validation and timing

```sh
npm test
node --expose-gc bench/realtime.js > realtime-measurement.json
# Optional earlier source tree, same fixture/init/iteration workload:
node --expose-gc bench/realtime.js --baseline=/absolute/path/to/old/src/index.js
```

The benchmark reports fresh-process first steps, warm one-shot fits, complete
session cost, per-step p50/p95/max (including committed checkpoint copying),
explicit snapshot cost, data/parameter-update cost, and memory observations.
Warm-start fits use a different starting state and are not reported as speedups
against cold fits. It writes only stdout; keep raw samples and host metadata
outside the distributable source tree. See [measured summary](REALTIME_BENCHMARK.md)
and [Python/update parity contract](PYTHON_REALTIME_CONTRACT.md).
