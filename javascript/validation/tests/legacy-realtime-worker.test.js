import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Worker } from 'node:worker_threads';
import { createRealtimeWorkerClient, createMetricScheduler, run } from '../../consumer/node_modules/ubukit-js/src/index.js';
const X = { data: Float64Array.of(0, .1, .2, 4, 4.1, 4.2), nSamples: 6, nFeatures: 1 };
const options = { nClusters: 2, initCenters: Float64Array.of(0, 4), maxIterations: 3, tolerance: 0, blockRows: 2 };
const factory = () => new Worker(new URL('../../consumer/node_modules/ubukit-js/src/realtime-worker.js', import.meta.url), { type: 'module' });
test('realtime Worker exact output, progressive copied snapshots, persistent warm reuse', async () => {
  let creations = 0; const c = createRealtimeWorkerClient({ workerFactory: () => { ++creations; return factory(); } });
  try {
    const progress = [];
    const r = await c.run('fcm', X, { ...options, onProgress: s => { progress.push(s); if (s.result) s.result.centers.fill(-999); } }, { maxChunks: 1, timeBudgetMs: 0, progressIntervalMs: 0 });
    assert.equal(r.requestId, 1); assert.deepEqual(r.result, run('fcm', X, options)); assert.ok(progress.length > 0);
    const r2 = await c.run('fcm', X, { ...options, m: 3, maxIterations: 1 }, { maxChunks: 2 });
    assert.equal(r2.requestId, 2); assert.equal(r2.invalidation.kind, 'membership-and-centers'); assert.equal(r2.result.m, 3); assert.equal(creations, 1);
  } finally { c.dispose(); }
});
test('latest input wins with bounded one-in-flight/one-pending backpressure; stale suppression', async () => {
  class Fake extends EventEmitter { sent = []; postMessage(m) { this.sent.push(m); } terminate() {} }
  const worker = new Fake(), c = createRealtimeWorkerClient({ workerFactory: () => worker });
  const forwarded = []; const ps = [];
  for (let i = 0; i < 20; ++i) ps.push(c.run('fcm', X, { ...options, m: 2 + i / 10, onProgress: e => forwarded.push(e) }).catch(e => e));
  assert.equal(worker.sent.filter(m => m.type === 'run').length, 1);
  assert.equal(worker.sent.filter(m => m.type === 'cancel').length, 1);
  worker.emit('message', { id: 1, type: 'progress', snapshot: { stale: true } }); assert.deepEqual(forwarded, []);
  worker.emit('message', { id: 1, type: 'result', snapshot: { stale: true } });
  assert.equal(worker.sent.filter(m => m.type === 'run').length, 2);
  assert.equal(worker.sent.at(-1).id, 20);
  worker.emit('message', { id: 1, type: 'progress', snapshot: { stale: true } });
  worker.emit('message', { id: 20, type: 'result', snapshot: { requestId: 20, result: 'latest' } });
  const results = await Promise.all(ps);
  assert.ok(results.slice(0, -1).every(e => e.name === 'AbortError'));
  assert.equal(results.at(-1).result, 'latest'); assert.equal(c.busy, false); c.dispose();
});
test('real Worker cancellation after progress, immediate replacement, data resize, reuse', async () => {
  const c = createRealtimeWorkerClient({ workerFactory: factory });
  try {
    let replace, newJob; const gate = new Promise(resolve => { replace = resolve; });
    const old = c.run('fcm', X, { ...options, maxIterations: 100, onProgress: () => {
      if (!newJob) { newJob = c.run('fcm', { data: Float64Array.of(1, 2, 8, 9), nSamples: 4, nFeatures: 1 }, options); replace(); }
    } }, { maxChunks: 1, progressIntervalMs: 0 }).catch(e => e);
    await gate; assert.equal((await old).name, 'AbortError');
    const r = await newJob; assert.equal(r.result.nSamples, 4); assert.equal(r.invalidation.kind, 'centers'); assert.equal(r.requestId, 2);
    const bad = await c.run('fcm', X, { ...options, m: 1 }).catch(e => e); assert.equal(bad.name, 'RangeError');
    assert.ok((await c.run('kmeans', X, options)).done);
  } finally { c.dispose(); }
});
test('abort, disposed, and worker error do not leave busy clients', async () => {
  const c = createRealtimeWorkerClient({ workerFactory: factory });
  const ac = new AbortController(); ac.abort(); await assert.rejects(c.run('fcm', X, { ...options, signal: ac.signal }), { name: 'AbortError' });
  const promise = c.run('fcm', X, options); const rejected = assert.rejects(promise, { name: 'AbortError' }); c.cancel(); await rejected;
  assert.ok((await c.run('fcm', X, options)).done);
  c.dispose(); await assert.rejects(c.run('fcm', X, options), /disposed/);
});
test('RMCM Worker retains graph for parameters, invalidates changed input', async () => {
  const c = createRealtimeWorkerClient({ workerFactory: factory });
  try {
    const o = { ...options, delta: .3 };
    const a = await c.run('rmcm', X, o); assert.ok(a.preparedGraph);
    const b = await c.run('rmcm', X, { ...o, maxIterations: 2 }); assert.equal(b.invalidation.preparedGraph, 'retained');
    const d = await c.run('rmcm', { ...X, data: Float64Array.from(X.data, x => x + .1) }, o); assert.equal(d.invalidation.preparedGraph, 'invalidated');
    assert.equal(d.result.membership.length, 12);
  } finally { c.dispose(); }
});
test('one-shot metrics debounce latest, content cache is copied and invalidated', async () => {
  const scheduler = createMetricScheduler({ debounceMs: 5, timeBudgetMs: 0, maxChunks: 1 });
  const o = { embedding: { ...X, data: X.data.slice() }, k: 1, blockRows: 1 };
  try {
    const a = scheduler.run(X, o).catch(e => e), b = scheduler.run(X, o);
    assert.equal((await a).name, 'AbortError'); const r = await b; assert.equal(r.cacheHit, false); assert.deepEqual(r.result, run('neighborhood', X, o));
    r.result.qualities[0].trustworthiness = -999;
    const hit = await scheduler.run(X, o); assert.equal(hit.cacheHit, true); assert.notEqual(hit.result.qualities[0].trustworthiness, -999);
    const next = { ...o, embedding: { ...X, data: Float64Array.from(X.data).reverse() } };
    assert.equal((await scheduler.run(X, next)).cacheHit, false);
    scheduler.clearCache(); assert.equal(scheduler.status.cached, false);
  } finally { scheduler.dispose(); }
});
test('metrics cancel during actual progress and reuse; memory guard', async () => {
  const scheduler = createMetricScheduler({ debounceMs: 0, maxChunks: 1, timeBudgetMs: 0 });
  try {
    await assert.rejects(scheduler.run(X, { embedding: X, k: 1, blockRows: 1, onProgress: () => scheduler.cancel() }), { name: 'AbortError' });
    assert.ok((await scheduler.run(X, { embedding: X, k: 1 })).result);
  } finally { scheduler.dispose(); }
  const tiny = createMetricScheduler({ maxMemoryBytes: 1 }); await assert.rejects(tiny.run(X, { embedding: X, k: 1 }), /reserve/); tiny.dispose();
});
for (const action of ['dispose', 'cancel', 'replace']) test(`progress callback ${action}+throw cannot corrupt active request`, async () => {
  const client = createRealtimeWorkerClient({ workerFactory: factory });
  let triggered = false, newer = null;
  try {
    const old = client.run('fcm', X, { ...options, maxIterations: 100, onProgress: () => {
      if (triggered) return; triggered = true;
      if (action === 'dispose') client.dispose();
      else if (action === 'cancel') client.cancel();
      else newer = client.run('fcm', X, { ...options, m: 3 });
      throw new Error('intentional callback failure after lifecycle action');
    } }, { maxChunks: 1, progressIntervalMs: 0 }).catch(e => e);
    assert.equal((await old).name, 'AbortError'); assert.ok(triggered);
    if (newer) assert.equal((await newer).result.m, 3);
    else if (action === 'cancel') assert.ok((await client.run('fcm', X, options)).done);
  } finally { client.dispose(); }
});
