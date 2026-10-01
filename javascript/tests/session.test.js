import test from 'node:test';
import assert from 'node:assert/strict';
import { createSession, run, sessionAlgorithms } from '../src/index.js';
const X = { data: Float64Array.of(0, 0, .1, -.1, .2, .1, 4, 4, 4.3, 4.1, 3.8, 4.2), nSamples: 6, nFeatures: 2 };
const centers = Float64Array.of(0, 0, 4, 4);
function opts(a) { return { nClusters: 2, initCenters: centers, maxIterations: 7, blockRows: 2, tolerance: 0,
  ...(a === 'rmcm' ? { delta: .7, graphBatchPairs: 2 } : {}),
  ...(a === 'som-olp' ? { grid: { data: Float64Array.of(-1, 1), nSamples: 2, nFeatures: 1 }, lambda: .4 } : {}) }; }
function finish(s, count = 1, chunks = 1) {
  let steps = 0;
  while (!s.status.done) { s.step(count, { timeBudgetMs: 1e6, maxChunks: chunks }); if (++steps > 10000) throw new Error('failed to terminate'); }
  return s.snapshot().result;
}
for (const a of sessionAlgorithms) {
  test(`${a}: segmented and uninterrupted historical result exactly agree`, () => {
    const o = opts(a), expected = run(a, X, o), s = createSession(a, X, o);
    assert.equal(s.snapshot().result, null);
    assert.deepEqual(finish(s), expected);
    assert.equal(s.status.iteration, expected.iterations);
    s.dispose();
  });
  test(`${a}: inputs and snapshots are owned`, () => {
    const x = { ...X, data: X.data.slice() }, o = opts(a), s = createSession(a, x, o);
    x.data.fill(999); o.initCenters.fill(42); // restore shared fixture below
    const result = finish(s), again = s.snapshot().result;
    assert.notEqual(result.centers[0], 999); result.centers.fill(-100);
    assert.deepEqual(s.snapshot().result.centers, again.centers);
    if (result.membership) { result.membership.fill(-3); assert.ok(s.snapshot().result.membership.every(v => v >= 0)); }
    centers.set([0, 0, 4, 4]);
  });
  test(`${a}: invalid updates are transactional`, () => {
    const s = createSession(a, X, opts(a)); finish(s);
    const before = s.snapshot();
    for (const patch of [{ maxIterations: -1 }, { blockRows: 0 }, { tolerance: -1 }, { seed: NaN }, { maxMemoryBytes: 1 }, { initCenters: Float64Array.of(1) }]) {
      assert.throws(() => s.updateParameters(patch)); assert.deepEqual(s.snapshot(), before);
    }
    assert.throws(() => s.updateData({ ...X, data: Float64Array.of(NaN) }));
    assert.deepEqual(s.snapshot(), before);
  });
  test(`${a}: moved/resized data uses centers and clears stale memberships`, () => {
    const s = createSession(a, X, opts(a)); finish(s);
    const moved = { ...X, data: Float64Array.from(X.data, v => v + .1) };
    const oldRevision = s.status.revision;
    s.updateData(moved); assert.equal(s.status.revision, oldRevision + 1); assert.equal(s.status.invalidation.kind, 'centers');
    assert.equal(s.snapshot().result, null); assert.equal(s.status.iteration, 0);
    // A second update before a complete iteration still retains compatible centers.
    s.step(1, { maxChunks: 1 });
    s.updateData({ data: Float64Array.of(0, 0, 3, 3, 3.1, 3.1), nSamples: 3, nFeatures: 2 });
    assert.equal(s.status.invalidation.kind, 'centers');
    const result = finish(s); assert.equal(result.nSamples, 3); assert.equal(result.labels.length, 3);
    if (result.membership) assert.equal(result.membership.length, 6);
  });
  test(`${a}: reset and disposal lifecycle`, () => {
    const s = createSession(a, X, opts(a)); const first = finish(s);
    const rev = s.status.revision; s.reset(); assert.equal(s.status.revision, rev + 1);
    assert.deepEqual(finish(s), first);
    s.dispose(); s.dispose(); assert.equal(s.status.disposed, true);
    for (const fn of [() => s.step(), () => s.snapshot(), () => s.reset(), () => s.updateData(X), () => s.updateParameters({})]) assert.throws(fn, /disposed/);
  });
}
test('partial iterations remain private and step budget counts actual chunks', () => {
  const s = createSession('fcm', X, opts('fcm'));
  let r = s.step(10, { timeBudgetMs: 0, maxChunks: 100 });
  assert.equal(r.chunks, 1); assert.equal(r.completedIterations, 0); assert.equal(s.snapshot().result, null);
  while (!s.status.iteration) s.step(1, { maxChunks: 1 });
  const committed = s.snapshot(); s.step(10, { maxChunks: 1 });
  assert.deepEqual(s.snapshot().result, committed.result);
  const zero = s.step(0); assert.equal(zero.chunks, 0);
});
test('FCM changing m preserves U as warm state and resets objective history', () => {
  const s = createSession('fcm', X, { ...opts('fcm'), m: 1.7, returnHistory: true });
  const old = finish(s); s.updateParameters({ m: 2.5, maxIterations: 1 });
  assert.equal(s.status.invalidation.kind, 'membership-and-centers'); assert.equal(s.snapshot().result, null);
  const now = finish(s), expected = run('fcm', X, { nClusters: 2, initMembership: old.membership, m: 2.5, tolerance: 0, maxIterations: 1, returnHistory: true });
  assert.deepEqual(now.membership, expected.membership); assert.deepEqual(now.centers, expected.centers);
  assert.equal(now.objectiveHistory.length, 1); assert.equal(now.m, 2.5);
});
for (const m of [1 + Number.EPSILON, 1.01, 2, 2.7, 100, 1e300]) test(`FCM m=${m} remains exact across chunks`, () => {
  const o = { ...opts('fcm'), m, maxIterations: 2, initCenters: Float64Array.of(0, 0, 0, 0) };
  assert.deepEqual(finish(createSession('fcm', X, o)), run('fcm', X, o));
});
for (const p of [Number.MIN_VALUE, .01, 1, 2, 80, 1e300]) test(`ExRCM p=${p} remains exact across chunks`, () => {
  const o = { ...opts('exrcm'), p, beta: .1, maxIterations: 4 };
  assert.deepEqual(finish(createSession('exrcm', X, o)), run('exrcm', X, o));
});
test('invalid m/p rejected transactionally, zero beta/delta accepted', () => {
  for (const [a, key, invalid] of [['fcm', 'm', [0, 1, Infinity, NaN]], ['exrcm', 'p', [0, -1, Infinity, NaN]], ['rcm', 'p', [2]]]) {
    const s = createSession(a, X, opts(a));
    const before = s.snapshot();
    for (const v of invalid) { assert.throws(() => s.updateParameters({ [key]: v })); assert.deepEqual(s.snapshot(), before); }
  }
  assert.ok(finish(createSession('rmcm', X, { ...opts('rmcm'), delta: 0 })));
});
test('changing dimensions and cluster count invalidates compatible state safely', () => {
  const s = createSession('fcm', X, opts('fcm')); finish(s);
  s.updateParameters({ nClusters: 3 }); assert.equal(s.status.invalidation.kind, 'cold'); assert.equal(finish(s).centers.length, 6);
  s.updateData({ data: Float64Array.of(0, 1, 5, 6), nSamples: 4, nFeatures: 1 });
  assert.equal(s.status.invalidation.kind, 'cold'); assert.equal(finish(s).centers.length, 3);
});
test('RMCM graph retained only for unchanged data/delta/backend and reset', () => {
  const s = createSession('rmcm', X, opts('rmcm')); finish(s); assert.equal(s.status.preparedGraph, true);
  s.updateParameters({ maxIterations: 2 }); assert.equal(s.status.invalidation.preparedGraph, 'retained'); finish(s);
  const rev = s.status.revision; s.updateData({ ...X, data: X.data.slice() }); assert.equal(s.status.revision, rev);
  s.reset(); assert.equal(s.status.preparedGraph, true); finish(s);
  s.updateParameters({ delta: .1 }); assert.equal(s.status.preparedGraph, false); finish(s);
  s.updateData({ ...X, data: Float64Array.from(X.data, x => x + 1) }); assert.equal(s.status.preparedGraph, false); finish(s);
  s.updateParameters({ backend: 'reference' }); assert.equal(s.status.preparedGraph, false); assert.match(finish(s).backend, /reference/);
});
test('SOM parameter update retains P/W, history resets; grid invalidates', () => {
  const s = createSession('som-olp', X, opts('som-olp')); finish(s);
  s.updateParameters({ lambda: .2 }); assert.equal(s.status.invalidation.kind, 'membership-and-centers'); finish(s);
  s.updateParameters({ grid: { data: Float64Array.of(-1, 0, 1), nSamples: 3, nFeatures: 1 }, initializer: 'sample' });
  assert.equal(s.status.invalidation.kind, 'cold'); assert.equal(finish(s).nUnits, 3);
});
test('cancelled session can reset, callback reentry cannot race state', () => {
  const abort = new AbortController(), s = createSession('fcm', X, { ...opts('fcm'), signal: abort.signal });
  abort.abort(); assert.throws(() => s.step(), { name: 'AbortError' }); assert.equal(s.status.phase, 'cancelled');
  s.updateParameters({ signal: undefined }); assert.ok(finish(s));
  let guarded = false; let t;
  t = createSession('fcm', X, { ...opts('fcm'), onProgress: () => { assert.throws(() => t.updateParameters({ m: 3 }), /during a step/); guarded = true; } });
  finish(t); assert.ok(guarded);
});
test('neighborhood metrics are not iterative fitting sessions', () => assert.throws(() => createSession('neighborhood', X, {}), /one-shot metric/));
test('configured row membership is invalidated even before the first iteration', () => {
  const u = Float64Array.of(1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1);
  const o = { nClusters: 2, initMembership: u, maxIterations: 1, seed: 3, tolerance: 0 };
  const s = createSession('fcm', X, o), moved = { ...X, data: Float64Array.from(X.data).reverse() };
  s.updateData(moved);
  assert.deepEqual(finish(s), run('fcm', moved, { nClusters: 2, maxIterations: 1, seed: 3, tolerance: 0 }));
});
test('memory-impossible cycle/scratch changes fail before mutating state', () => {
  const s = createSession('exrcm', X, opts('exrcm')); finish(s); const before = s.snapshot();
  assert.throws(() => s.updateParameters({ cycleWindow: Number.MAX_SAFE_INTEGER })); assert.deepEqual(s.snapshot(), before);
  const t = createSession('som-olp', X, opts('som-olp')); finish(t); const prior = t.snapshot();
  assert.throws(() => t.updateParameters({ maxScratchBytes: 1 })); assert.deepEqual(t.snapshot(), prior);
});
test('atomic full configure updates simultaneous data/K and rejects bad replacement', () => {
  const s = createSession('fcm', X, opts('fcm')); finish(s);
  const data = { data: Float64Array.of(1, 2, 3), nSamples: 3, nFeatures: 1 };
  s.configure(data, { nClusters: 3, maxIterations: 2 }); assert.equal(finish(s).membership.length, 9);
  const before = s.snapshot(); assert.throws(() => s.configure(data, { nClusters: 0 })); assert.deepEqual(s.snapshot(), before);
});
test('RMCM session preserves exact cycle state across chunks, not false convergence', () => {
  const x = { data: Float64Array.of(-1,-3,5,-3,3,4,-1,-2,1,-3,2,2), nSamples: 6, nFeatures: 2 };
  for (const backend of ['adjoint', 'reference']) {
    const o = { delta: 7.3, initCenters: Float64Array.of(1,-3,2,2), blockRows: 1, graphBatchPairs: 1, backend };
    const actual = finish(createSession('rmcm', x, o));
    assert.deepEqual(actual, run('rmcm', x, o)); assert.equal(actual.stopReason, 'cycle'); assert.equal(actual.cycleLength, 2); assert.equal(actual.converged, false);
  }
});
