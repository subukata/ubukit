import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X } from './helpers.js';

// Four clusters for three blobs, so that every method takes several iterations.
const converging = {
  kmeans: n => ub.steps.kmeans(X, 4, { seed: 1, maxIter: n }),
  fcm: n => ub.steps.fcm(X, 4, { seed: 1, maxIter: n }),
  efcm: n => ub.steps.efcm(X, 4, { seed: 1, tau: 0.5, maxIter: n }),
  rcm: n => ub.steps.rcm(X, 4, { seed: 1, maxIter: n }),
  rmcm: n => ub.steps.rmcm(X, 4, 0.5, { seed: 1, maxIter: n }),
  somOlp: n => ub.steps.somOlp(X, [3, 3], { lam: 0.5, gamma: 1, maxIter: n }),
};

test('progress results are the run stopped at that iteration', () => {
  // result() at iteration t equals the run with maxIter = t, even when called
  // after the loop has moved on (spreading the generator runs it to the end).
  for (const [name, start] of Object.entries(converging)) {
    const results = [...start(1000)].map(p => p.result());
    assert.ok(results.length > 1, name);
    for (const t of new Set([1, 2, results.length])) assert.deepEqual(results[t - 1], ub.run(start(t)), `${name} at ${t}`);
    assert.deepEqual(results.at(-1), ub.run(start(1000)), name);
  }
  // Scheduled maps: the last epoch is the result; earlier ones are not complete.
  for (const start of [() => ub.steps.som(X, [3, 3], { epochs: 3, seed: 2 }), () => ub.steps.batchSom(X, [3, 3], { epochs: 4 })]) {
    const results = [...start()].map(p => p.result());
    assert.deepEqual(results.at(-1), ub.run(start()));
    assert.ok(results.slice(0, -1).every(r => !r.converged) && results.at(-1).converged);
  }
});

test('changing a progress result leaves the run unchanged', () => {
  // A Result shares no arrays with the run: SOM-OLP's next step reads the
  // memberships, so changing them in place used to change the run.
  for (const [name, start] of Object.entries(converging)) {
    const run = start(1000), first = run.next().value.result();
    first.labels.fill(0);
    first.membership?.data.fill(1 / first.membership.cols);
    assert.deepEqual(ub.run(run), ub.run(start(1000)), name);
  }
});

/** Results equal to 1e-10 in centers, labels, memberships and embeddings. */
function assertClose(a, b, label) {
  for (const field of ['centers', 'labels', 'membership', 'embedding']) {
    const x = a[field], y = b[field];
    if (x === null || y === null) { assert.equal(x, y, `${label} ${field}`); continue; }
    const p = x.data ?? x, q = y.data ?? y;
    assert.equal(p.length, q.length, `${label} ${field}`);
    for (let i = 0; i < p.length; i++) assert.ok(Math.abs(p[i] - q[i]) <= 1e-10 * (1 + Math.abs(q[i])), `${label} ${field}[${i}]: ${p[i]} vs ${q[i]}`);
  }
}

const maps = {
  som: () => ub.steps.som(X, [3, 3], { epochs: 6, seed: 2 }),
  batchSom: () => ub.steps.batchSom(X, [3, 3], { epochs: 6 }),
};
const moved = X.map(([a, b], i) => [a + 2 + 0.3 * Math.sin(i), b - 1 + 0.3 * Math.cos(i)]);

test('sending the same data changes nothing', () => {
  // The data are an input of every iteration; passing them again carries the
  // prototypes and kept state over, so the iterates are the same.
  const starts = { ...Object.fromEntries(Object.entries(converging).map(([k, f]) => [k, () => f(1000)])), ...maps };
  for (const [name, start] of Object.entries(starts)) {
    const plain = [...start()].map(p => p.result());
    const run = start(), fed = [run.next().value.result()];
    for (let t = 1; t < plain.length; t++) fed.push(run.next(X).value.result());
    plain.forEach((r, t) => assertClose(fed[t], r, `${name} at ${t + 1}`));
  }
});

test('new data continue from the current prototypes', () => {
  // These methods carry only their prototypes (maps with constant schedules),
  // so the iteration after new data, here moved and shifted, equals one
  // iteration on them started from where the run had got to.
  const steady = { sigma: 1, sigmaEnd: 1 };
  const cases = {
    kmeans: [(Y, o) => ub.steps.kmeans(Y, 4, { seed: 1, ...o }), { maxIter: 1 }],
    fcm: [(Y, o) => ub.steps.fcm(Y, 4, { seed: 1, ...o }), { maxIter: 1 }],
    efcm: [(Y, o) => ub.steps.efcm(Y, 4, { seed: 1, tau: 0.5, ...o }), { maxIter: 1 }],
    rcm: [(Y, o) => ub.steps.rcm(Y, 4, { seed: 1, ...o }), { maxIter: 1 }],
    rmcm: [(Y, o) => ub.steps.rmcm(Y, 4, 0.5, { seed: 1, ...o }), { maxIter: 1 }],
    som: [(Y, o) => ub.steps.som(Y, [3, 3], { ...steady, lr: 0.1, lrEnd: 0.1, shuffle: false, ...o }), { epochs: 1 }],
    batchSom: [(Y, o) => ub.steps.batchSom(Y, [3, 3], { ...steady, ...o }), { epochs: 1 }],
  };
  for (const [name, [start, one]] of Object.entries(cases)) {
    const run = start(X, {}), prototypes = run.next().value.result().centers;
    const after = run.next(moved).value.result();
    assertClose(after, ub.run(start(moved, { init: prototypes, ...one })), name);
  }
});

test('somOlp keeps memberships for the same points', () => {
  // The latent positions come from the previous memberships while the number
  // of rows stays the same; with another number there are none, as in the
  // first iteration.
  const options = { lam: 0.5, gamma: 1 };
  const run = ub.steps.somOlp(X, [3, 3], options);
  const first = run.next().value.result(), kept = run.next(X).value.result();
  const once = ub.somOlp(X, [3, 3], { ...options, init: first.centers, maxIter: 1 });
  assert.ok(kept.centers.data.some((v, i) => Math.abs(v - once.centers.data[i]) > 1e-6), 'the latent term was used');
  const fewer = X.slice(0, -30);
  const restarted = run.next(fewer).value.result();
  assertClose(restarted, ub.somOlp(fewer, [3, 3], { ...options, init: kept.centers, maxIter: 1 }), 'fewer points');
});

test('new data are checked, and no iteration limit runs to convergence', () => {
  let run = ub.steps.fcm(X, 3, { seed: 0 });
  run.next();
  assert.throws(() => run.next([[1, 2, 3]]), /features/);
  run = ub.steps.fcm(X, 3, { seed: 0 });
  run.next();
  assert.throws(() => run.next(X.map(() => [NaN, 0])), /finite/);
  assertClose(ub.fcm(X, 3, { seed: 0, maxIter: Infinity }), ub.fcm(X, 3, { seed: 0, maxIter: 10_000 }), 'maxIter');
});

test('arguments are checked before the data', () => {
  // The data are invalid too: a wrong argument is reported before any work
  // on the data (centering, seeding, the PCA start).
  const bad = X.map(() => [NaN, 0]);
  for (const [call, name] of [
    [() => ub.kmeans(bad, 3, { maxIter: 0 }), /maxIter/], [() => ub.fcm(bad, 3, { tol: -1 }), /tol/],
    [() => ub.efcm(bad, 3, { maxIter: 0 }), /maxIter/], [() => ub.rcm(bad, 3, { maxIter: 0 }), /maxIter/],
    [() => ub.rmcm(bad, 3, 0.5, { maxIter: 0 }), /maxIter/], [() => ub.som(bad, [3, 3], { epochs: 0 }), /epochs/],
    [() => ub.batchSom(bad, [3, 3], { sigma: 0 }), /sigma/],
    [() => ub.somOlp(bad, [3, 3], { lam: 1, gamma: 1, tol: -1 }), /tol/],
  ]) assert.throws(call, name);
});

test('runAsync reports progress and honors abort', async () => {
  const seen = [];
  const r = await ub.runAsync(ub.steps.fcm(X, 3, { seed: 0 }), { onProgress: p => seen.push(p.iteration) });
  assert.equal(seen.length, r.nIter);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(ub.runAsync(ub.steps.kmeans(X, 3), { signal: controller.signal }), { name: 'AbortError' });
});

test('runAsync gives the event loop its turn between iterations', async () => {
  // With no time budget it yields after every iteration, so a timer set
  // before the run fires while the run is still going, and an abort from
  // that timer stops it.
  let fired = 0, seenAfter = 0;
  setTimeout(() => (fired = 1), 0);
  const r = await ub.runAsync(ub.steps.som(X, [3, 3], { epochs: 5, seed: 0 }), { budgetMs: 0, onProgress: () => (seenAfter += fired) });
  assert.equal(seenAfter, r.nIter - 1, `${seenAfter} of ${r.nIter}`);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 0);
  await assert.rejects(ub.runAsync(ub.steps.som(X, [3, 3], { epochs: 5, seed: 0 }), { budgetMs: 0, signal: controller.signal }), { name: 'AbortError' });
});
