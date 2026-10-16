import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X } from './helpers.js';

const C = [[1, 1], [4, 1], [1, 4]];
const sq = (a, b) => a.reduce((s, v, f) => s + (v - b[f]) ** 2, 0);

test('a clustering applied assigns as its training does', () => {
  // The first iteration of training from the same centers assigns the same
  // memberships and labels with the same objective; applying never moves the centers.
  for (const [name, args, options] of [
    ['kmeans', [], {}], ['fcm', [], { m: 1.5 }], ['efcm', [], { tau: 0.7 }],
    ['rcm', [], { alpha: 1.3, beta: 0.2 }], ['rmcm', [0.8], {}],
  ]) {
    const applied = ub.apply[name](X, C, ...args, options), first = ub[name](X, 3, ...args, { init: C, maxIter: 1, ...options });
    assert.deepEqual(applied.labels, first.labels, name);
    assert.deepEqual(applied.membership, first.membership, name);
    applied.history.forEach((h, t) => assert.ok(Math.abs(h - first.history[t]) <= 1e-12 * Math.abs(h), name));
    assert.deepEqual(ub.toRows(applied.centers), C, name);
    assert.deepEqual([applied.nIter, applied.stopReason, applied.embedding, applied.unitCoordinates], [1, 'schedule', null, null], name);
  }
});

test('the centers stay exactly as given', () => {
  // Centered on data at 1e10 and restored, the center 0.1 would come back as
  // 0.10000038146972656, and new data passed to the run would add up such changes.
  const C2 = [[0.1], [5]];
  assert.deepEqual(ub.toRows(ub.apply.fcm([[1e10]], C2).centers), C2);
  const run = ub.steps.apply.somOlp([[1e10]], C2, [[0], [1]], { lam: 1, gamma: 1, maxIter: Infinity });
  const results = [run.next().value.result()];
  for (let t = 2; t < 5; t++) results.push(run.next([[t * 1e10]]).value.result());
  for (const r of results) assert.deepEqual(ub.toRows(r.centers), C2);
});

test('a map applied gives the best-matching units', () => {
  const m = ub.batchSom(X, [3, 4], { epochs: 5 }), W = ub.toRows(m.centers), shifted = X.map(r => r.map(v => v + 0.3));
  for (const fit of [ub.apply.som, ub.apply.batchSom]) {
    const r = fit(shifted, m.centers, m.unitCoordinates), units = ub.toRows(m.unitCoordinates);
    assert.deepEqual(Array.from(r.labels), shifted.map(x => W.reduce((b, w, j) => (sq(x, w) < sq(x, W[b]) ? j : b), 0)));
    assert.deepEqual(ub.toRows(r.embedding), Array.from(r.labels, j => units[j]));
    assert.deepEqual(r.unitCoordinates, m.unitCoordinates);
    assert.deepEqual([r.membership, r.nIter, r.stopReason], [null, 1, 'schedule']);
    assert.deepEqual(fit(X, m.centers, [3, 4]).labels, m.labels); // the grid as [rows, cols]
  }
});

test('somOlp applied iterates the memberships with the prototypes fixed', () => {
  const m = ub.somOlp(X, [3, 4], { lam: 0.5, gamma: 1 }), W = ub.toRows(m.centers), R = ub.toRows(m.unitCoordinates);
  const points = X.filter((_, i) => i % 7 === 0).map(r => r.map(v => v + 0.4));
  // Reference: the memberships from the distances alone, then with the latent positions.
  const softmin = cost => {
    const low = Math.min(...cost), e = cost.map(c => Math.exp(-(c - low) / 0.5)), s = e.reduce((a, b) => a + b);
    return e.map(v => v / s);
  };
  let P = points.map(x => softmin(W.map(w => sq(x, w)))), n = 1;
  for (let t = 2; t <= 1000; t++) {
    const next = points.map((x, i) => {
      const v = [0, 1].map(c => P[i].reduce((s, p, j) => s + p * R[j][c], 0));
      return softmin(W.map((w, j) => sq(x, w) + sq(v, R[j])));
    });
    const change = Math.max(...next.flatMap((row, i) => row.map((p, j) => Math.abs(p - P[i][j]))));
    P = next, n = t;
    if (change <= 1e-6) break;
  }
  const r = ub.apply.somOlp(points, m.centers, m.unitCoordinates, { lam: 0.5, gamma: 1 });
  assert.deepEqual([r.nIter, r.stopReason], [n, 'tol']);
  assert.ok(n > 2);
  ub.toRows(r.membership).forEach((row, i) => row.forEach((p, j) => assert.ok(Math.abs(p - P[i][j]) < 1e-9)));
  ub.toRows(r.embedding).forEach((v, i) => v.forEach((e, c) => {
    assert.ok(Math.abs(e - P[i].reduce((s, p, j) => s + p * R[j][c], 0)) < 1e-9);
  }));
  assert.deepEqual(ub.toRows(r.centers), W);
  for (let t = 2; t < r.history.length; t++) assert.ok(r.history[t] <= r.history[t - 1] + 1e-9 * Math.abs(r.history[t - 1]));
  const once = ub.apply.somOlp(points, m.centers, m.unitCoordinates, { lam: 0.5, gamma: 1, maxIter: 1 });
  assert.equal(once.stopReason, 'limit');
  ub.toRows(once.membership).forEach((row, i) => {
    const expected = softmin(W.map(w => sq(points[i], w)));
    row.forEach((p, j) => assert.ok(Math.abs(p - expected[j]) < 1e-12));
  });
});

test('somOlp applied places each point on its own, and takes the same points moved', () => {
  // Each point's update depends on it alone; with points away from the midpoints
  // between units and a tight tolerance, applying to part of the points gives
  // their rows of applying to all.
  const m = ub.somOlp(X, [3, 4], { lam: 0.5, gamma: 1 }), options = { lam: 0.5, gamma: 1, tol: 1e-12 };
  const whole = ub.apply.somOlp(X, m.centers, m.unitCoordinates, options);
  const part = ub.apply.somOlp(X.slice(0, 40), m.centers, m.unitCoordinates, options);
  part.membership.data.forEach((p, i) => assert.ok(Math.abs(p - whole.membership.data[i]) < 1e-9));
  // After the memberships settle, the same points moved continue the run.
  const run = ub.steps.apply.somOlp(X, m.centers, [3, 4], { lam: 0.5, gamma: 1, maxIter: Infinity });
  let p = run.next().value;
  while (p.result().stopReason === null) p = run.next().value;
  assert.equal(run.next(X.map(r => r.map(v => v + 0.2))).value.result().nIter, p.iteration + 1);
});

test('rmcm applied depends on the points given with it', () => {
  // The neighborhoods are those within the data given: the point at 4 is all in
  // the first cluster beside another at 4, half in each beside 6, and given
  // alone it gets the hard assignment.
  const centers = [[0], [10]];
  assert.deepEqual(Array.from(ub.apply.rmcm([[4], [4]], centers, 3).membership.data.slice(0, 2)), [1, 0]);
  assert.deepEqual(Array.from(ub.apply.rmcm([[4], [6]], centers, 3).membership.data.slice(0, 2)), [0.5, 0.5]);
  assert.deepEqual(Array.from(ub.apply.rmcm([[4]], centers, 3).membership.data), [1, 0]);
});

test('fewer points than centers, and one assignment ends the run', () => {
  const r = ub.apply.fcm([[2, 1]], C);
  assert.deepEqual([r.membership.rows, r.membership.cols, r.labels[0]], [1, 3, 0]);
  const run = ub.steps.apply.fcm(X, C);
  run.next();
  const end = run.next(X); // one assignment: new rows need a new application
  assert.ok(end.done && end.value.stopReason === 'schedule');
});

test('apply checks its arguments', () => {
  assert.throws(() => ub.apply.fcm(X, [[1, 1, 1], [2, 2, 2]]), /columns/);
  assert.throws(() => ub.apply.som(X, Array(5).fill([0, 0]), [2, 2]), /4 rows/);
  assert.throws(() => ub.apply.fcm(X, C, { m: 1 }), RangeError);
  assert.throws(() => ub.apply.somOlp(X, Array(4).fill([0, 0]), [2, 2], { lam: 0, gamma: 1 }), /lam/);
  assert.throws(() => ub.apply.fcm(X, C, { maxIter: 3 }), { name: 'TypeError', message: 'unknown option maxIter' });
  assert.throws(() => ub.apply.kmeans(X, C, { maxIter: 3 }), { name: 'TypeError', message: 'unknown option maxIter' });
  assert.throws(() => ub.apply.batchSom(X, Array(4).fill([0, 0]), [2, 2], { epochs: 3 }), TypeError);
  assert.throws(() => ub.steps.apply.som(X, Array(4).fill([0, 0]), [2, 2], { epochs: 3 }).next(), TypeError);
  // Arguments are checked before the data, which are invalid too.
  assert.throws(() => ub.apply.fcm([[NaN, 0]], [[NaN, 0]], { m: 1 }), /m must be/);
  assert.throws(() => ub.apply.somOlp([[NaN, 0]], [[NaN, 0]], [[0]], { lam: 0, gamma: 1 }), /lam must be/);
  assert.throws(() => ub.apply.somOlp(X, Array(4).fill([0, 0]), [2, 2], { lam: 1, gamma: 1, epochs: 3 }), TypeError);
});
