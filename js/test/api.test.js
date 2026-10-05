import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';

const rand = (() => { let s = 7; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); })();
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const centers = [[0, 0], [5, 0], [0, 5]];
const X = centers.flatMap(c => Array.from({ length: 80 }, () => [c[0] + 0.5 * gauss(), c[1] + 0.5 * gauss()]));

function nearestError(found) {
  const rows = ub.toRows(found);
  return Math.max(...centers.map(c => Math.min(...rows.map(r => Math.hypot(r[0] - c[0], r[1] - c[1])))));
}

for (const [name, fit] of [
  ['kmeans', () => ub.kmeans(X, 3, { seed: 0 })],
  ['fcm', () => ub.fcm(X, 3, { seed: 0 })],
  ['fcm m=1.000001', () => ub.fcm(X, 3, { m: 1.000001, seed: 0 })],
  ['efcm', () => ub.efcm(X, 3, { seed: 0 })],
  ['rcm', () => ub.rcm(X, 3, { seed: 0 })],
  ['exrcm', () => ub.rcm(X, 3, { p: 2, beta: 0.3, seed: 0 })],
  ['rmcm', () => ub.rmcm(X, 3, 0.5, { seed: 0 })],
]) {
  test(`${name} recovers the blobs`, () => {
    const r = fit();
    assert.ok(r.converged);
    assert.ok(nearestError(r.centers) < 0.2);
    if (r.membership) {
      for (const row of ub.toRows(r.membership)) assert.ok(Math.abs(row.reduce((a, b) => a + b) - 1) < 1e-12);
    }
  });
}

test('kmeans iterates are lloyds', () => {
  // Hamerly's bounds only skip work: compare with a plain Lloyd loop.
  for (const k of [1, 2, 7, 40]) {
    const init = X.filter((_, i) => i % Math.floor(X.length / k) === 0).slice(0, k).map(r => r.map(v => v + 0.01));
    let V = init.map(r => r.slice()), labels = [], history = [];
    for (let t = 0; t < 300; t++) {
      labels = X.map(x => V.reduce((b, v, j) => (x.reduce((s, xf, f) => s + (xf - v[f]) ** 2, 0) < x.reduce((s, xf, f) => s + (xf - V[b][f]) ** 2, 0) ? j : b), 0));
      history.push(X.reduce((s, x, i) => s + x.reduce((a, xf, f) => a + (xf - V[labels[i]][f]) ** 2, 0), 0));
      const next = V.map((v, c) => {
        const members = X.filter((_, i) => labels[i] === c);
        return members.length ? v.map((_, f) => members.reduce((s, m) => s + m[f], 0) / members.length) : v;
      });
      if (next.every((v, c) => v.every((a, f) => a === V[c][f]))) break;
      V = next;
    }
    const r = ub.kmeans(X, k, { init });
    assert.deepEqual(Array.from(r.labels), labels);
    assert.equal(r.nIter, history.length);
    ub.toRows(r.centers).forEach((row, c) => row.forEach((a, f) => assert.ok(Math.abs(a - V[c][f]) < 1e-9)));
    r.history.forEach((h, t) => assert.ok(Math.abs(h - history[t]) <= 1e-9 * history[t]));
  }
});

test('default seeding rarely merges separated blobs', () => {
  // Greedy k-means++ keeps the best of 2 + ln k draws per seed (18 of 20 seeds
  // recover these blobs exactly; single draws recovered 4).
  let s = 11;
  const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const g = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const blobCenters = Array.from({ length: 10 }, () => Array.from({ length: 16 }, () => r() * 10 - 5));
  const y = Array.from({ length: 2000 }, (_, i) => i % 10);
  const data = y.map(c => blobCenters[c].map(v => v + g()));
  let perfect = 0;
  for (let seed = 0; seed < 20; seed++) perfect += ub.ari(y, ub.kmeans(data, 10, { seed }).labels) === 1;
  assert.ok(perfect >= 13, `${perfect} of 20`);
});

test('fcm with a large fuzzifier keeps moving', () => {
  const m = 1000, r = ub.fcm(X, 3, { m, seed: 0, maxIter: 1000 });
  assert.ok(r.converged && r.nIter > 1);
  // Centers are the u^m-weighted means; u^m underflows, so weigh with logarithms.
  const U = ub.toRows(r.membership), C = ub.toRows(r.centers);
  for (let c = 0; c < 3; c++) {
    const L = U.map(u => m * Math.log(u[c])), top = Math.max(...L), w = L.map(l => Math.exp(l - top));
    const total = w.reduce((a, b) => a + b);
    for (let f = 0; f < 2; f++) {
      const mean = X.reduce((s, x, i) => s + w[i] * x[f], 0) / total;
      assert.ok(Math.abs(mean - C[c][f]) < 1e-9, `${mean} vs ${C[c][f]}`);
    }
  }
});

test('objective never increases', () => {
  for (const r of [ub.kmeans(X, 4, { seed: 1 }), ub.fcm(X, 4, { m: 1.7, seed: 1 }), ub.efcm(X, 4, { tau: 0.5, seed: 1 })]) {
    for (let t = 1; t < r.history.length; t++) assert.ok(r.history[t] <= r.history[t - 1] * (1 + 1e-12));
  }
});

test('maps produce grid embeddings', () => {
  const b = ub.batchSom(X, [4, 4], { epochs: 10 });
  assert.equal(b.centers.rows, 16);
  assert.equal(b.embedding.rows, X.length);
  const s = ub.som(X, [3, 3], { epochs: 2, seed: 1 });
  assert.deepEqual(s.centers.data, ub.som(X, [3, 3], { epochs: 2, seed: 1 }).centers.data);
  const o = ub.somOlp(X, [3, 3], { lam: 0.5, gamma: 1 });
  assert.equal(o.embedding.cols, 2);
  assert.equal(ub.somOlp(X, undefined, { lam: 0.5, gamma: 1, maxIter: 2 }).centers.rows, 100);
});

test('pca orientation does not depend on rounding', () => {
  // Standardized 2-D data have axes (1, +-1)/sqrt(2) whose components tie in
  // magnitude; reordering the rows changes only the rounding.
  const fit = A => ub.somOlp(A, [3, 4], { lam: 1, gamma: 0, maxIter: 1 }).membership.data;
  for (let t = 0; t < 20; t++) {
    const Z = Array.from({ length: 200 }, () => { const a = gauss(), b = gauss(); return [a, 0.6 * a + 0.8 * b]; });
    const S = Z.map(z => z.slice());
    for (let f = 0; f < 2; f++) {
      const mean = Z.reduce((s, z) => s + z[f], 0) / Z.length;
      const sd = Math.sqrt(Z.reduce((s, z) => s + (z[f] - mean) ** 2, 0) / Z.length);
      S.forEach(z => (z[f] = (z[f] - mean) / sd));
    }
    const order = S.map((_, i) => (i * 37) % S.length);
    const a = fit(S), b = fit(order.map(i => S[i]));
    order.forEach((i, r) => { for (let j = 0; j < 12; j++) assert.ok(Math.abs(a[i * 12 + j] - b[r * 12 + j]) < 1e-9); });
  }
});

test('matrix input accepts {data, rows, cols} and validates', () => {
  const m = ub.matrix(X);
  assert.deepEqual(ub.kmeans(m, 3, { seed: 2 }).centers, ub.kmeans(X, 3, { seed: 2 }).centers);
  assert.throws(() => ub.kmeans([[1, NaN]], 1), RangeError);
  assert.throws(() => ub.kmeans([[1, 2], [3]], 1), RangeError);
  assert.throws(() => ub.matrix({ data: new Float64Array(5), rows: 2.5, cols: 2 }), RangeError);
  assert.throws(() => ub.kmeans(X.map(row => row.map(v => v * 1e200)), 3), /scale/);
  assert.throws(() => ub.kmeans(X.map(row => row.map(v => v * 1e-170)), 3), /scale/);
  assert.deepEqual(Array.from(ub.fcm([[1, 1], [1, 1], [1, 1]], 2, { seed: 0 }).centers.data), [1, 1, 1, 1]);
  assert.throws(() => ub.rmcm(X, 3, 0, { maxEdges: 10 }), /maxEdges/);
  assert.throws(() => ub.somOlp(X, [3, 3], { lam: 1, gamma: 1, pcaScale: 0 }), RangeError);
  assert.throws(() => ub.kmeans(X, 0), RangeError);
  assert.throws(() => ub.fcm(X, 3, { m: 1 }), RangeError);
  assert.throws(() => ub.rmcm(X, 3, 100, { maxEdges: 1000 }), /maxEdges/);
  assert.throws(() => ub.kmeans(X, 3, { init: 'random' }), RangeError);
  assert.throws(() => ub.somOlp(X, [3, 3], { lam: 0, gamma: 1 }), RangeError);
});

test('runAsync reports progress and honors abort', async () => {
  const seen = [];
  const r = await ub.runAsync(ub.steps.fcm(X, 3, { seed: 0 }), { onProgress: p => seen.push(p.iteration) });
  assert.equal(seen.length, r.nIter);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(ub.runAsync(ub.steps.kmeans(X, 3), { signal: controller.signal }), { name: 'AbortError' });
});

test('metrics edge cases', () => {
  assert.equal(ub.ari([0, 0, 1, 1], [5, 5, 3, 3]), 1);
  assert.equal(ub.ami(['a', 'b'], ['x', 'y']), 1);
  assert.equal(ub.ari([], []), 1);
  assert.throws(() => ub.ami([0, 1], [0, 1], { average: 'median' }), RangeError);
  assert.throws(() => ub.trustworthiness(X, X, 200), RangeError);
});

test('tpe minimizes a mixed objective', async () => {
  const space = { x: ub.uniform(-5, 5), lr: ub.loguniform(1e-5, 1e-1), n: ub.integer(1, 20), kind: ub.choice('a', 'b', 'c') };
  const f = p => (p.x - 1.5) ** 2 + Math.log10(p.lr / 1e-3) ** 2 + (p.n - 7) ** 2 + (p.kind !== 'b');
  const best = [];
  for (let seed = 0; seed < 5; seed++) best.push((await ub.minimize(f, space, { nTrials: 80, seed })).bestValue);
  const rnd = [];
  for (let seed = 0; seed < 5; seed++) rnd.push((await ub.minimize(f, space, { nTrials: 80, seed, nStartup: 80 })).bestValue);
  const median = a => a.sort((x, y) => x - y)[2];
  assert.ok(median(best) < 0.25 * median(rnd), `${median(best)} vs ${median(rnd)}`);
  const tpe = new ub.TPE(space, { seed: 1 });
  for (let t = 0; t < 30; t++) {
    const p = tpe.ask();
    assert.ok(Number.isInteger(p.n) && p.n >= 1 && p.n <= 20 && p.lr >= 1e-5 && p.lr <= 1e-1);
    tpe.tell(p, f(p));
  }
  assert.throws(() => tpe.tell({ x: 0 }, 1), RangeError);
  const p = tpe.ask();
  for (const bad of [{ x: 99 }, { x: '0' }, { n: 2.5 }, { kind: 'z' }]) assert.throws(() => tpe.tell({ ...p, ...bad }, 1), RangeError);
  assert.throws(() => ub.uniform(1, 1), RangeError);
  assert.throws(() => ub.uniform(-1e308, 1e308), RangeError);
  assert.throws(() => ub.integer(0, 2 ** 60), /9007199254740991/);
  const many = new ub.TPE({ x: ub.uniform(0, 1) }, { seed: 0 });
  for (let t = 0; t < 200_000; t++) many.tell({ x: 0.5 }, 200_000 - t);
  assert.equal(many.result().bestValue, 1);
});
