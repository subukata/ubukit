import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X, nearestError } from './helpers.js';

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
