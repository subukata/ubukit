import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X, gauss } from './helpers.js';

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

test('somOlp history is the objective', () => {
  // Two iterations from given prototypes, against the definition (the history
  // uses an identity that holds at the memberships of the costs).
  const lam = 0.7, gamma = 0.5, R = Array.from({ length: 12 }, (_, j) => [Math.floor(j / 4), j % 4]);
  const W0 = R.map(([a, b]) => [a + 0.3, b - 0.2]);
  const sq = (a, b) => a.reduce((s, v, f) => s + (v - b[f]) ** 2, 0);
  const softmax = row => {
    const top = Math.max(...row), e = row.map(v => Math.exp(v - top)), s = e.reduce((a, b) => a + b);
    return e.map(v => v / s);
  };
  const J = (P, C) => P.reduce((s, row, i) => s + row.reduce((t, p, j) => t + p * C[i][j] + (p > 0 ? lam * p * Math.log(p) : 0), 0), 0);
  let C = X.map(x => W0.map(w => sq(x, w))), P = C.map(row => softmax(row.map(c => -c / lam)));
  const expected = [J(P, C)];
  const W = R.map((_, j) => {
    const mass = P.reduce((s, row) => s + row[j], 0);
    return [0, 1].map(f => P.reduce((s, row, i) => s + row[j] * X[i][f], 0) / mass);
  });
  const V = P.map(row => [0, 1].map(c => row.reduce((s, p, j) => s + p * R[j][c], 0)));
  C = X.map((x, i) => W.map((w, j) => sq(x, w) + gamma * sq(V[i], R[j])));
  P = C.map(row => softmax(row.map(c => -c / lam)));
  expected.push(J(P, C));
  const r = ub.somOlp(X, [3, 4], { lam, gamma, init: W0, maxIter: 2, tol: 0 });
  expected.forEach((e, t) => assert.ok(Math.abs(r.history[t] - e) <= 1e-9 * Math.abs(e), `${r.history[t]} vs ${e}`));
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
