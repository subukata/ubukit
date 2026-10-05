/**
 * Partitional clustering on the shared engine. Each method only defines how
 * memberships follow from squared distances; see docs/algorithms.md.
 * Every function here is a generator: use run()/runAsync() or the wrappers in index.js.
 */
import {
  argmaxRows, argminRows, checkInt, checkNumber, iterate, labelMean, lloyd, mapMatrix,
  prepare, shift, softmaxRows, TINY, weightedMean,
} from './core.js';

/** @typedef {import('./core.js').MatrixLike} MatrixLike */
/** @typedef {import('./core.js').Result} Result */
/** @typedef {import('./core.js').Progress} Progress */
/** @typedef {{ init?: 'k-means++' | MatrixLike, maxIter?: number, seed?: number }} Common */

/** @returns {import('./core.js').Result} */
function result(mean, { V, state: U, nIter, converged, history }, hard = false) {
  return {
    centers: shift(V, mean), labels: hard ? U : argmaxRows(U), membership: hard ? null : U,
    nIter, converged, history, embedding: null,
  };
}

/**
 * Lloyd's k-means; distance ties go to the lowest center index.
 * @param {MatrixLike} X @param {number} k @param {Common} [options]
 * @returns {Generator<Progress, Result>}
 */
export function* kmeans(X, k, { init = 'k-means++', maxIter = 300, seed } = {}) {
  const p = prepare(X, k, init, seed);
  const step = lloyd(p.X, {
    assign: D => argminRows(D),
    update: (labels, V) => labelMean(p.X, labels, V),
    objective: (D, labels) => {
      let J = 0;
      for (let i = 0; i < labels.length; i++) J += D.data[i * D.cols + labels[i]];
      return J;
    },
  });
  const out = yield* iterate(p.X, p.V, step, { maxIter: checkInt(maxIter, 'maxIter', 1), tol: 0 });
  return result(p.mean, out, true);
}

/**
 * Fuzzy c-means: u_ic proportional to d_ic^(-2/(m-1)), as a log-domain softmax.
 * @param {MatrixLike} X @param {number} k @param {Common & { m?: number, tol?: number }} [options]
 * @returns {Generator<Progress, Result>}
 */
export function* fcm(X, k, { m = 2, init = 'k-means++', maxIter = 300, tol = 1e-6, seed } = {}) {
  checkNumber(m, 'm', 1, true);
  const p = prepare(X, k, init, seed);
  const step = lloyd(p.X, {
    assign: D => softmaxRows(mapMatrix(D, d => Math.log(Math.max(d, TINY)) / (1 - m))),
    update: (U, V) => weightedMean(p.X, fuzzyWeights(U, m), V),
    objective: (D, U) => {
      let J = 0;
      for (let i = 0; i < U.data.length; i++) J += U.data[i] ** m * D.data[i];
      return J;
    },
  });
  const out = yield* iterate(p.X, p.V, step, { maxIter: checkInt(maxIter, 'maxIter', 1), tol: checkNumber(tol, 'tol', 0) });
  return result(p.mean, out);
}

/**
 * u^m with each column scaled by its maximum first, which leaves the weighted
 * means unchanged and keeps u^m from underflowing to all zeros for large m.
 */
function fuzzyWeights(U, m) {
  const { rows: n, cols: k, data } = U, top = new Float64Array(k).fill(TINY), out = new Float64Array(n * k);
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) top[j] = Math.max(top[j], data[i * k + j]);
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[i * k + j] = (data[i * k + j] / top[j]) ** m;
  return { data: out, rows: n, cols: k };
}

/**
 * Entropy-regularized fuzzy c-means: u_ic = softmax_c(-d_ic^2 / tau).
 * @param {MatrixLike} X @param {number} k @param {Common & { tau?: number, tol?: number }} [options]
 * @returns {Generator<Progress, Result>}
 */
export function* efcm(X, k, { tau = 1, init = 'k-means++', maxIter = 300, tol = 1e-6, seed } = {}) {
  checkNumber(tau, 'tau', 0, true);
  const p = prepare(X, k, init, seed);
  const step = lloyd(p.X, {
    assign: D => softmaxRows(mapMatrix(D, d => -d / tau)),
    update: (U, V) => weightedMean(p.X, U, V),
    objective: (D, U) => {
      let J = 0;
      for (let i = 0; i < U.data.length; i++) {
        const u = U.data[i];
        J = J + u * D.data[i] + (u > 0 ? tau * u * Math.log(u) : 0);
      }
      return J;
    },
  });
  const out = yield* iterate(p.X, p.V, step, { maxIter: checkInt(maxIter, 'maxIter', 1), tol: checkNumber(tol, 'tol', 0) });
  return result(p.mean, out);
}

/**
 * Rough c-means (p = 1) and ExRCM: cluster c is admissible when
 * d_ic^p <= (alpha d_min)^p + beta^p; membership is shared equally.
 * @param {MatrixLike} X @param {number} k @param {Common & { alpha?: number, beta?: number, p?: number }} [options]
 * @returns {Generator<Progress, Result>}
 */
export function* rcm(X, k, { alpha = 1.1, beta = 0, p = 1, init = 'k-means++', maxIter = 300, seed } = {}) {
  checkNumber(alpha, 'alpha', 1);
  checkNumber(beta, 'beta', 0);
  checkNumber(p, 'p', 0, true);
  const prep = prepare(X, k, init, seed);
  const assign = D => {
    const { rows: n, cols: kk } = D, U = { data: new Float64Array(n * kk), rows: n, cols: kk };
    for (let i = 0; i < n; i++) {
      let dmin = Infinity;
      for (let j = 0; j < kk; j++) dmin = Math.min(dmin, Math.sqrt(D.data[i * kk + j]));
      const a = alpha * dmin, large = Math.max(a, beta), ratio = large > 0 ? Math.min(a, beta) / large : 0;
      const radius = large * (1 + ratio ** p) ** (1 / p);
      let count = 0;
      for (let j = 0; j < kk; j++) if (Math.sqrt(D.data[i * kk + j]) <= radius) U.data[i * kk + j] = 1, count++;
      for (let j = 0; j < kk; j++) U.data[i * kk + j] /= count;
    }
    return U;
  };
  const step = lloyd(prep.X, { assign, update: (U, V) => weightedMean(prep.X, U, V) });
  const out = yield* iterate(prep.X, prep.V, step, { maxIter: checkInt(maxIter, 'maxIter', 1), tol: 0 });
  return result(prep.mean, out);
}

/**
 * Rough membership c-means: R = P H with P the row-normalized
 * delta-neighborhood graph (self included) and H the nearest-center one-hot.
 * @param {MatrixLike} X @param {number} k @param {number} delta @param {Common & { maxEdges?: number }} [options]
 * @returns {Generator<Progress, Result>}
 */
export function* rmcm(X, k, delta, { init = 'k-means++', maxIter = 300, maxEdges = 10_000_000, seed } = {}) {
  checkNumber(delta, 'delta', 0);
  const p = prepare(X, k, init, seed);
  const graph = neighborhood(p.X, delta, checkInt(maxEdges, 'maxEdges', 1));
  const assign = D => {
    const labels = argminRows(D), n = labels.length, R = { data: new Float64Array(n * k), rows: n, cols: k };
    for (let i = 0; i < n; i++) {
      const start = graph.offsets[i], end = graph.offsets[i + 1];
      for (let e = start; e < end; e++) R.data[i * k + labels[graph.neighbors[e]]] += 1 / (end - start);
    }
    return R;
  };
  const step = lloyd(p.X, { assign, update: (U, V) => weightedMean(p.X, U, V) });
  const out = yield* iterate(p.X, p.V, step, { maxIter: checkInt(maxIter, 'maxIter', 1), tol: 0 });
  return result(p.mean, out);
}

/** Adjacency lists of ||x_i - x_j|| <= delta (self included), O(N^2 D). */
function neighborhood(X, delta, maxEdges) {
  const { rows: n, cols: d, data } = X, lists = Array.from({ length: n }, (_, i) => [i]);
  const r2 = delta * delta;
  let edges = 0;
  // Directed edges, self loops included, as counted in Python.
  const add = count => {
    if ((edges += count) > maxEdges) throw new RangeError(`delta=${delta} gives more than maxEdges=${maxEdges} edges`);
  };
  add(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let f = 0; f < d && s <= r2; f++) s += (data[i * d + f] - data[j * d + f]) ** 2;
      if (s <= r2) {
        add(2);
        lists[i].push(j);
        lists[j].push(i);
      }
    }
  }
  const offsets = new Int32Array(n + 1);
  lists.forEach((l, i) => (offsets[i + 1] = offsets[i] + l.length));
  const neighbors = new Int32Array(offsets[n]);
  lists.forEach((l, i) => neighbors.set(l, offsets[i]));
  return { offsets, neighbors };
}

