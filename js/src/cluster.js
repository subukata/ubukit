/**
 * Partitional clustering on the shared engine. Each method only defines how
 * memberships follow from squared distances; see docs/algorithms.md.
 * Every function here is a generator: use run()/runAsync() or the wrappers in index.js.
 */
import {
  argmaxRows, argminRows, asData, checkInt, checkMaxIter, checkNumber, copyMatrix, iterate, labelMean,
  lloyd, mapMatrix, shift, softmaxRows, start, sumMinMax, TINY, weightedMean,
} from './core.js';

/** @typedef {import('./core.js').MatrixLike} MatrixLike */
/** @typedef {import('./core.js').Result} Result */
/** @typedef {import('./core.js').Progress} Progress */
/** @typedef {{ init?: 'k-means++' | MatrixLike, maxIter?: number, seed?: number }} Common */

/** The Result of a clustering; it copies the state (see iterate). @returns {import('./core.js').Result} */
function result({ data, V, state: U, nIter, converged, history }, hard = false) {
  return {
    centers: shift(V, data.mean), labels: hard ? U.slice() : argmaxRows(U), membership: hard ? null : copyMatrix(U),
    nIter, converged, history, embedding: null,
  };
}

/**
 * Lloyd's k-means; distance ties go to the lowest center index. Iterates with
 * Hamerly's (2010) bounds, which skip only distance computations that cannot
 * change a label, so the iterates are Lloyd's.
 * @param {MatrixLike} X @param {number} k @param {Common} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* kmeans(X, k, { init = 'k-means++', maxIter = 300, seed } = {}) {
  const { data, V } = start(X, k, init, seed);
  const view = loop => result({ ...loop, state: loop.state.labels }, true);
  return yield* iterate(data, V, hamerly, { maxIter: checkMaxIter(maxIter), tol: 0, view });
}

/**
 * Lloyd's step with Hamerly's bounds; the state is { labels, lower }. A point
 * keeps its label without computing its other distances when its exact
 * distance to its center is below both half the distance from that center to
 * the nearest other one (which proves the center nearest, for any labeling)
 * and a lower bound on its distance to every other center: the second-nearest
 * distance when last computed, minus how far the other centers have moved
 * since. Points within 1e-9 of a bound are recomputed, so rounding never
 * decides a label. The bounds hold for one data set, so new data start them
 * again (iterate keeps no state by default).
 * @type {import('./core.js').Step}
 */
function hamerly(data, V, state) {
  const X = data.X, { rows: n, cols: d, data: x } = X;
  {
    const k = V.rows, v = V.data;
    // Fresh arrays: a state once returned is never modified (see iterate).
    const labels = Int32Array.from(state?.labels ?? new Int32Array(n));
    const lower = Float64Array.from(state?.lower ?? new Float64Array(n));
    const dist = (i, j) => {
      let s = 0;
      for (let f = 0, a = i * d, b = j * d; f < d; f++) {
        const t = x[a + f] - v[b + f];
        s += t * t;
      }
      return s;
    };
    const half = new Float64Array(k).fill(Infinity);
    for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) {
      let s = 0;
      for (let f = 0; f < d; f++) s += (v[a * d + f] - v[b * d + f]) ** 2;
      half[a] = Math.min(half[a], Math.sqrt(s) / 2);
      half[b] = Math.min(half[b], Math.sqrt(s) / 2);
    }
    let objective = 0;
    for (let i = 0; i < n; i++) {
      let own = dist(i, labels[i]);
      if (Math.sqrt(own) >= Math.max(half[labels[i]], lower[i]) * (1 - 1e-9)) {
        let best = Infinity, second = Infinity, nearest = 0;
        for (let j = 0; j < k; j++) {
          const s = dist(i, j);
          if (s < best) second = best, best = s, nearest = j;
          else if (s < second) second = s;
        }
        labels[i] = nearest, own = best, lower[i] = Math.sqrt(second);
      }
      objective += own;
    }
    const W = labelMean(X, labels, V);
    if (k > 1) {
      // Moving the centers loosens each lower bound by the largest move of another center.
      let first = 0, second = 0, top = 0;
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let f = 0; f < d; f++) s += (W.data[j * d + f] - v[j * d + f]) ** 2;
        const m = Math.sqrt(s);
        if (m > first) second = first, first = m, top = j;
        else if (m > second) second = m;
      }
      for (let i = 0; i < n; i++) lower[i] -= labels[i] === top ? second : first;
    }
    return { V: W, state: { labels, lower }, value: objective };
  }
}

/**
 * Fuzzy c-means: u_ic proportional to d_ic^(-2/(m-1)), as powers of the ratios d_min / d_ic <= 1.
 * In high dimensions a large m (from about D/(D-2) for isotropic data) can draw every center to
 * the mean of X; see "Degenerate solutions" in docs/algorithms.md.
 * @param {MatrixLike} X @param {number} k @param {Common & { m?: number, tol?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* fcm(X, k, { m = 2, init = 'k-means++', maxIter = 300, tol = 1e-6, seed } = {}) {
  checkNumber(m, 'm', 1, true);
  const { data, V } = start(X, k, init, seed);
  const step = lloyd({
    assign: (_, D) => fuzzyMemberships(D, m),
    update: (data, U, V) => weightedMean(data.X, fuzzyWeights(U, m), V),
    // At the memberships of these distances, sum_c u^m d^2 = d_min^2 u_max^(m-1) per point.
    objective: (D, U) => sumMinMax(D, U, (d, u) => d * u ** (m - 1)),
  });
  return yield* iterate(data, V, step, { maxIter: checkMaxIter(maxIter), tol: checkNumber(tol, 'tol', 0), view: result });
}

/**
 * (d_min / d)^(2/(m-1)) per row, normalized: the softmax of -log d^2 / (m - 1)
 * without logarithms or exponentials (none at all for m = 2).
 */
function fuzzyMemberships(D, m) {
  const { rows: n, cols: k, data } = D, a = 1 / (m - 1), out = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    let low = Infinity, sum = 0;
    for (let j = 0; j < k; j++) low = Math.min(low, Math.max(data[i * k + j], TINY));
    for (let j = 0; j < k; j++) sum += out[i * k + j] = (low / Math.max(data[i * k + j], TINY)) ** a;
    for (let j = 0; j < k; j++) out[i * k + j] /= sum;
  }
  return { data: out, rows: n, cols: k };
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
 * Entropy-regularized fuzzy c-means: u_ic = softmax_c(-d_ic^2 / tau). A tau of at least twice the
 * largest variance of X can draw every center to the mean of X; see docs/algorithms.md.
 * @param {MatrixLike} X @param {number} k @param {Common & { tau?: number, tol?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* efcm(X, k, { tau = 1, init = 'k-means++', maxIter = 300, tol = 1e-6, seed } = {}) {
  checkNumber(tau, 'tau', 0, true);
  const { data, V } = start(X, k, init, seed);
  const step = lloyd({
    assign: (_, D) => softmaxRows(mapMatrix(D, d => -d / tau)),
    update: (data, U, V) => weightedMean(data.X, U, V),
    // At the memberships of these distances, sum_c u d^2 + tau u log u = d_min^2 + tau log u_max per point.
    objective: (D, U) => sumMinMax(D, U, (d, u) => d + tau * Math.log(u)),
  });
  return yield* iterate(data, V, step, { maxIter: checkMaxIter(maxIter), tol: checkNumber(tol, 'tol', 0), view: result });
}

/**
 * Rough c-means (p = 1) and ExRCM: cluster c is admissible when
 * d_ic^p <= (alpha d_min)^p + beta^p; membership is shared equally.
 * @param {MatrixLike} X @param {number} k @param {Common & { alpha?: number, beta?: number, p?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* rcm(X, k, { alpha = 1.1, beta = 0, p = 1, init = 'k-means++', maxIter = 300, seed } = {}) {
  checkNumber(alpha, 'alpha', 1);
  checkNumber(beta, 'beta', 0);
  checkNumber(p, 'p', 0, true);
  const { data, V } = start(X, k, init, seed);
  const assign = (_, D) => {
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
  const step = lloyd({ assign, update: (data, U, V) => weightedMean(data.X, U, V) });
  return yield* iterate(data, V, step, { maxIter: checkMaxIter(maxIter), tol: 0, view: result });
}

/**
 * Rough membership c-means: R = P H with P the row-normalized
 * delta-neighborhood graph (self included) and H the nearest-center one-hot.
 * @param {MatrixLike} X @param {number} k @param {number} delta @param {Common & { maxEdges?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* rmcm(X, k, delta, { init = 'k-means++', maxIter = 300, maxEdges = 10_000_000, seed } = {}) {
  checkNumber(delta, 'delta', 0);
  checkInt(maxEdges, 'maxEdges', 1);
  // The data with their neighborhood graph, built again for new data.
  const prepare = (rows, nFeatures) => {
    const d = asData(rows, nFeatures);
    return { ...d, graph: neighborhood(d.X, delta, maxEdges) };
  };
  const { data, V } = start(X, k, init, seed, prepare);
  const assign = ({ graph }, D) => {
    const labels = argminRows(D), n = labels.length, R = { data: new Float64Array(n * k), rows: n, cols: k };
    for (let i = 0; i < n; i++) {
      const first = graph.offsets[i], end = graph.offsets[i + 1];
      for (let e = first; e < end; e++) R.data[i * k + labels[graph.neighbors[e]]] += 1 / (end - first);
    }
    return R;
  };
  const step = lloyd({ assign, update: (data, U, V) => weightedMean(data.X, U, V) });
  return yield* iterate(data, V, step, { maxIter: checkMaxIter(maxIter), tol: 0, view: result, prepare });
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

