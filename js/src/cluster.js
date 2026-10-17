/**
 * Partitional clustering on the shared engine. The fuzzy and rough c-means
 * define how memberships follow from squared distances and run the standard
 * step; k-means runs Lloyd's step with Hamerly's bounds. See docs/algorithms.md.
 * Every function here is a generator: use run()/runAsync() or the wrappers in index.js.
 */
import {
  argmaxRows, argminRows, asData, checkInt, checkMaxIter, checkNumber, copyMatrix, fitted, iterate, labelMean,
  lloyd, named, nearest, rejectUnknown, schedule, softmin, start, startFixed, TINY, tolerance, unchanged, weightedMean,
} from './core.js';

/** @typedef {import('./core.js').MatrixLike} MatrixLike */
/** @typedef {import('./core.js').Result} Result */
/** @typedef {import('./core.js').Progress} Progress */
/** @typedef {{ init?: 'k-means++' | MatrixLike, maxIter?: number, seed?: number }} Common */

/** The view of soft and rough clusterings: memberships are the state, copied (see iterate). */
const soft = loop => fitted(loop, argmaxRows(loop.state), copyMatrix(loop.state));

/**
 * Lloyd's k-means; distance ties go to the lowest center index. Iterates with
 * Hamerly's (2010) bounds, which skip only distance computations that cannot
 * change a label, so the iterates are Lloyd's.
 * @param {MatrixLike} X @param {number} k @param {Common} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* kmeans(X, k, options) {
  const { init = 'k-means++', maxIter = 300, seed, ...unknown } = named(options);
  rejectUnknown(unknown);
  checkMaxIter(maxIter);
  const { data, V } = start(X, k, init, seed);
  const view = loop => fitted(loop, loop.state.labels.slice());
  return yield* iterate(data, V, hamerly, { stopping: tolerance(0, maxIter), view });
}

/**
 * The labels of X by the nearest of the fixed centers (apply.kmeans); ties go
 * to the lowest index, as in kmeans, and history holds the sum of squared
 * distances to the nearest centers.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {Record<string, never>} [options] none
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applyKmeans(X, centers, options) {
  const { ...unknown } = named(options);
  rejectUnknown(unknown);
  const { data, V, fixed } = startFixed(X, centers);
  const step = (data, V) => {
    const { rows: n, cols: d, data: x } = data.X, labels = nearest(data.X, V);
    let value = 0;
    for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) value += (x[i * d + f] - V.data[labels[i] * d + f]) ** 2;
    return { V, state: labels, value };
  };
  return yield* iterate(data, V, step, { stopping: schedule(1), view: loop => fitted(loop, loop.state.slice()), fixed });
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

/**
 * Fuzzy c-means: u_ic proportional to d_ic^(-2/(m-1)), as powers of the ratios d_min / d_ic <= 1.
 * In high dimensions a large m (from about D/(D-2) for isotropic data) can draw every center to
 * the mean of X; see "Degenerate solutions" in docs/algorithms.md.
 * @param {MatrixLike} X @param {number} k @param {Common & { m?: number, tol?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* fcm(X, k, options) {
  const { m = 2, init = 'k-means++', maxIter = 300, tol = 1e-6, seed, ...unknown } = named(options);
  rejectUnknown(unknown);
  const rule = fuzzy(m);
  checkMaxIter(maxIter);
  checkNumber(tol, 'tol', 0);
  const { data, V } = start(X, k, init, seed);
  return yield* iterate(data, V, lloyd(rule), { stopping: tolerance(tol, maxIter), view: soft });
}

/**
 * The FCM memberships of X for the fixed centers (apply.fcm); give the m they were trained with.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {{ m?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applyFcm(X, centers, options) {
  const { m = 2, ...unknown } = named(options);
  rejectUnknown(unknown);
  return yield* applied(X, centers, fuzzy(m).assign);
}

/** FCM's assignment and update for the fuzzifier m. */
function fuzzy(m) {
  checkNumber(m, 'm', 1, true);
  return {
    assign: (_, D) => fuzzyMemberships(D, m),
    update: (data, U, V) => weightedMean(data.X, fuzzyWeights(U, m), V),
  };
}

/**
 * (d_min / d)^(2/(m-1)) per row, normalized: the softmax of -log d^2 / (m - 1)
 * without logarithms or exponentials (none at all for m = 2). The largest
 * ratio is 1, so u_max = 1 / sum, and the objective at these memberships,
 * sum_c u_ic^m d_ic^2, is d_min^2 u_max^(m-1) per point.
 */
function fuzzyMemberships(D, m) {
  const { rows: n, cols: k, data } = D, a = 1 / (m - 1), out = new Float64Array(n * k);
  let value = 0;
  for (let i = 0; i < n; i++) {
    let low = Infinity, sum = 0;
    for (let j = 0; j < k; j++) low = Math.min(low, Math.max(data[i * k + j], TINY));
    for (let j = 0; j < k; j++) sum += out[i * k + j] = (low / Math.max(data[i * k + j], TINY)) ** a;
    for (let j = 0; j < k; j++) out[i * k + j] /= sum;
    value += low * sum ** (1 - m);
  }
  return { U: { data: out, rows: n, cols: k }, value };
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
export function* efcm(X, k, options) {
  const { tau = 1, init = 'k-means++', maxIter = 300, tol = 1e-6, seed, ...unknown } = named(options);
  rejectUnknown(unknown);
  const assign = entropic(tau);
  checkMaxIter(maxIter);
  checkNumber(tol, 'tol', 0);
  const { data, V } = start(X, k, init, seed);
  return yield* iterate(data, V, lloyd({ assign }), { stopping: tolerance(tol, maxIter), view: soft });
}

/**
 * The EFCM memberships of X for the fixed centers (apply.efcm); give the tau they were trained with.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {{ tau?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applyEfcm(X, centers, options) {
  const { tau = 1, ...unknown } = named(options);
  rejectUnknown(unknown);
  return yield* applied(X, centers, entropic(tau));
}

/** EFCM's assignment for the temperature tau. */
function entropic(tau) {
  checkNumber(tau, 'tau', 0, true);
  return (_, D) => softmin(D, tau, 'tau');
}

/**
 * Rough c-means (p = 1) and ExRCM: cluster c is admissible when
 * d_ic^p <= (alpha d_min)^p + beta^p; membership is shared equally.
 * @param {MatrixLike} X @param {number} k @param {Common & { alpha?: number, beta?: number, p?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* rcm(X, k, options) {
  const { alpha = 1.1, beta = 0, p = 1, init = 'k-means++', maxIter = 300, seed, ...unknown } = named(options);
  rejectUnknown(unknown);
  const assign = rough(alpha, beta, p);
  checkMaxIter(maxIter);
  const { data, V } = start(X, k, init, seed);
  return yield* iterate(data, V, lloyd({ assign }), { stopping: tolerance(0, maxIter), view: soft });
}

/**
 * The rough memberships of X for the fixed centers (apply.rcm); give the alpha, beta and p they
 * were trained with.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {{ alpha?: number, beta?: number, p?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applyRcm(X, centers, options) {
  const { alpha = 1.1, beta = 0, p = 1, ...unknown } = named(options);
  rejectUnknown(unknown);
  return yield* applied(X, centers, rough(alpha, beta, p));
}

/** RCM's assignment: an equal share of each admissible cluster. */
function rough(alpha, beta, p) {
  checkNumber(alpha, 'alpha', 1);
  checkNumber(beta, 'beta', 0);
  checkNumber(p, 'p', 0, true);
  return (_, D) => {
    const { rows: n, cols: k } = D, U = { data: new Float64Array(n * k), rows: n, cols: k };
    for (let i = 0; i < n; i++) {
      let dmin = Infinity;
      for (let j = 0; j < k; j++) dmin = Math.min(dmin, Math.sqrt(D.data[i * k + j]));
      const a = alpha * dmin, large = Math.max(a, beta), ratio = large > 0 ? Math.min(a, beta) / large : 0;
      // 1 ** Infinity is NaN in JavaScript but 1 in IEEE pow, Python's included,
      // which is what (1 + 0)^(1/p) is for any p: a p whose 1 / p overflows
      // gave NaN memberships here, where Python gives the limit.
      const radius = ratio > 0 ? large * (1 + ratio ** p) ** (1 / p) : large;
      let count = 0;
      for (let j = 0; j < k; j++) if (Math.sqrt(D.data[i * k + j]) <= radius) U.data[i * k + j] = 1, count++;
      for (let j = 0; j < k; j++) U.data[i * k + j] /= count;
    }
    return { U, value: null };
  };
}

/**
 * Rough membership c-means: R = P H with P the row-normalized
 * delta-neighborhood graph (self included) and H the nearest-center one-hot.
 * delta, the neighborhood radius, is a required option.
 * @param {MatrixLike} X @param {number} k @param {Common & { delta: number, maxEdges?: number }} options
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* rmcm(X, k, options) {
  const { delta, init = 'k-means++', maxIter = 300, maxEdges = 10_000_000, seed, ...unknown } = named(options);
  rejectUnknown(unknown);
  const { assign, prepare } = roughMembership(delta, maxEdges);
  checkMaxIter(maxIter);
  const { data, V } = start(X, k, init, seed, prepare);
  return yield* iterate(data, V, lloyd({ assign }), { stopping: tolerance(0, maxIter), view: soft, prepare });
}

/**
 * The rough memberships of X for the fixed centers (apply.rmcm). The neighborhoods are those
 * within X, so the memberships of a point depend on the points given with it: X given whole and
 * X given in parts differ, and a point given alone gets the hard assignment of k-means.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {{ delta: number, maxEdges?: number }} options
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applyRmcm(X, centers, options) {
  const { delta, maxEdges = 10_000_000, ...unknown } = named(options);
  rejectUnknown(unknown);
  const { assign, prepare } = roughMembership(delta, maxEdges);
  return yield* applied(X, centers, assign, prepare);
}

/** RMCM's assignment, and the preparation that builds each data set's graph. */
function roughMembership(delta, maxEdges) {
  checkNumber(delta, 'delta', 0);
  checkInt(maxEdges, 'maxEdges', 1);
  return {
    // The data with their neighborhood graph, built again for new data.
    prepare: (rows, nFeatures) => {
      const d = asData(rows, nFeatures);
      return { ...d, graph: neighborhood(d.X, delta, maxEdges) };
    },
    assign: ({ graph }, D) => {
      const labels = argminRows(D), n = labels.length, k = D.cols, R = { data: new Float64Array(n * k), rows: n, cols: k };
      for (let i = 0; i < n; i++) {
        const first = graph.offsets[i], end = graph.offsets[i + 1];
        for (let e = first; e < end; e++) R.data[i * k + labels[graph.neighbors[e]]] += 1 / (end - first);
      }
      return { U: R, value: null };
    },
  };
}

/** A soft or rough clustering applied: one assignment, the centers fixed. */
function* applied(X, centers, assign, prepare = asData) {
  const { data, V, fixed } = startFixed(X, centers, null, prepare);
  const step = lloyd({ assign, update: unchanged });
  return yield* iterate(data, V, step, { stopping: schedule(1), view: soft, prepare, fixed });
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

