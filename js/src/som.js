/**
 * Self-organizing maps: online SOM, batch SOM and SOM-OLP (generators).
 * Unit j = row * cols + col sits at (row, col); prototypes start on the
 * leading principal plane unless given.
 */
import {
  argmaxRows, asData, center, checkInt, checkMaxIter, checkNumber, checkScale, copyMatrix, fitted, isList, iterate, labelSums, mapMatrix,
  matrix, MAX_SQ_NORM, named, nearest, random, rejectUnknown, schedule, shift, softmin, sqdist, startFixed, steady, tolerance, weightedMean,
} from './core.js';

/** @typedef {import('./core.js').MatrixLike} MatrixLike */
/** @typedef {import('./core.js').Result} Result */
/** @typedef {import('./core.js').Progress} Progress */
/** @typedef {number[] | MatrixLike} Grid [rows, cols] or a (K, Q) matrix of unit coordinates. */
/** @typedef {{ init?: 'pca' | MatrixLike }} MapInit */

/**
 * Online (sequential) SOM: w_j += lr_t h_t(j, bmu) (x - w_j); sigma and lr
 * decay geometrically over all epochs * N updates.
 * @param {MatrixLike} X @param {Grid} [grid]
 * @param {MapInit & { epochs?: number, sigma?: number, sigmaEnd?: number, lr?: number, lrEnd?: number, shuffle?: boolean, seed?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* som(X, grid = [10, 10], options) {
  const { epochs = 10, sigma, sigmaEnd = 0.5, lr = 0.5, lrEnd = 0.01, init = 'pca', shuffle = true, seed, ...unknown } = named(options);
  rejectUnknown(unknown);
  const R = gridCoordinates(grid);
  checkInt(epochs, 'epochs', 1);
  const [s0, s1] = sigmas(sigma, sigmaEnd, R);
  checkNumber(lr, 'lr', 0, true);
  checkNumber(lrEnd, 'lrEnd', 0, true);
  if (Math.max(lr, lrEnd) > 1) throw new RangeError('lr and lrEnd must be at most 1');
  const rand = random(seed), { data, W } = setup(X, R, init), k = R.rows, q = R.cols;
  const epoch = (data, W, _, e) => {
    const { rows: n, cols: d, data: x } = data.X, steps = epochs * n;
    const w = W.data.slice(), diff = new Float64Array(k * d), order = Int32Array.from({ length: n }, (_, i) => i);
    if (shuffle) for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let r = 0; r < n; r++) {
      const i = order[r], f = steps > 1 ? (e * n + r) / (steps - 1) : 0;
      const sg = geometric(s0, s1, f), eta = geometric(lr, lrEnd, f);
      let bmu = 0, best = Infinity;
      for (let j = 0; j < k; j++) {
        let dist = 0;
        for (let c = 0; c < d; c++) dist += (diff[j * d + c] = x[i * d + c] - w[j * d + c]) ** 2;
        if (dist < best) best = dist, bmu = j;
      }
      for (let j = 0; j < k; j++) {
        // The grid differences are divided by sg before squaring: a tiny width
        // keeps h = 1 for the winner and sends the others to 0, as in the limit.
        let g = 0;
        for (let c = 0; c < q; c++) {
          const t = (R.data[j * q + c] - R.data[bmu * q + c]) / sg;
          g += t * t;
        }
        const h = eta * Math.exp(-0.5 * g);
        for (let c = 0; c < d; c++) w[j * d + c] += h * diff[j * d + c];
      }
    }
    return { V: { data: w, rows: k, cols: d }, state: null, value: null };
  };
  return yield* iterate(data, W, epoch, { stopping: schedule(epochs), view: mapView(R) });
}

/**
 * Batch SOM: w_j = sum_i h(bmu_i, j) x_i / sum_i h(bmu_i, j), separable kernel.
 * @param {MatrixLike} X @param {number[]} [grid] [rows, cols]
 * @param {MapInit & { epochs?: number, sigma?: number, sigmaEnd?: number }} [options]
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* batchSom(X, grid = [10, 10], options) {
  const { epochs = 50, sigma, sigmaEnd = 0.5, init = 'pca', ...unknown } = named(options);
  rejectUnknown(unknown);
  const [rows, cols] = shape(grid), R = gridCoordinates(grid);
  checkInt(epochs, 'epochs', 1);
  const [s0, s1] = sigmas(sigma, sigmaEnd, R);
  const { data, W } = setup(X, R, init), d = data.X.cols, k = rows * cols;
  const kernel = (size, sg) => Float64Array.from({ length: size * size }, (_, i) => {
    const a = Math.floor(i / size) - (i % size);
    return Math.exp(-0.5 * (a / sg) ** 2); // finite for any width, the identity in its limit
  });
  const step = (data, W, _, t) => {
    const labels = nearest(data.X, W);
    const sg = epochs > 1 ? geometric(s0, s1, t / (epochs - 1)) : s0;
    const Kr = kernel(rows, sg), Kc = kernel(cols, sg);
    const { sums, counts } = labelSums(data.X, labels, k);
    // Smooth (sums | counts) along rows, then along columns.
    const width = d + 1, a = new Float64Array(k * width), b = new Float64Array(k * width);
    for (let j = 0; j < k; j++) { a.set(sums.subarray(j * d, (j + 1) * d), j * width); a[j * width + d] = counts[j]; }
    for (let r = 0; r < rows; r++) for (let r2 = 0; r2 < rows; r2++) {
      const h = Kr[r * rows + r2];
      for (let c = 0; c < cols; c++) for (let f = 0; f < width; f++) b[(r * cols + c) * width + f] += h * a[(r2 * cols + c) * width + f];
    }
    a.fill(0);
    for (let c = 0; c < cols; c++) for (let c2 = 0; c2 < cols; c2++) {
      const h = Kc[c * cols + c2];
      for (let r = 0; r < rows; r++) for (let f = 0; f < width; f++) a[(r * cols + c) * width + f] += h * b[(r * cols + c2) * width + f];
    }
    const out = W.data.slice();
    for (let j = 0; j < k; j++) {
      const den = a[j * width + d];
      if (den > 0) for (let f = 0; f < d; f++) out[j * d + f] = a[j * width + f] / den;
    }
    return { V: { data: out, rows: k, cols: d }, state: labels, value: null };
  };
  return yield* iterate(data, W, step, { stopping: schedule(epochs), view: mapView(R) });
}

/**
 * SOM with optimized latent positions (SOM-OLP, Ubukata): minimizes
 * sum p (||x - w||^2 + gamma ||v - r||^2) + lam sum p log p with v_i = sum_j p_ij r_j.
 * grid is [rows, cols] or a (K, Q) matrix of unit coordinates.
 * @param {MatrixLike} X @param {Grid | undefined} grid
 * @param {MapInit & { lam: number, gamma: number, pcaScale?: number, maxIter?: number, tol?: number }} options
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* somOlp(X, grid = [10, 10], options) {
  const { lam, gamma, init = 'pca', pcaScale = 2, maxIter = 100, tol = 1e-6, ...unknown } = named(options);
  rejectUnknown(unknown);
  const R = gridCoordinates(grid), assign = olp(R, lam, gamma);
  checkNumber(pcaScale, 'pcaScale', 0, true);
  checkMaxIter(maxIter);
  checkNumber(tol, 'tol', 0);
  const { data, W } = setup(X, R, init, pcaScale);
  const step = (data, W, P) => {
    const { U, value } = assign(data, W, P);
    return { V: weightedMean(data.X, U, W), state: U, value };
  };
  return yield* iterate(data, W, step, { stopping: tolerance(tol, maxIter), view: olpView(R), keep: samePoints });
}

/**
 * The best-matching units of X among the fixed prototypes (centers) of the units at grid (apply.som
 * and apply.batchSom). grid is [rows, cols] or a (K, Q) matrix, as the map's unitCoordinates;
 * embedding holds the coordinates of each point's unit.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {Grid} grid @param {Record<string, never>} [options] none
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applySom(X, centers, grid, options) {
  const { ...unknown } = named(options);
  rejectUnknown(unknown);
  const R = gridCoordinates(grid), { data, V, fixed } = startFixed(X, centers, R.rows);
  const step = (data, W) => ({ V: W, state: null, value: null }); // the view finds the best-matching units
  return yield* iterate(data, V, step, { stopping: schedule(1), view: mapView(R), fixed });
}

export const applyBatchSom = applySom;

/**
 * The memberships and latent positions of X on a fixed SOM-OLP map (apply.somOlp). centers are
 * the prototypes of the units at grid; give the lam and gamma they were trained with. The
 * memberships start from the distances alone, as a training run's first iteration does, and are
 * iterated with the prototypes fixed until no membership changes by more than tol, or maxIter
 * iterations (more than training's default: they settle slowly for some settings). Each point's
 * update depends on it alone, but the rule stops all points together, so a point's result can
 * depend on the points applied with it, mostly for one midway between two units. The latent positions of earlier memberships pull later ones, so
 * the iteration can settle elsewhere than training left the same points.
 * @param {MatrixLike} X @param {MatrixLike} centers @param {Grid} grid
 * @param {{ lam: number, gamma: number, maxIter?: number, tol?: number }} options
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* applySomOlp(X, centers, grid, options) {
  const { lam, gamma, maxIter = 1000, tol = 1e-6, ...unknown } = named(options);
  rejectUnknown(unknown);
  const R = gridCoordinates(grid), assign = olp(R, lam, gamma);
  checkMaxIter(maxIter);
  checkNumber(tol, 'tol', 0);
  const { data, V, fixed } = startFixed(X, centers, R.rows);
  const step = (data, W, P) => {
    const { U, value } = assign(data, W, P);
    return { V: W, state: U, value };
  };
  return yield* iterate(data, V, step, { stopping: steady(tol, maxIter), view: olpView(R), keep: samePoints, fixed });
}

/**
 * SOM-OLP's assignment for the units at R: assign(data, W, P) gives the memberships, from the
 * latent positions P R of the previous ones (none at first), with the objective at them.
 */
function olp(R, lam, gamma) {
  checkNumber(lam, 'lam', 0, true);
  checkNumber(gamma, 'gamma', 0);
  // The latent costs gamma ||v - r_j||^2 come from centered unit coordinates,
  // as distances come from centered data: rounding then follows the extent of
  // the units, not their distance from the origin. The first unit is
  // subtracted before the mean, which would otherwise be computed with an
  // error that follows the offset (coincident units then center to exactly
  // zero). The costs must stay finite beside the data's, which the scale
  // contract keeps below about 4e300. A latent position v lies among the units,
  // so ||v - r_j||^2 is at most their squared extent, bounded by four times
  // their largest squared norm once centered.
  const Rc = center(shift(R, R.data.slice(0, R.cols).map(v => -v))).X, { rows: k, cols: q, data: r } = Rc;
  let largest = 0;
  for (let j = 0; j < k; j++) {
    let s = 0;
    for (let c = 0; c < q; c++) s += r[j * q + c] ** 2;
    largest = Math.max(largest, s);
  }
  const extent = 4 * largest;
  if (gamma * extent > MAX_SQ_NORM) {
    throw new RangeError(`gamma is too large for the unit coordinates: gamma times their squared extent (${extent}) must be at most ${MAX_SQ_NORM}`);
  }
  return (data, W, P) => {
    let cost = sqdist(data.X, W);
    if (P) {
      const extra = sqdist(multiply(P, Rc), Rc).data;
      cost = mapMatrix(cost, (v, i) => v + gamma * extra[i]);
    }
    return softmin(cost, lam, 'lam'); // value: the objective at these memberships
  };
}

/**
 * SOM-OLP's memberships give each point's latent position for the next
 * iteration. Points are known only by their rows, so they carry over while the
 * number of rows stays the same, row i being the same point.
 */
const samePoints = (P, data) => (P.rows === data.X.rows ? P : null);

/** The view of SOM-OLP: a copy of the memberships (the next step reads them), and the latent positions. */
const olpView = R => loop => fitted(loop, argmaxRows(loop.state), copyMatrix(loop.state), multiply(loop.state, R), copyMatrix(R));

function multiply(A, B) {
  const out = new Float64Array(A.rows * B.cols);
  for (let i = 0; i < A.rows; i++) for (let j = 0; j < A.cols; j++) {
    const a = A.data[i * A.cols + j];
    if (a !== 0) for (let c = 0; c < B.cols; c++) out[i * B.cols + c] += a * B.data[j * B.cols + c];
  }
  return { data: out, rows: A.rows, cols: B.cols };
}

function shape(grid) {
  if (!Array.isArray(grid) || grid.length !== 2 || isList(grid[0])) throw new RangeError('grid must be [rows, cols]');
  return [checkInt(grid[0], 'rows', 1), checkInt(grid[1], 'cols', 1)];
}

/** Unit coordinates: [rows, cols] -> a rectangular grid; rows of coordinates (arrays or typed arrays, as matrix reads them) or a matrix -> a copy. */
function gridCoordinates(grid) {
  if (!Array.isArray(grid) || isList(grid[0])) return matrix(grid, 'grid'); // a copy: the run owns it
  const [rows, cols] = shape(grid);
  const data = new Float64Array(rows * cols * 2);
  for (let j = 0; j < rows * cols; j++) data[2 * j] = Math.floor(j / cols), data[2 * j + 1] = j % cols;
  return { data, rows: rows * cols, cols: 2 };
}

/** The data and the initial prototypes of the units at R (centered like the data). */
function setup(X, R, init, pcaScale = 2) {
  if (typeof init === 'string' && init !== 'pca') throw new RangeError("init must be 'pca' or a (nUnits, nFeatures) matrix");
  const data = asData(X), Xc = data.X;
  if (init === 'pca') {
    // A start made here passes the scale check that a given one must.
    const W = pcaInit(Xc, R, pcaScale);
    checkScale(shift(W, data.mean), 'the start that pcaScale gives', 'change pcaScale');
    return { data, W };
  }
  const W = matrix(init, 'init');
  if (W.rows !== R.rows || W.cols !== Xc.cols) throw new RangeError(`init must have shape (${R.rows}, ${Xc.cols})`);
  return { data, W: shift(W, data.mean.map(v => -v)) };
}

/**
 * The point f of the geometric schedule from a to b, as a^(1-f) b^f: no ratio
 * b / a is formed, which overflows or underflows for far-apart ends; f = 0, 1
 * give a, b.
 */
function geometric(a, b, f) {
  return a ** (1 - f) * b ** f;
}

function sigmas(sigma, sigmaEnd, R) {
  if (sigma === undefined) {
    let extent = 0;
    for (let c = 0; c < R.cols; c++) {
      let lo = Infinity, hi = -Infinity;
      for (let j = 0; j < R.rows; j++) lo = Math.min(lo, R.data[j * R.cols + c]), hi = Math.max(hi, R.data[j * R.cols + c]);
      extent = Math.max(extent, hi - lo);
    }
    sigma = extent / 2 || 1;
  }
  return [checkNumber(sigma, 'sigma', 0, true), checkNumber(sigmaEnd, 'sigmaEnd', 0, true)];
}

/** The view of a map: the best-matching units of the prototypes reached. */
function mapView(R) {
  return loop => {
    const labels = nearest(loop.data.X, loop.V), q = R.cols, emb = new Float64Array(labels.length * q);
    labels.forEach((j, i) => emb.set(R.data.subarray(j * q, (j + 1) * q), i * q));
    return fitted(loop, labels, null, { data: emb, rows: labels.length, cols: q }, copyMatrix(R));
  };
}

/** Spread the normalized grid over the leading principal axes of centered X. */
function pcaInit(X, R, scale) {
  const d = X.cols, W = new Float64Array(R.rows * d);
  principalAxes(X, Math.min(R.cols, d)).forEach(({ value, axis }, h) => {
    // Make the first clearly nonzero component positive (the largest one is
    // ambiguous: standardized 2-D data have axes (1, +-1)/sqrt(2)).
    const top = axis.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    const sign = Math.sign(axis.find(v => Math.abs(v) > 1e-6 * top) ?? 0), spread = scale * Math.sqrt(Math.max(value, 0));
    let mean = 0, extent = 0;
    for (let j = 0; j < R.rows; j++) mean += R.data[j * R.cols + h] / R.rows;
    for (let j = 0; j < R.rows; j++) extent = Math.max(extent, Math.abs(R.data[j * R.cols + h] - mean));
    extent = Math.max(extent, 1e-12);
    for (let j = 0; j < R.rows; j++) {
      const g = (R.data[j * R.cols + h] - mean) / extent * spread * sign;
      for (let f = 0; f < d; f++) W[j * d + f] += g * axis[f];
    }
  });
  return { data: W, rows: R.rows, cols: d };
}

/**
 * Leading q eigenpairs of the covariance X^T X / N by Rayleigh-Ritz on a
 * Krylov basis of at most 64 vectors (Lanczos with full reorthogonalization,
 * restarted from a fresh direction when the space closes). The basis spans
 * R^D when D <= 64, so the axes are then exact; each step costs O(N D) and
 * the D x D covariance is never formed.
 */
function principalAxes(X, q) {
  const { rows: n, cols: d, data } = X, size = Math.min(d, 64), rand = random(0), xv = new Float64Array(n);
  const basis = [], images = [];
  const dot = (a, b) => { let s = 0; for (let f = 0; f < d; f++) s += a[f] * b[f]; return s; };
  const covariance = v => {
    const out = new Float64Array(d);
    for (let i = 0; i < n; i++) { let s = 0; for (let f = 0; f < d; f++) s += data[i * d + f] * v[f]; xv[i] = s / n; }
    for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) out[f] += data[i * d + f] * xv[i];
    return out;
  };
  // v orthogonalized against the basis (twice, for stability) and normalized; null if it vanishes.
  const unit = (v, scale) => {
    for (let pass = 0; pass < 2; pass++) for (const u of basis) { const c = dot(u, v); for (let f = 0; f < d; f++) v[f] -= c * u[f]; }
    const length = norm(v);
    return length > 1e-12 * scale ? v.map(x => x / length) : null;
  };
  const fresh = () => unit(Float64Array.from({ length: d }, () => rand() - 0.5), 0);
  for (let v = fresh(); ;) {
    const w = covariance(v);
    basis.push(v);
    images.push(w);
    if (basis.length === size) break;
    v = unit(w.slice(), norm(w)) ?? fresh();
  }
  const H = new Float64Array(size * size);
  for (let a = 0; a < size; a++) for (let b = a; b < size; b++) {
    H[a * size + b] = H[b * size + a] = (dot(basis[a], images[b]) + dot(basis[b], images[a])) / 2;
  }
  const { values, vectors } = symmetricEigen(H, size);
  return Array.from(values.keys()).sort((a, b) => values[b] - values[a]).slice(0, q).map(e => ({
    value: values[e],
    axis: Array.from({ length: d }, (_, f) => basis.reduce((s, u, a) => s + vectors[a * size + e] * u[f], 0)),
  }));
}

/**
 * The Euclidean norm of v, its entries scaled by the largest before squaring,
 * so that data of any scale within the contract neither overflow nor vanish
 * (the squares of a covariance of data at 1e80 overflow, at 1e-100 underflow).
 */
function norm(v) {
  let top = 0;
  for (const x of v) top = Math.max(top, Math.abs(x));
  if (!(top > 0 && top < Infinity)) return top;
  let s = 0;
  for (const x of v) s += (x / top) ** 2;
  return top * Math.sqrt(s);
}

/** Cyclic Jacobi eigendecomposition of a small symmetric matrix; vectors are columns. */
function symmetricEigen(S, n) {
  const a = S.slice(), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  const offDiagonal = new Float64Array(n * n - n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let m = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) offDiagonal[m++] = a[i * n + j];
    if (norm(offDiagonal) <= 1e-15 * norm(a)) break;
    for (let p = 0; p < n - 1; p++) for (let r = p + 1; r < n; r++) {
      const apr = a[p * n + r];
      if (apr === 0) continue;
      const theta = (a[r * n + r] - a[p * n + p]) / (2 * apr);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.hypot(theta, 1));
      const c = 1 / Math.hypot(t, 1), s = t * c;
      for (let i = 0; i < n; i++) {
        const x = a[i * n + p], y = a[i * n + r];
        a[i * n + p] = c * x - s * y; a[i * n + r] = s * x + c * y;
      }
      for (let i = 0; i < n; i++) {
        const x = a[p * n + i], y = a[r * n + i];
        a[p * n + i] = c * x - s * y; a[r * n + i] = s * x + c * y;
      }
      for (let i = 0; i < n; i++) {
        const x = v[i * n + p], y = v[i * n + r];
        v[i * n + p] = c * x - s * y; v[i * n + r] = s * x + c * y;
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => a[i * n + i]), vectors: v };
}
