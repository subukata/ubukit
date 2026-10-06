/**
 * Shared numerics: matrices, validation, distances, initialization and the
 * engine. Every iterative method runs in one loop, iterate, which repeats the
 * method's step until the prototypes stop moving; it is a generator so callers
 * can observe, pause or cancel it. The fuzzy and rough c-means use the
 * standard step lloyd, D = ||x_i - v_c||^2 -> U = assign(D) -> V = update(U);
 * k-means (with Hamerly's bounds) and the maps, whose iterations have other
 * shapes, supply their own steps.
 */

/** @typedef {{ data: Float64Array, rows: number, cols: number }} Matrix */
/** @typedef {number[][] | Matrix} MatrixLike */
/**
 * For clusterings and SOM-OLP, labels and membership come from the last
 * assignment step, i.e. for the prototypes before the final update; they
 * coincide at a fixed point. For som and batchSom, labels are the
 * best-matching units of the final prototypes.
 * @typedef {object} Result
 * @property {Matrix} centers (K, D) prototypes.
 * @property {Int32Array} labels strongest membership (nearest prototype for kmeans, best-matching unit for som and batchSom).
 * @property {Matrix | null} membership (N, K) memberships; null for hard methods.
 * @property {number} nIter iterations performed (epochs for SOMs).
 * @property {boolean} converged stopping rule met (som and batchSom: schedule completed).
 * @property {Float64Array} history objective per iteration; empty when undefined.
 * @property {Matrix | null} embedding (N, Q) map coordinates for SOMs.
 */
/**
 * Yielded after every iteration (epoch for SOMs); result() is the Result the
 * run would return had it stopped there. It shares no arrays with the run,
 * so changing it leaves the iterations that follow unchanged. Passing rows to
 * the generator's next(X) runs the following iterations on them.
 * @typedef {{ iteration: number, result: () => Result }} Progress
 */

/** Smallest positive normal float64. */
export const TINY = 2.2250738585072014e-308;
// "Ordinary scale": row norms below 1e150 keep every squared distance between
// data, prototypes and grid points (~1e301 at most) far from float64 overflow,
// and some feature spanning at least 1e-150 (unless all rows are equal) keeps
// the squared distances of the data from underflowing to zero.
const MAX_SQ_NORM = 1e300;
const MIN_SPAN = 1e-150;

/**
 * Copy rows or a {data, rows, cols} matrix into a validated Float64 matrix
 * that is finite, non-empty and of ordinary scale.
 * @param {MatrixLike} values
 * @returns {Matrix}
 */
export function matrix(values, name = 'X') {
  /** @type {Matrix} */
  let out;
  if (Array.isArray(values)) {
    const rows = values.length;
    const cols = rows ? values[0].length : 0;
    out = { data: new Float64Array(rows * cols), rows, cols };
    values.forEach((row, i) => {
      if (row.length !== cols) throw new RangeError(`${name} rows must have equal length`);
      out.data.set(row, i * cols);
    });
  } else if (values && ArrayBuffer.isView(values.data)) {
    out = { data: Float64Array.from(values.data), rows: values.rows, cols: values.cols };
  } else {
    throw new TypeError(`${name} must be an array of rows or {data, rows, cols}`);
  }
  const { rows, cols, data } = out;
  if (!(Number.isSafeInteger(rows) && Number.isSafeInteger(cols) && rows > 0 && cols > 0) || data.length !== rows * cols) {
    throw new RangeError(`${name} must be a non-empty rows x cols matrix`);
  }
  const lo = new Float64Array(cols).fill(Infinity), hi = new Float64Array(cols).fill(-Infinity);
  for (let i = 0; i < rows; i++) {
    let norm = 0;
    for (let c = 0; c < cols; c++) {
      const v = data[i * cols + c];
      if (!Number.isFinite(v)) throw new RangeError(`${name} must contain only finite values`);
      norm += v * v;
      lo[c] = Math.min(lo[c], v);
      hi[c] = Math.max(hi[c], v);
    }
    if (!(norm < MAX_SQ_NORM)) throw new RangeError(`${name} is too large in scale for float64 distances; standardize it`);
  }
  let span = 0;
  for (let c = 0; c < cols; c++) span = Math.max(span, hi[c] - lo[c]);
  if (span > 0 && span < MIN_SPAN) throw new RangeError(`${name} is too small in scale for float64 distances; standardize it`);
  return out;
}

/**
 * Rows of a matrix as plain arrays.
 * @param {Matrix} m
 * @returns {number[][]}
 */
export function toRows(m) {
  return Array.from({ length: m.rows }, (_, i) => Array.from(m.data.subarray(i * m.cols, (i + 1) * m.cols)));
}

/**
 * Validate a safe integer in [low, high].
 * @param {number} value @param {string} name @param {number} low @param {number} [high]
 * @returns {number}
 */
export function checkInt(value, name, low, high = Infinity) {
  if (!Number.isSafeInteger(value) || value < low || value > high) {
    throw new RangeError(`${name} must be an integer ${high === Infinity ? `>= ${low}` : `in [${low}, ${high}]`}`);
  }
  return value;
}

/**
 * Validate a finite number >= low (> low when strict).
 * @param {number} value @param {string} name @param {number} [low] @param {boolean} [strict]
 * @returns {number}
 */
export function checkNumber(value, name, low = -Infinity, strict = false) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || (strict && value === low)) {
    throw new RangeError(`${name} must be a finite number ${strict ? '>' : '>='} ${low}`);
  }
  return value;
}

/** Seeded uniform [0, 1) generator (Mulberry32); random seed when omitted. */
export function random(seed) {
  let s = seed === undefined ? (Math.random() * 2 ** 32) >>> 0 : checkInt(seed, 'seed', 0, 2 ** 32 - 1);
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Center the columns of X; returns the centered copy and the mean. */
function center(X) {
  const { rows: n, cols: d, data } = X;
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) mean[f] += data[i * d + f];
  for (let f = 0; f < d; f++) mean[f] /= n;
  const out = new Float64Array(data.length);
  for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) out[i * d + f] = data[i * d + f] - mean[f];
  return { X: { data: out, rows: n, cols: d }, mean };
}

/** A copy of a matrix. */
export function copyMatrix(M) {
  return { data: M.data.slice(), rows: M.rows, cols: M.cols };
}

/** Add a row vector to every row (new matrix). */
export function shift(M, v) {
  const data = M.data.slice();
  for (let i = 0; i < data.length; i++) data[i] += v[i % M.cols];
  return { data, rows: M.rows, cols: M.cols };
}

/** Squared Euclidean distances (N, K) from direct differences. */
export function sqdist(X, C) {
  const { rows: n, cols: d } = X, k = C.rows, x = X.data, c = C.data;
  const out = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let f = 0, a = i * d, b = j * d; f < d; f++) {
        const t = x[a + f] - c[b + f];
        s += t * t;
      }
      out[i * k + j] = s;
    }
  }
  return { data: out, rows: n, cols: k };
}

/**
 * Index of the nearest row of C for each row of X, ties to the lowest index:
 * argminRows(sqdist(X, C)) without the (N, K) matrix.
 */
export function nearest(X, C) {
  const { rows: n, cols: d } = X, k = C.rows, x = X.data, c = C.data, out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = Infinity;
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let f = 0, a = i * d, b = j * d; f < d; f++) {
        const t = x[a + f] - c[b + f];
        s += t * t;
      }
      if (s < best) best = s, out[i] = j;
    }
  }
  return out;
}

/** Index of the smallest entry per row; ties go to the lowest index. */
export function argminRows(D) {
  const { rows: n, cols: k, data } = D, out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let j = 1; j < k; j++) if (data[i * k + j] < data[i * k + best]) best = j;
    out[i] = best;
  }
  return out;
}

/** Index of the largest entry per row. */
export function argmaxRows(U) {
  const { rows: n, cols: k, data } = U, out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let j = 1; j < k; j++) if (data[i * k + j] > data[i * k + best]) best = j;
    out[i] = best;
  }
  return out;
}

/**
 * Memberships p_ij proportional to exp(-c_ij / T), and the sum of the rows'
 * soft minima -T log sum_j exp(-c_ij / T). A soft minimum equals
 * sum_j p_ij c_ij + T p_ij log p_ij at these memberships, the
 * entropy-regularized objective; it comes from the row maxima and sums that
 * the normalization computes.
 * @param {Matrix} C @param {number} temperature
 * @returns {{ U: Matrix, value: number }}
 */
export function softmin(C, temperature) {
  const { rows: n, cols: k, data: c } = C, out = new Float64Array(n * k);
  let value = 0;
  for (let i = 0; i < n; i++) {
    let max = -Infinity, sum = 0;
    for (let j = 0; j < k; j++) max = Math.max(max, out[i * k + j] = -c[i * k + j] / temperature);
    for (let j = 0; j < k; j++) sum += out[i * k + j] = Math.exp(out[i * k + j] - max);
    for (let j = 0; j < k; j++) out[i * k + j] /= sum;
    value += max + Math.log(sum);
  }
  return { U: { data: out, rows: n, cols: k }, value: -temperature * value };
}

/** Map every entry (new matrix). */
export function mapMatrix(M, fn) {
  // A plain loop: TypedArray callbacks (map, reduce, forEach) are several times slower.
  const out = new Float64Array(M.data.length);
  for (let i = 0; i < out.length; i++) out[i] = fn(M.data[i], i);
  return { data: out, rows: M.rows, cols: M.cols };
}

/** Means of X weighted by the columns of W; columns without mass keep V. */
export function weightedMean(X, W, V) {
  const { rows: n, cols: d } = X, k = W.cols, sums = new Float64Array(k * d), mass = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      const w = W.data[i * k + j];
      if (w === 0) continue;
      mass[j] += w;
      for (let f = 0; f < d; f++) sums[j * d + f] += w * X.data[i * d + f];
    }
  }
  return divide(sums, mass, V);
}

/** Per-label sums of X rows and label counts. */
export function labelSums(X, labels, k) {
  const d = X.cols, sums = new Float64Array(k * d), counts = new Float64Array(k);
  for (let i = 0; i < labels.length; i++) {
    const j = labels[i];
    counts[j]++;
    for (let f = 0; f < d; f++) sums[j * d + f] += X.data[i * d + f];
  }
  return { sums, counts };
}

/** Per-label means of X; empty labels keep V. */
export function labelMean(X, labels, V) {
  const { sums, counts } = labelSums(X, labels, V.rows);
  return divide(sums, counts, V);
}

function divide(sums, mass, V) {
  const d = V.cols, out = V.data.slice();
  for (let j = 0; j < V.rows; j++) {
    if (mass[j] > 0) for (let f = 0; f < d; f++) out[j * d + f] = sums[j * d + f] / mass[j];
  }
  return { data: out, rows: V.rows, cols: d };
}

/**
 * Greedy k-means++ seeding (Arthur & Vassilvitskii, 2007): each new seed is the
 * best, by the resulting sum of squared distances to the nearest seed, of
 * 2 + floor(ln k) candidates drawn in proportion to that squared distance.
 */
function kmeansPlusPlus(X, k, rand) {
  const { rows: n, cols: d } = X, chosen = [Math.floor(rand() * n)], trials = 2 + Math.floor(Math.log(k));
  const row = i => ({ data: X.data.subarray(i * d, (i + 1) * d), rows: 1, cols: d });
  let closest = sqdist(X, row(chosen[0])).data;
  while (chosen.length < k) {
    let total = 0;
    for (const v of closest) total += v;
    let best = -1, bestSum = Infinity, bestClosest = closest;
    for (let trial = 0; trial < trials; trial++) {
      let i = n - 1;
      if (total > 0) {
        let target = rand() * total;
        for (let j = 0; j < n; j++) if ((target -= closest[j]) < 0) { i = j; break; }
      } else i = Math.floor(rand() * n);
      const next = sqdist(X, row(i)).data;
      let sum = 0;
      for (let j = 0; j < n; j++) sum += next[j] = Math.min(closest[j], next[j]);
      if (sum < bestSum) best = i, bestSum = sum, bestClosest = next;
    }
    chosen.push(best);
    closest = bestClosest;
  }
  const V = new Float64Array(k * d);
  chosen.forEach((i, j) => V.set(row(i).data, j * d));
  return { data: V, rows: k, cols: d };
}

/**
 * The input of an iteration: the rows centered on their mean, and the RMS
 * norm of the centered rows, the scale of tol.
 * @typedef {{ X: Matrix, mean: Float64Array, radius: number }} Data
 */

/**
 * Validate and center X as the input of an iteration.
 * @param {MatrixLike} X @param {number} [nFeatures] the feature count of the data a run started on
 * @returns {Data}
 */
export function asData(X, nFeatures) {
  const M = matrix(X);
  if (nFeatures !== undefined && M.cols !== nFeatures) {
    throw new RangeError(`X must have ${nFeatures} features, like the data the run started on`);
  }
  const { X: Xc, mean } = center(M);
  let s = 0;
  for (const v of Xc.data) s += v * v;
  return { X: Xc, mean, radius: Math.sqrt(s / Xc.rows) };
}

/**
 * The data and the initial centers (centered like the data) of a clustering.
 * @param {MatrixLike} X @param {number} k @param {'k-means++' | MatrixLike} [init] @param {number} [seed]
 * @param {(X: MatrixLike, nFeatures?: number) => Data} [prepare]
 * @returns {{ data: Data, V: Matrix }}
 */
export function start(X, k, init = 'k-means++', seed, prepare = asData) {
  const data = prepare(X), Xc = data.X;
  checkInt(k, 'k', 1, Xc.rows);
  if (init === 'k-means++') {
    const V = kmeansPlusPlus(Xc, k, random(seed));
    // Start from the seeds' cell means: a center on a data point gives that
    // point full weight, and for large m stalls there (docs/algorithms.md).
    return { data, V: labelMean(Xc, nearest(Xc, V), V) };
  }
  if (typeof init === 'string') throw new RangeError("init must be 'k-means++' or a (k, nFeatures) matrix");
  const V = matrix(init, 'init');
  if (V.rows !== k || V.cols !== Xc.cols) throw new RangeError(`init must have shape (${k}, ${Xc.cols})`);
  return { data, V: shift(V, data.mean.map(v => -v)) };
}

/** An iteration limit >= 1, or Infinity for none (a run fed new data until it converges). */
export function checkMaxIter(value) {
  return value === Infinity ? value : checkInt(value, 'maxIter', 1);
}

/**
 * A step maps (data, prototypes, state, t) to { V, state, value }: new
 * prototypes, the state the method carries (null at first) and an objective
 * value or null. Steps read the data from their argument, never keep it.
 * @typedef {(data: Data, V: Matrix, state: any, t: number) => { V: Matrix, state: any, value: number | null }} Step
 */

/**
 * What the loop has reached: the data, prototypes, the step's state,
 * iterations, whether the stopping rule held at the last one, and the
 * objective history.
 * @typedef {{ data: Data, V: Matrix, state: any, nIter: number, converged: boolean, history: Float64Array }} Loop
 */

/**
 * The one loop: repeat step until the prototypes stop moving, or maxIter.
 *
 * The data are an input of every iteration: rows passed to the generator's
 * next(X) are prepared by prepare and used from the next iteration on. The
 * prototypes carry over (shifted to the new data's centering), and so does
 * the state that keep(state, data) returns; without keep the state starts
 * again, as caches of the old data must. The schedule (t) goes on.
 *
 * The run ends when no prototype coordinate moves more than tol times the
 * RMS radius of the data and no new data arrive (tol = 0: an exact fixed
 * point), or after maxIter iterations (Infinity: no limit); tol = null means
 * a fixed schedule, which converges by completing its maxIter iterations.
 * It returns view(loop), the method's Result. After every step it yields
 * { iteration, result }, where result() builds the same Result as if the run
 * had stopped there; it costs nothing unless called and stays valid as the
 * loop goes on, because steps never modify a state they have returned, and
 * it shares no arrays with the run, because views copy the state they put in
 * a Result.
 * @param {Data} data @param {Matrix} V @param {Step} step
 * @param {{ maxIter: number, tol: number | null, view: (loop: Loop) => Result,
 *   prepare?: (X: MatrixLike, nFeatures?: number) => Data, keep?: ((state: any, data: Data) => any) | null }} options
 * @returns {Generator<Progress, Result, MatrixLike | undefined>}
 */
export function* iterate(data, V, step, { maxIter, tol, view, prepare = asData, keep = null }) {
  const history = [];
  let state = null;
  for (let t = 1; ; t++) {
    const previous = V, next = step(data, V, state, t - 1);
    ({ V, state } = next);
    if (next.value !== null) history.push(next.value);
    let move = 0;
    for (let i = 0; i < V.data.length; i++) move = Math.max(move, Math.abs(V.data[i] - previous.data[i]));
    const converged = tol === null ? t === maxIter : move <= tol * data.radius;
    const loop = { data, V, state, nIter: t, converged }, length = history.length;
    const result = () => view({ ...loop, history: Float64Array.from(history.slice(0, length)) });
    const rows = yield { iteration: t, result };
    if (t === maxIter) return result();
    if (rows !== undefined && rows !== null) {
      const fresh = prepare(rows, V.cols);
      V = shift(V, data.mean.map((m, f) => m - fresh.mean[f]));
      state = keep ? keep(state, fresh) : null;
      data = fresh;
    } else if (converged) return result();
  }
}

/**
 * The Result of a loop with the labels, memberships and embedding of its method.
 * @param {Loop} loop @param {Int32Array} labels @param {Matrix | null} [membership] @param {Matrix | null} [embedding]
 * @returns {Result}
 */
export function fitted({ data, V, nIter, converged, history }, labels, membership = null, embedding = null) {
  return { centers: shift(V, data.mean), labels, membership, nIter, converged, history, embedding };
}

/** The standard update: the means of the data weighted by the memberships. */
export function means(data, U, V) {
  return weightedMean(data.X, U, V);
}

/**
 * The standard step: D = ||x - v||^2, { U, value } = assign(data, D), V = update(data, U, V).
 * assign gives the memberships U for these distances, the step's state
 * (which the method's view reads), and the method's objective at them, one
 * history value (null for a method without one). The memberships minimize
 * the objective for the given prototypes, and the quantities that compute
 * them give its value, so it costs almost nothing. update defaults to the
 * means weighted by U.
 * @returns {Step}
 */
export function lloyd({ assign, update = means }) {
  return (data, V) => {
    const { U, value } = assign(data, sqdist(data.X, V));
    return { V: update(data, U, V), state: U, value };
  };
}

/**
 * Run a step generator to completion.
 * @template T
 * @param {Generator<Progress, T>} steps
 * @returns {T}
 */
export function run(steps) {
  for (;;) {
    const { value, done } = steps.next();
    if (done) return value;
  }
}

/**
 * Run a step generator cooperatively, yielding to the event loop about every
 * budgetMs and stopping when signal aborts.
 * @template T
 * @param {Generator<Progress, T>} steps
 * @param {{ signal?: AbortSignal, onProgress?: (p: Progress) => void, budgetMs?: number }} [options]
 * @returns {Promise<T>}
 */
export async function runAsync(steps, { signal, onProgress, budgetMs = 12 } = {}) {
  let deadline = performance.now() + budgetMs;
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { value, done } = steps.next();
      if (done) return value;
      onProgress?.(/** @type {Progress} */ (value));
      if (performance.now() >= deadline) {
        await new Promise(resolve => setTimeout(resolve, 0));
        deadline = performance.now() + budgetMs;
      }
    }
  } finally {
    steps.return(undefined);
  }
}
