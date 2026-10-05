/** External agreement (ARI, AMI) and neighborhood preservation (trustworthiness, continuity). */
import { checkInt, matrix } from './core.js';

const AVERAGES = {
  arithmetic: (a, b) => (a + b) / 2,
  geometric: (a, b) => Math.sqrt(a * b),
  min: Math.min,
  max: Math.max,
};

/**
 * Adjusted Rand index (Hubert & Arabie, 1985).
 * @param {ArrayLike<number | string>} labelsTrue
 * @param {ArrayLike<number | string>} labelsPred
 */
export function ari(labelsTrue, labelsPred) {
  const { n, a, b, cells } = contingency(labelsTrue, labelsPred);
  /** @type {(v: number[]) => bigint} */
  const square = v => v.reduce((s, x) => s + BigInt(x) * BigInt(x), 0n);
  const N = BigInt(n), sa = square(a), sb = square(b), s = square([...cells.values()]);
  const tp = s - N, fp = sb - s, fn = sa - s, tn = N * N - sa - sb + s;
  if (fp === 0n && fn === 0n) return 1;
  return Number(2n * (tp * tn - fn * fp)) / Number((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
}

/**
 * Adjusted mutual information (Vinh, Epps & Bailey, 2010).
 * @param {ArrayLike<number | string>} labelsTrue
 * @param {ArrayLike<number | string>} labelsPred
 * @param {{ average?: 'arithmetic' | 'geometric' | 'min' | 'max' }} [options]
 */
export function ami(labelsTrue, labelsPred, { average = 'arithmetic' } = {}) {
  if (!(average in AVERAGES)) throw new RangeError(`average must be one of ${Object.keys(AVERAGES).join(', ')}`);
  const { n, a, b, cells } = contingency(labelsTrue, labelsPred);
  if (a.length === b.length && a.length <= 1) return 1;
  let mi = 0;
  for (const [key, nij] of cells) {
    const i = Math.floor(key / b.length), j = key % b.length;
    mi += (nij / n) * (Math.log(nij) + Math.log(n) - Math.log(a[i]) - Math.log(b[j]));
  }
  const emi = expectedMutualInformation(n, a, b);
  let denominator = AVERAGES[average](entropy(a, n), entropy(b, n)) - emi;
  denominator = denominator < 0 ? Math.min(denominator, -Number.EPSILON) : Math.max(denominator, Number.EPSILON);
  return (mi - emi) / denominator;
}

/**
 * Trustworthiness of embedding Y of X (Venna & Kaski, 2001): penalizes the k
 * nearest neighbors in Y that are not among those in X, by their rank in X.
 * Distance ties are ordered by sample index.
 * @param {import('./core.js').MatrixLike} X
 * @param {import('./core.js').MatrixLike} Y
 */
export function trustworthiness(X, Y, k = 5) {
  X = matrix(X);
  Y = matrix(Y, 'Y');
  const n = X.rows;
  if (Y.rows !== n) throw new RangeError('X and Y must have the same number of rows');
  checkInt(k, 'k', 1);
  if (!(2 * k < n)) throw new RangeError('k must satisfy 1 <= k < n / 2');
  const dx = new Float64Array(n), dy = new Float64Array(n), nearest = new Int32Array(k);
  let penalty = 0;
  for (let i = 0; i < n; i++) {
    distances(X, i, dx);
    distances(Y, i, dy);
    // k smallest of dy by (distance, index), kept sorted by insertion.
    let size = 0;
    for (let j = 0; j < n; j++) {
      if (size === k && dy[j] >= dy[nearest[k - 1]]) continue;
      let p = Math.min(size, k - 1);
      while (p > 0 && dy[nearest[p - 1]] > dy[j]) nearest[p] = nearest[p - 1], p--;
      nearest[p] = j;
      size = Math.min(size + 1, k);
    }
    for (let q = 0; q < k; q++) {
      const j = nearest[q], target = dx[j];
      let rank = 1;
      for (let l = 0; l < n; l++) if (dx[l] < target || (dx[l] === target && l < j)) rank++;
      penalty += Math.max(rank - k, 0);
    }
  }
  return 1 - (2 * penalty) / (n * k * (2 * n - 3 * k - 1));
}

/**
 * Continuity of embedding Y of X: trustworthiness with the roles swapped.
 * @param {import('./core.js').MatrixLike} X
 * @param {import('./core.js').MatrixLike} Y
 */
export function continuity(X, Y, k = 5) {
  return trustworthiness(Y, X, k);
}

function distances(M, i, out) {
  const { rows: n, cols: d, data } = M;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let f = 0; f < d; f++) s += (data[i * d + f] - data[j * d + f]) ** 2;
    out[j] = s;
  }
  out[i] = Infinity;
}

function contingency(labelsTrue, labelsPred) {
  const n = labelsTrue.length;
  if (n !== labelsPred.length) throw new RangeError('labels must have equal length');
  const encode = labels => {
    const ids = new Map(), codes = new Int32Array(n), counts = [];
    for (let i = 0; i < n; i++) {
      let id = ids.get(labels[i]);
      if (id === undefined) ids.set(labels[i], (id = counts.length)), counts.push(0);
      codes[i] = id;
      counts[id]++;
    }
    return { codes, counts };
  };
  const u = encode(labelsTrue), v = encode(labelsPred), kb = v.counts.length, cells = new Map();
  for (let i = 0; i < n; i++) {
    const key = u.codes[i] * kb + v.codes[i];
    cells.set(key, (cells.get(key) ?? 0) + 1);
  }
  return { n, a: u.counts, b: v.counts, cells };
}

function entropy(counts, n) {
  return -counts.reduce((s, c) => s + (c / n) * Math.log(c / n), 0);
}

/** E[MI] under the hypergeometric model, grouped by distinct marginal sizes. */
function expectedMutualInformation(n, a, b) {
  const histogram = counts => counts.reduce((m, c) => m.set(c, (m.get(c) ?? 0) + 1), new Map());
  const ha = histogram(a), hb = histogram(b), lgN = lgamma(n + 1);
  let emi = 0;
  for (const [x, cx] of ha) for (const [y, cy] of hb) {
    const base = lgamma(x + 1) + lgamma(y + 1) + lgamma(n - x + 1) + lgamma(n - y + 1) - lgN;
    let sum = 0;
    for (let nij = Math.max(1, x + y - n); nij <= Math.min(x, y); nij++) {
      const logP = base - lgamma(nij + 1) - lgamma(x - nij + 1) - lgamma(y - nij + 1) - lgamma(n - x - y + nij + 1);
      sum += (nij / n) * Math.log((n * nij) / (x * y)) * Math.exp(logP);
    }
    emi += cx * cy * sum;
  }
  return emi;
}

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** log Gamma(x) for x > 0 (Lanczos, g = 7). */
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let sum = LANCZOS[0];
  for (let i = 1; i < 9; i++) sum += LANCZOS[i] / (x + i);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}
