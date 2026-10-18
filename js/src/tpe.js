/**
 * Tree-structured Parzen Estimator for small mixed search spaces.
 * Multivariate TPE (Bergstra et al., 2011; Falkner et al., 2018): trials are
 * split into the best `gamma` fraction and the rest, each side is a Parzen
 * mixture whose components keep one trial's dimensions together, and the
 * candidate maximizing l(x) / g(x) is proposed. Numeric dimensions live on [0, 1].
 */
import { checkInt, checkNumber, named, random, rejectUnknown } from './core.js';

/** @typedef {{ readonly kind: 'numeric', readonly low: number, readonly high: number, readonly log: boolean, readonly integer: boolean }} NumericDim */
/** @typedef {{ readonly kind: 'choice', readonly options: readonly unknown[] }} ChoiceDim */
/** @typedef {Record<string, NumericDim | ChoiceDim>} Space */
/** @typedef {{ bestParams: Record<string, unknown>, bestValue: number, params: Record<string, unknown>[], values: Float64Array }} TPEResult */
/** @typedef {{ seed?: number, nStartup?: number, nCandidates?: number, gamma?: number }} TPEOptions */

/** Real values in [low, high]. @param {number} low @param {number} high @returns {NumericDim} */
export const uniform = (low, high) => numeric(low, high, false, false);
/** Positive real values in [low, high], searched on a log scale. @param {number} low @param {number} high @returns {NumericDim} */
export const loguniform = (low, high) => numeric(low, high, true, false);
/**
 * Integers in [low, high], optionally on a log scale.
 * @param {number} low @param {number} high @param {{ log?: boolean }} [options] @returns {NumericDim}
 */
export const integer = (low, high, options) => {
  const { log = false, ...unknown } = named(options);
  rejectUnknown(unknown);
  checkInt(low, 'low', Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);
  return numeric(low, checkInt(high, 'high', low, Number.MAX_SAFE_INTEGER), log, true);
};
/** One of the given options (compared with ===). @param {...unknown} options @returns {ChoiceDim} */
export function choice(...options) {
  if (!options.length) throw new RangeError('choice needs at least one option');
  return Object.freeze({ kind: 'choice', options: Object.freeze(options) });
}

/** A frozen numeric dimension, as Python's are. @returns {NumericDim} */
function numeric(low, high, log, integer) {
  checkNumber(low, 'low', log ? 0 : -Infinity, log);
  checkNumber(high, 'high', low, !integer);
  if (!Number.isFinite(high - low)) throw new RangeError('high - low must be finite');
  const dim = { kind: /** @type {'numeric'} */ ('numeric'), low, high, log, integer };
  // Values are searched as positions between the ends, in logarithms on a
  // log scale, where ends close together can coincide.
  const [lo, hi] = bounds(dim);
  if (!(lo < hi)) throw new RangeError(`[${low}, ${high}] is too narrow to search on a log scale`);
  return Object.freeze(dim);
}

/**
 * A checked copy of a dimension of a space, as its factory makes it, so that
 * a dimension written by hand or changed later cannot change a search.
 * @param {any} dim @param {string} name @returns {NumericDim | ChoiceDim}
 */
function dimension(dim, name) {
  if (dim?.kind === 'choice' && Array.isArray(dim.options)) return choice(...dim.options);
  if (dim?.kind === 'numeric') {
    return dim.integer ? integer(dim.low, dim.high, { log: Boolean(dim.log) }) : numeric(dim.low, dim.high, Boolean(dim.log), false);
  }
  throw new RangeError(`space.${name} must come from uniform/loguniform/integer/choice`);
}

function bounds(dim) {
  const [lo, hi] = dim.integer ? [dim.low - 0.5, dim.high + 0.5] : [dim.low, dim.high];
  return dim.log ? [Math.log(lo), Math.log(hi)] : [lo, hi];
}
function encode(dim, x) {
  if (dim.kind === 'choice') {
    // SameValueZero, as includes compares: indexOf never finds NaN, so a NaN
    // option could not be told back as ask gave it (Python finds it by identity).
    const i = dim.options.findIndex(o => o === x || (o !== o && x !== x));
    if (i < 0) throw new RangeError(`${x} is not an option`);
    return i;
  }
  if (typeof x !== 'number' || !(x >= dim.low && x <= dim.high) || (dim.integer && !Number.isInteger(x))) {
    throw new RangeError(`${x} is not ${dim.integer ? 'an integer' : 'a number'} in [${dim.low}, ${dim.high}]`);
  }
  const [lo, hi] = bounds(dim);
  return ((dim.log ? Math.log(x) : x) - lo) / (hi - lo);
}
function decode(dim, u) {
  if (dim.kind === 'choice') return dim.options[u];
  const [lo, hi] = bounds(dim);
  let x = lo + u * (hi - lo);
  if (dim.log) x = Math.exp(x);
  if (dim.integer) x = Math.round(x);
  return Math.min(Math.max(x, dim.low), dim.high);
}

/** Ask/tell TPE minimizer. */
export class TPE {
  #space; #dims; #rand; #encoded = []; #params = []; #values = []; #nStartup; #nCandidates; #gamma;

  /**
   * @param {Space} space
   * @param {TPEOptions} [options]
   */
  constructor(space, options) {
    const { seed, nStartup = 10, nCandidates = 24, gamma = 0.15, ...unknown } = named(options);
    rejectUnknown(unknown);
    if (!space || typeof space !== 'object' || !Object.keys(space).length) throw new RangeError('space must be a non-empty object');
    this.#space = Object.fromEntries(Object.entries(space).map(([name, dim]) => [name, dimension(dim, name)]));
    this.#dims = Object.values(this.#space);
    this.#rand = random(seed);
    this.#nStartup = checkInt(nStartup, 'nStartup', 2);
    this.#nCandidates = checkInt(nCandidates, 'nCandidates', 1);
    this.#gamma = checkNumber(gamma, 'gamma', 0, true);
    if (gamma >= 1) throw new RangeError('gamma must be < 1');
  }

  // The settings are read-only: a value set later would skip the checks above.
  /** Random trials before the model is used. @returns {number} */
  get nStartup() { return this.#nStartup; }
  /** Candidates drawn from l(x) per proposal. @returns {number} */
  get nCandidates() { return this.#nCandidates; }
  /** Fraction of trials treated as good. @returns {number} */
  get gamma() { return this.#gamma; }

  /** Propose parameters to evaluate next. @returns {Record<string, unknown>} */
  ask() {
    const u = this.#values.length < this.nStartup
      ? this.#dims.map(d => (d.kind === 'choice' ? Math.floor(this.#rand() * d.options.length) : this.#rand()))
      : this.#propose();
    return Object.fromEntries(Object.entries(this.#space).map(([name, dim], j) => [name, decode(dim, u[j])]));
  }

  /**
   * Record the objective value of params (lower is better).
   * @param {Record<string, unknown>} params @param {number} value
   */
  tell(params, value) {
    const names = Object.keys(this.#space);
    if (Object.keys(params).length !== names.length || !names.every(k => Object.hasOwn(params, k))) throw new RangeError('params must have exactly the keys of the space');
    checkNumber(value, 'value');
    this.#encoded.push(names.map(k => encode(this.#space[k], params[k])));
    this.#params.push({ ...params });
    this.#values.push(value);
  }

  /** @returns {TPEResult} */
  result() {
    if (!this.#values.length) throw new RangeError('no trials have been told');
    const best = this.#values.reduce((b, v, i, all) => (v < all[b] ? i : b), 0);
    return { bestParams: { ...this.#params[best] }, bestValue: this.#values[best], params: this.#params.map(p => ({ ...p })), values: Float64Array.from(this.#values) };
  }

  #propose() {
    const n = this.#values.length, order = Array.from(this.#values.keys()).sort((a, b) => this.#values[a] - this.#values[b] || a - b);
    const nGood = Math.min(n - 1, Math.max(1, Math.ceil(this.gamma * n)));
    const good = parzen(order.slice(0, nGood).map(i => this.#encoded[i]), this.#dims);
    const bad = parzen(order.slice(nGood).map(i => this.#encoded[i]), this.#dims);
    let best = null, bestScore = -Infinity;
    for (let c = 0; c < this.nCandidates; c++) {
      const z = good.sample(this.#rand), score = good.logpdf(z) - bad.logpdf(z);
      if (best === null || score > bestScore) best = z, bestScore = score;
    }
    return best;
  }
}

/**
 * Minimize f(params) over space with TPE; f may return a promise. Negate f to maximize.
 * @param {(params: Record<string, unknown>) => number | Promise<number>} f
 * @param {Space} space
 * @param {TPEOptions & { nTrials?: number }} [options] the number of trials and the options of TPE
 * @returns {Promise<TPEResult>}
 */
export async function minimize(f, space, options) {
  const { nTrials = 100, ...tpeOptions } = named(options);
  checkInt(nTrials, 'nTrials', 1);
  const tpe = new TPE(space, tpeOptions);
  for (let t = 0; t < nTrials; t++) {
    const params = tpe.ask();
    tpe.tell(params, await f(params));
  }
  return tpe.result();
}

/** Equal-weight mixture: one component per observation plus a wide prior. */
function parzen(obs, dims) {
  const m = obs.length, comps = m + 1, logWeight = -Math.log(comps);
  const columns = dims.map((dim, j) => {
    if (dim.kind === 'choice') {
      const k = dim.options.length, eps = 1 / comps;
      const prob = (c, x) => (c === m ? 1 / k : eps / k + (obs[c][j] === x ? 1 - eps : 0));
      return { choice: true, k, prob };
    }
    // Truncated normals on [0, 1]; width = larger gap to neighboring centers.
    const mu = [...obs.map(o => o[j]), 0.5], order = mu.map((_, i) => i).sort((a, b) => mu[a] - mu[b]);
    const sigma = new Array(comps), floor = Math.max(0.03, comps ** -2);
    order.forEach((c, r) => {
      const left = mu[c] - (r > 0 ? mu[order[r - 1]] : 0), right = (r < comps - 1 ? mu[order[r + 1]] : 1) - mu[c];
      sigma[c] = Math.min(Math.max(left, right, floor), 1);
    });
    sigma[m] = 1;
    const lo = mu.map((u, c) => ndtr(-u / sigma[c])), hi = mu.map((u, c) => ndtr((1 - u) / sigma[c]));
    const logNorm = sigma.map((s, c) => Math.log(s * Math.sqrt(2 * Math.PI) * (hi[c] - lo[c])));
    return { choice: false, mu, sigma, lo, hi, logNorm };
  });
  return {
    sample(rand) {
      const c = Math.floor(rand() * comps);
      return columns.map(col => {
        if (col.choice) {
          let r = rand();
          for (let x = 0; x < col.k; x++) if ((r -= col.prob(c, x)) < 0) return x;
          return col.k - 1;
        }
        const z = col.mu[c] + col.sigma[c] * ndtri(col.lo[c] + rand() * (col.hi[c] - col.lo[c]));
        return Math.min(Math.max(z, 0), 1);
      });
    },
    logpdf(z) {
      const terms = Array.from({ length: comps }, (_, c) => logWeight + columns.reduce((s, col, j) => {
        if (col.choice) return s + Math.log(col.prob(c, z[j]));
        const t = (z[j] - col.mu[c]) / col.sigma[c];
        return s - 0.5 * t * t - col.logNorm[c];
      }, 0));
      const max = terms.reduce((a, b) => Math.max(a, b), -Infinity);
      return max + Math.log(terms.reduce((s, t) => s + Math.exp(t - max), 0));
    },
  };
}

/** Standard normal CDF via erfc (relative error < 1.2e-7). */
function ndtr(x) {
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
  const erfc = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
    + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - erfc / 2 : erfc / 2;
}

/** Inverse standard normal CDF (Acklam, relative error < 1.2e-9). */
function ndtri(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const tail = q => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (p < 0.02425) return tail(Math.sqrt(-2 * Math.log(p)));
  if (p > 0.97575) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
