/** Adjusted clustering agreement scores for browsers and Node.
 * Scikit-learn-compatible definitions; see NOTICE-EXTERNAL-METRICS.txt and
 * LICENSE-SCIKIT-LEARN.txt for compatibility-expression attribution.
 * No imports, process-global data caches, randomization, or input mutation.
 */
const MAX_AMI_SAMPLES = 2 ** 26;
const DEFAULT_EXPECTED_TERMS = 10_000_000;
const METHODS = new Set(['arithmetic', 'geometric', 'min', 'max']);

/** A requested AMI is numerically singular or exceeds a stated work/domain limit. */
export class ExternalMetricDomainError extends RangeError {
  constructor(message, code) { super(message); this.name = 'ExternalMetricDomainError'; this.code = code; }
}

function optionsForAMI(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('options must be an object');
  const averageMethod = options.averageMethod ?? 'arithmetic';
  if (!METHODS.has(averageMethod)) throw new RangeError('averageMethod must be arithmetic, geometric, min, or max');
  const maxExpectedTerms = options.maxExpectedTerms ?? DEFAULT_EXPECTED_TERMS;
  if (!Number.isSafeInteger(maxExpectedTerms) || maxExpectedTerms < 1) throw new RangeError('maxExpectedTerms must be a positive safe integer');
  return { averageMethod, maxExpectedTerms };
}

function validateArray(labels, name) {
  if (!Array.isArray(labels) && !(ArrayBuffer.isView(labels) && !(labels instanceof DataView))) throw new TypeError(`${name} must be an Array or numeric TypedArray`);
  if (!Number.isSafeInteger(labels.length) || labels.length > 0xffffffff) throw new RangeError(`${name} exceeds the supported length`);
}

function encode(labels, name) {
  const n = labels.length, codes = new Uint32Array(n), counts = [], ids = new Map();
  let kind;
  for (let i = 0; i < n; ++i) {
    const label = labels[i], type = typeof label;
    if (type !== 'string' && (type !== 'number' || !Number.isSafeInteger(label))) throw new TypeError(`${name}[${i}] must be a string or safe integer`);
    if (kind === undefined) kind = type;
    else if (kind !== type) throw new TypeError(`${name} must not mix string and integer labels`);
    let id = ids.get(label);
    if (id === undefined) { id = counts.length; ids.set(label, id); counts.push(0); }
    codes[i] = id; ++counts[id];
  }
  return { codes, counts };
}

function contingency(labelsTrue, labelsPred, forAMI = false) {
  validateArray(labelsTrue, 'labelsTrue'); validateArray(labelsPred, 'labelsPred');
  const n = labelsTrue.length;
  if (n !== labelsPred.length) throw new RangeError('labelsTrue and labelsPred must have equal lengths');
  if (forAMI && n > MAX_AMI_SAMPLES) throw new ExternalMetricDomainError(`AMI supports at most ${MAX_AMI_SAMPLES} samples`, 'AMI_SAMPLE_LIMIT');
  const { codes: u, counts: a } = encode(labelsTrue, 'labelsTrue');
  const { codes: v, counts: b } = encode(labelsPred, 'labelsPred');
  const cells = [], rows = [], cols = [], kb = b.length, size = a.length * kb;
  if (size <= Math.max(256, 4 * n) && size <= 1_000_000) {
    const table = new Uint32Array(size), occupied = [];
    for (let i = 0; i < n; ++i) {
      const key = u[i] * kb + v[i];
      if (table[key]++ === 0) occupied.push(key);
    }
    for (const key of occupied) { cells.push(table[key]); rows.push(Math.floor(key / kb)); cols.push(key % kb); }
  } else if (Number.isSafeInteger(size)) {
    const table = new Map();
    for (let i = 0; i < n; ++i) { const key = u[i] * kb + v[i]; table.set(key, (table.get(key) ?? 0) + 1); }
    for (const [key, count] of table) { cells.push(count); rows.push(Math.floor(key / kb)); cols.push(key % kb); }
  } else {
    // Never collapse distinct pairs by rounding an unsafe packed integer key.
    const table = new Map();
    for (let i = 0; i < n; ++i) {
      let row = table.get(u[i]);
      if (row === undefined) { row = new Map(); table.set(u[i], row); }
      row.set(v[i], (row.get(v[i]) ?? 0) + 1);
    }
    for (const [r, row] of table) for (const [c, count] of row) { cells.push(count); rows.push(r); cols.push(c); }
  }
  return { n, a, b, cells, rows, cols };
}

function squareSum(counts, n) {
  if (n <= 94_906_265) {
    let value = 0;
    for (const c of counts) value += c * c;
    return BigInt(value);
  }
  let value = 0n;
  for (const c of counts) { const x = BigInt(c); value += x * x; }
  return value;
}

function ariFromTable({ n, a, b, cells }) {
  const nn = BigInt(n), q = squareSum(cells, n), sa = squareSum(a, n), sb = squareSum(b, n);
  const tp = q - nn, fp = sb - q, fn = sa - q;
  if (fp === 0n && fn === 0n) return 1;
  const tn = nn * nn - sa - sb + q;
  const numerator = 2n * (tp * tn - fn * fp);
  const denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn);
  return Number(numerator) / Number(denominator);
}

// Neumaier summation retains small corrections even when a new term is larger.
function add(sum, value) {
  const next = sum[0] + value;
  sum[1] += Math.abs(sum[0]) >= Math.abs(value) ? (sum[0] - next) + value : (value - next) + sum[0];
  sum[0] = next;
}
function total(sum) { return sum[0] + sum[1]; }
function entropy(counts, n) {
  const sum = [0, 0];
  // Canonical size order makes entropy and the orientation decision invariant
  // to label order, even when different histograms have equal entropy.
  for (const [count, frequency] of [...frequencies(counts)].sort((x, y) => x[0] - y[0])) add(sum, frequency * (count / n) * Math.log1p((n - count) / count));
  return total(sum);
}
function compareMargins(a, b) {
  const aa = [...frequencies(a)].sort((x, y) => x[0] - y[0]);
  const bb = [...frequencies(b)].sort((x, y) => x[0] - y[0]);
  for (let i = 0; i < Math.min(aa.length, bb.length); ++i) {
    const difference = aa[i][0] - bb[i][0] || aa[i][1] - bb[i][1];
    if (difference !== 0) return difference;
  }
  return aa.length - bb.length;
}
function observedConditional({ n, a, b, cells, rows, cols }, lowIsA) {
  const sum = [0, 0], other = lowIsA ? b : a, indices = lowIsA ? cols : rows;
  for (let i = 0; i < cells.length; ++i) {
    const x = cells[i];
    add(sum, (x / n) * Math.log1p((other[indices[i]] - x) / x));
  }
  return total(sum);
}
function frequencies(counts) {
  const result = new Map();
  for (const count of counts) result.set(count, (result.get(count) ?? 0) + 1);
  return result;
}
function marginPairs(n, a, b, maxExpectedTerms) {
  const aa = frequencies(a), bb = frequencies(b), pairs = new Map();
  let terms = 0;
  for (const [x, nx] of aa) for (const [y, ny] of bb) {
    // y is the conditioning partition's margin. Preserve orientation.
    if (y === 1) continue; // X is 0 or 1, and X log(1/X) is exactly zero.
    const key = x * (n + 1) + y;
    const known = pairs.get(key);
    if (known !== undefined) known[2] += nx * ny;
    else {
      terms += Math.min(x, y) - Math.max(0, x + y - n) + 1;
      if (terms > maxExpectedTerms) throw new ExternalMetricDomainError(`AMI expectation exceeds maxExpectedTerms=${maxExpectedTerms}; raise the limit deliberately or use a worker`, 'AMI_WORK_LIMIT');
      pairs.set(key, [x, y, nx * ny]);
    }
  }
  return pairs.values();
}
function pairExpectation(n, first, other) {
  const a = Math.min(first, other), b = Math.max(first, other);
  const low = Math.max(0, a + b - n), high = a;
  const mode = Math.min(high, Math.max(low, Math.floor(((a + 1) * (b + 1)) / (n + 2))));
  const weights = [1, 0], information = [mode === 0 ? 0 : (mode / n) * Math.log1p((other - mode) / mode), 0];
  let weight = 1;
  for (let x = mode + 1; x <= high; ++x) {
    weight *= ((a - x + 1) / x) * ((b - x + 1) / (n - a - b + x));
    add(weights, weight);
    if (weight !== 0) add(information, weight * (x / n) * Math.log1p((other - x) / x));
  }
  weight = 1;
  for (let x = mode - 1; x >= low; --x) {
    weight *= ((x + 1) / (a - x)) * ((n - a - b + x + 1) / (b - x));
    add(weights, weight);
    if (x !== 0 && weight !== 0) add(information, weight * (x / n) * Math.log1p((other - x) / x));
  }
  return total(information) / total(weights);
}
function expectedInfo(n, a, b, maxExpectedTerms) {
  const sum = [0, 0];
  for (const [x, y, multiplicity] of marginPairs(n, a, b, maxExpectedTerms)) add(sum, multiplicity * pairExpectation(n, x, y));
  return total(sum);
}
function amiFromTable(table, { averageMethod, maxExpectedTerms }) {
  const { n, a, b, cells } = table, ka = a.length, kb = b.length;
  if (n > MAX_AMI_SAMPLES) throw new ExternalMetricDomainError(`AMI supports at most ${MAX_AMI_SAMPLES} samples`, 'AMI_SAMPLE_LIMIT');
  if ((ka === 0 && kb === 0) || (ka === 1 && kb === 1)) return 1;
  if (ka === 1 || kb === 1) return 0;
  if (cells.length === ka && ka === kb) return 1;
  if (ka === n || kb === n) {
    if (averageMethod === 'min') throw new ExternalMetricDomainError('AMI with min normalization is undefined for singleton versus nonidentical partitions (0/0)', 'AMI_SINGULAR_NORMALIZATION');
    // MI == EMI exactly here. sklearn can return roundoff around zero.
    return 0;
  }
  const ha = entropy(a, n), hb = entropy(b, n), lowIsA = ha < hb || (ha === hb && compareMargins(a, b) <= 0);
  const low = Math.min(ha, hb), high = Math.max(ha, hb);
  // Compute H(low) - EMI and H(low) - MI directly as conditional entropies.
  // All expectation terms are nonnegative. This avoids subtracting entropies
  // close to log(n), the source of large high-K AMI rounding artifacts.
  const expected = expectedInfo(n, lowIsA ? a : b, lowIsA ? b : a, maxExpectedTerms);
  const observed = observedConditional(table, lowIsA);
  let gap;
  if (averageMethod === 'arithmetic') gap = (high - low) / 2;
  else if (averageMethod === 'geometric') gap = low * (high - low) / (Math.sqrt(low * high) + low);
  else if (averageMethod === 'min') gap = 0;
  else gap = high - low;
  const denominator = expected + gap;
  if (!(denominator > 0) || !Number.isFinite(denominator)) throw new ExternalMetricDomainError('AMI normalization is not representable as a positive finite binary64 value', 'AMI_ILL_CONDITIONED');
  return (expected - observed) / denominator;
}

/** ARI, with exact integer combinatorics and one final binary64 quotient. */
export function adjustedRandScore(labelsTrue, labelsPred) {
  return ariFromTable(contingency(labelsTrue, labelsPred));
}
/** AMI under the permutation model; arithmetic normalization is the default. */
export function adjustedMutualInfoScore(labelsTrue, labelsPred, options = {}) {
  const config = optionsForAMI(options);
  return amiFromTable(contingency(labelsTrue, labelsPred, true), config);
}
/** Compute {ari, ami} with a single shared encoding and contingency table. */
export function adjustedScores(labelsTrue, labelsPred, options = {}) {
  const config = optionsForAMI(options), table = contingency(labelsTrue, labelsPred, true);
  return { ari: ariFromTable(table), ami: amiFromTable(table, config) };
}
export { adjustedRandScore as adjusted_rand_score, adjustedMutualInfoScore as adjusted_mutual_info_score, adjustedScores as adjusted_scores };
