/** Straightforward, allocation-reusing Float64 FCM reference.
 * Membership update is O(NK), not a pairwise cluster-ratio calculation.
 * Both this reference and production use the m=2 multiplication/division path.
 * Keeps ordinary weight and distance matrices so each algorithm stage remains
 * independently readable. No intentionally slow per-element JS array methods.
 */
import { normalizeInput, positiveInteger, finiteNumber, assertFinite, translatedData, restoreCenters, labelsFromMembership } from '../../consumer/node_modules/ubukit-js/src/core.js';
const TINY = 2.2250738585072014e-308;
function power(value, m) { return m === 2 ? value * value : value ** m; }
function distance(x, xo, c, co, d) {
  let sum = 0;
  for (let f = 0; f < d; f++) { const delta = x[xo + f] - c[co + f]; sum += delta * delta; }
  if (!Number.isFinite(sum)) throw new RangeError('squared distance overflow; rescale input');
  return sum;
}
function objectiveTerm(u, d2, m) {
  if (u === 0 || d2 === 0) return 0;
  const w = power(u, m);
  return w < TINY ? Math.exp(m * Math.log(u) + Math.log(d2)) : w * d2;
}
export function referenceFcmLinear(input, options = {}) {
  const original = normalizeInput(input), n = original.nSamples, d = original.nFeatures;
  const k = positiveInteger(options.nClusters, 'nClusters');
  const maxIterations = positiveInteger(options.maxIterations ?? 100, 'maxIterations');
  const tolerance = finiteNumber(options.tolerance ?? 1e-5, 'tolerance', 0), m = options.m ?? 2;
  if (!Number.isFinite(m) || m <= 1) throw new RangeError('m must be finite and >1');
  if (!ArrayBuffer.isView(options.initMembership) || options.initMembership.length !== n * k) throw new RangeError('explicit N*K initMembership required by benchmark reference');
  let u = Float64Array.from(options.initMembership), next = new Float64Array(n * k);
  assertFinite(u, 'initMembership');
  for (let i = 0; i < n; i++) {
    const off = i * k; let maximum = 0, sum = 0;
    for (let c = 0; c < k; c++) { if (u[off + c] < 0) throw new RangeError('negative membership'); maximum = Math.max(maximum, u[off + c]); }
    if (!maximum) throw new RangeError('membership row has zero mass');
    for (let c = 0; c < k; c++) { u[off + c] /= maximum; sum += u[off + c]; }
    for (let c = 0; c < k; c++) u[off + c] /= sum;
  }
  const shifted = translatedData(original), x = shifted.data;
  const centers = new Float64Array(k * d), numerator = new Float64Array(k * d), sums = new Float64Array(k);
  const weights = new Float64Array(n * k), d2 = new Float64Array(n * k);
  for (let f = 0; f < d; f++) { let mean = 0; for (let i = 0; i < n; i++) mean += x[i * d + f] / n; for (let c = 0; c < k; c++) centers[c * d + f] = mean; }
  const history = []; let iterations = 0, delta = Infinity, converged = false;
  for (iterations = 1; iterations <= maxIterations; iterations++) {
    // U^m: same specialized m=2 path as production.
    if (m === 2) for (let z = 0; z < u.length; z++) weights[z] = u[z] * u[z];
    else for (let z = 0; z < u.length; z++) weights[z] = u[z] ** m;
    sums.fill(0); numerator.fill(0);
    for (let i = 0; i < n; i++) {
      const io = i * d, off = i * k;
      for (let c = 0; c < k; c++) {
        const w = weights[off + c], co = c * d; sums[c] += w;
        for (let f = 0; f < d; f++) numerator[co + f] += w * x[io + f];
      }
    }
    for (let c = 0; c < k; c++) {
      if (sums[c] < TINY) {
        let maximum = 0; for (let i = 0; i < n; i++) maximum = Math.max(maximum, u[i * k + c]);
        if (maximum > 0) {
          sums[c] = 0; numerator.fill(0, c * d, (c + 1) * d);
          for (let i = 0; i < n; i++) { const w = (u[i * k + c] / maximum) ** m; sums[c] += w; for (let f = 0; f < d; f++) numerator[c * d + f] += w * x[i * d + f]; }
        }
      }
      if (sums[c] > 0) for (let f = 0; f < d; f++) centers[c * d + f] = numerator[c * d + f] / sums[c];
    }
    assertFinite(centers, 'centers');
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) d2[i * k + c] = distance(x, i * d, centers, c * d, d);
    // Three linear passes over each K-entry row, including exact zero ties.
    for (let i = 0; i < n; i++) {
      const off = i * k; let minimum = Infinity, sum = 0;
      for (let c = 0; c < k; c++) minimum = Math.min(minimum, d2[off + c]);
      if (minimum === 0) {
        for (let c = 0; c < k; c++) { const w = d2[off + c] === 0 ? 1 : 0; next[off + c] = w; sum += w; }
      } else if (m === 2) {
        for (let c = 0; c < k; c++) { const w = minimum / d2[off + c]; next[off + c] = w; sum += w; }
      } else {
        const exponent = 1 / (m - 1);
        for (let c = 0; c < k; c++) { const ratio = minimum / d2[off + c]; const w = ratio < TINY ? Math.exp((Math.log(minimum) - Math.log(d2[off + c])) * exponent) : ratio ** exponent; next[off + c] = w; sum += w; }
      }
      for (let c = 0; c < k; c++) next[off + c] /= sum;
    }
    let delta2 = 0, objective = 0;
    for (let z = 0; z < u.length; z++) { const diff = next[z] - u[z]; delta2 += diff * diff; objective += objectiveTerm(next[z], d2[z], m); }
    delta = Math.sqrt(delta2); [u, next] = [next, u]; converged = delta < tolerance;
    if (options.returnHistory) history.push(objective);
    if (converged) break;
  }
  iterations = Math.min(iterations, maxIterations); restoreCenters(centers, shifted.origin);
  let objective = 0, fpc = 0;
  for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) { const value = u[i * k + c]; objective += objectiveTerm(value, distance(original.data, i * d, centers, c * d, d), m); fpc += value * value; }
  finiteNumber(objective, 'objective', 0);
  return { centers, membership: u, labels: labelsFromMembership(u, n, k), objective, fpc: fpc / n, delta, iterations, converged, ...(options.returnHistory ? {objectiveHistory: Float64Array.from(history)} : {}) };
}
