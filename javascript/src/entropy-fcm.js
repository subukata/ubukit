/** Entropy-regularized fuzzy c-means, independently derived coordinate descent.
 * J(U,V) = sum U[i,c] ||X[i]-V[c]||^2 + tau sum U[i,c] log U[i,c].
 * This model uses linear U weights, simplex rows, tau > 0, and 0 log 0 = 0.
 */
import { normalizeInput, positiveInteger, finiteNumber, assertFinite, seededRandom,
  checkCancelled, progress, labelsFromMembership } from './core.js';

// All updates use ordinary float64 arithmetic. Reject arithmetic overflow
// instead of switching to exact arithmetic or returning extended-range values.
function finiteIntermediate(value, operation) {
  if (!Number.isFinite(value))
    throw new RangeError(`entropy-fcm ${operation} is nonfinite in float64; rescale input or tau`);
  return value;
}
function weightedMean(data, feature, d, n, mass, maximum, lower, upper, membership, cluster, k) {
  // A constant feature has a known mean; no summation is needed.
  if (lower === upper) return lower;
  let sum = 0;
  for (let i = 0; i < n; ++i) {
    // Normalize first to avoid avoidable overflow in an unnormalized numerator.
    const term = membership
      ? (membership[i * k + cluster] / maximum / mass) * data[i * d + feature]
      : data[i * d + feature] / n;
    finiteIntermediate(term, 'center product');
    sum = finiteIntermediate(sum + term, 'center sum');
  }
  // Roundoff can move a convex mean just outside the observed feature range.
  return Math.max(lower, Math.min(upper, sum));
}

function squaredCost(data, offset, centers, centerOffset, d) {
  let sum = 0;
  for (let f = 0; f < d; ++f) {
    const difference = finiteIntermediate(data[offset + f] - centers[centerOffset + f], 'coordinate difference');
    const square = finiteIntermediate(difference * difference, 'squared distance');
    sum = finiteIntermediate(sum + square, 'squared distance sum');
  }
  return sum;
}
function objectiveResult(objective) {
  finiteIntermediate(objective, 'objective');
  const objectiveSign = objective < 0 ? -1 : objective > 0 ? 1 : 0;
  return { objective, objectiveRepresentation: 'finite', objectiveSign,
    objectiveLogAbs: objectiveSign ? Math.log(Math.abs(objective)) : -Infinity };
}

function typedArray(value, length, name) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView || typeof value[0] === 'bigint' || value.length !== length)
    throw new RangeError(`${name} must be a numeric TypedArray of ${length} values`);
  assertFinite(value, name);
}
function parameters(input, options) {
  if (options.backend !== undefined && options.backend !== 'javascript-float64')
    throw new RangeError('entropy-fcm backend must be javascript-float64; exact/reference backends are not supported');
  if (options.initMembership != null && options.initCenters != null)
    throw new RangeError('choose either initMembership or initCenters, not both');
  const x = normalizeInput(input), n = x.nSamples, d = x.nFeatures;
  const inferred = options.initCenters?.length / d || options.initMembership?.length / n;
  const k = positiveInteger(options.nClusters ?? inferred, 'nClusters');
  if (k > 2147483647 || !Number.isSafeInteger(n * k) || !Number.isSafeInteger(k * d))
    throw new RangeError('output dimensions exceed safe integer or Int32 label range');
  const tau = finiteNumber(options.tau ?? 1, 'tau', 0);
  if (tau === 0) throw new RangeError('tau must be positive');
  const maxIterations = positiveInteger(options.maxIterations ?? 100, 'maxIterations');
  const blockRows = positiveInteger(options.blockRows ?? 512, 'blockRows');
  const tolerance = finiteNumber(options.tolerance ?? 1e-5, 'tolerance', 0);
  const maxMemoryBytes = finiteNumber(options.maxMemoryBytes ?? 512 * 1024 ** 2, 'maxMemoryBytes', 1);
  // Includes input copy/conversion allowance, two U arrays, centers/labels,
  // history, and conservative per-cluster float64 workspace. Not a JS heap cap.
  const estimatedBytes = 16 * n * d + 16 * n * k + 16 * k * d + 4 * n + 16 * k + 16 * d +
    (options.returnHistory ? 16 * maxIterations : 0) + 8192;
  if (!Number.isSafeInteger(estimatedBytes) || estimatedBytes > maxMemoryBytes)
    throw new RangeError(`estimated primary arrays/workspace ${estimatedBytes} bytes exceed maxMemoryBytes=${maxMemoryBytes}`);
  if (options.initMembership != null) typedArray(options.initMembership, n * k, 'initMembership');
  if (options.initCenters != null) typedArray(options.initCenters, k * d, 'initCenters');
  checkCancelled(options);
  return { x, n, d, k, tau, maxIterations, blockRows, tolerance };
}
function normalizeMembership(u, n, k) {
  for (let i = 0; i < n; ++i) {
    const offset = i * k; let maximum = 0, sum = 0;
    for (let c = 0; c < k; ++c) maximum = Math.max(maximum, finiteNumber(u[offset + c], 'initMembership', 0));
    if (!(maximum > 0)) throw new RangeError('every initMembership row must have a positive sum');
    for (let c = 0; c < k; ++c) sum += u[offset + c] /= maximum;
    for (let c = 0; c < k; ++c) u[offset + c] /= sum;
  }
}
function membershipRow(data, centers, row, d, k, tau, u, costs) {
  let minimum = Infinity;
  for (let c = 0; c < k; ++c) {
    costs[c] = squaredCost(data, row * d, centers, c * d, d);
    minimum = Math.min(minimum, costs[c]);
  }
  let total = 0;
  for (let c = 0; c < k; ++c) {
    // Costs are finite and nonnegative. Subtract before division so a tiny
    // temperature does not cause Infinity - Infinity. A positive ratio may
    // overflow and exp(-ratio) may underflow; both correctly give zero weight.
    const ratio = (costs[c] - minimum) / tau;
    const value = Math.exp(-ratio);
    u[row * k + c] = value;
    total += value;
  }
  // At least one minimum has exp(0) = 1, so the finite sum is always positive.
  for (let c = 0; c < k; ++c) u[row * k + c] /= total;
}
function objectiveRow(u, row, k, tau, sum, costs) {
  for (let c = 0; c < k; ++c) {
    const member = u[row * k + c];
    if (member === 0) continue; // 0 log 0 = 0.
    // costs belongs to the immediately preceding membershipRow call. There is
    // no yield or callback between producing and consuming this one-row cache.
    const distortion = finiteIntermediate(member * costs[c], 'objective distance product');
    const entropy = finiteIntermediate(tau * member * Math.log(member), 'objective entropy product');
    const term = finiteIntermediate(distortion + entropy, 'objective term');
    sum = finiteIntermediate(sum + term, 'objective sum');
  }
  return sum;
}

export function entropyFcm(input, options = {}) {
  const iterator = entropyFcmSteps(input, options);
  for (;;) { const next = iterator.next(); if (next.done) return next.value; }
}

/** Cooperative row blocks; no stateful/session integration in this release. */
export function* entropyFcmSteps(input, options = {}) {
  const { x, n, d, k, tau, maxIterations, blockRows, tolerance } = parameters(input, options);
  const data = new Float64Array(x.data), centers = new Float64Array(k * d);
  let u = new Float64Array(n * k), nextU = new Float64Array(n * k);
  const costs = new Float64Array(k), minima = new Float64Array(d), maxima = new Float64Array(d), history = [];
  // Empty columns retain the previous center, initially the float64 data mean.
  for (let f = 0; f < d; ++f) {
    checkCancelled(options);
    let lower = data[f], upper = data[f];
    for (let i = 1; i < n; ++i) {
      lower = Math.min(lower, data[i * d + f]);
      upper = Math.max(upper, data[i * d + f]);
    }
    minima[f] = lower; maxima[f] = upper;
    const mean = weightedMean(data, f, d, n, n, 1, lower, upper);
    for (let c = 0; c < k; ++c) centers[c * d + f] = mean;
  }
  if (options.initMembership != null) { u.set(options.initMembership); normalizeMembership(u, n, k); }
  else if (options.initCenters != null) {
    const initialCenters = new Float64Array(options.initCenters);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      const end = Math.min(n, start + blockRows);
      for (let i = start; i < end; ++i) membershipRow(data, initialCenters, i, d, k, tau, u, costs);
      yield { phase: 'initialize', completedRows: end, totalRows: n };
    }
  } else {
    const random = seededRandom(options.seed ?? 0);
    for (let j = 0; j < u.length; ++j) u[j] = random() || Number.EPSILON;
    normalizeMembership(u, n, k);
  }
  let iterations = 0, converged = false, delta = Infinity, obj;
  for (iterations = 1; iterations <= maxIterations; ++iterations) {
    for (let c = 0; c < k; ++c) {
      checkCancelled(options);
      let maximum = 0;
      for (let i = 0; i < n; ++i) maximum = Math.max(maximum, u[i * k + c]);
      if (maximum > 0) {
        let mass = 0;
        for (let i = 0; i < n; ++i) mass += u[i * k + c] / maximum;
        finiteIntermediate(mass, 'center weight sum');
        for (let f = 0; f < d; ++f)
          centers[c * d + f] = weightedMean(data, f, d, n, mass, maximum, minima[f], maxima[f], u, c, k);
      }
      yield { phase: 'centers', iteration: iterations, completedRows: c + 1, totalRows: k };
    }
    delta = 0;
    let objectiveSum = 0;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      const end = Math.min(n, start + blockRows);
      for (let i = start; i < end; ++i) {
        membershipRow(data, centers, i, d, k, tau, nextU, costs);
        for (let c = 0; c < k; ++c) delta = Math.hypot(delta, nextU[i * k + c] - u[i * k + c]);
        objectiveSum = objectiveRow(nextU, i, k, tau, objectiveSum, costs);
      }
      yield { phase: 'membership', iteration: iterations, completedRows: end, totalRows: n };
    }
    [u, nextU] = [nextU, u];
    obj = objectiveResult(objectiveSum);
    if (options.returnHistory) history.push(obj.objective);
    converged = delta < tolerance;
    const event = { algorithm: 'entropy-fcm', iteration: iterations, maxIterations, ...obj, delta, converged };
    progress(options, event); yield event;
    if (converged) break;
  }
  checkCancelled(options);
  let fpc = 0; for (const member of u) fpc += member * member;
  return { algorithm: 'entropy-fcm', centers, membership: u, membershipLayout: 'samples-clusters',
    labels: labelsFromMembership(u, n, k), ...obj, fpc: fpc / n, tau, delta,
    iterations: Math.min(iterations, maxIterations), converged, nSamples: n, nFeatures: d, nClusters: k,
    backend: 'javascript-float64', numericalMode: 'float64', stoppingRule: 'membership-frobenius',
    ...(options.returnHistory ? { objectiveHistory: Float64Array.from(history) } : {}) };
}
