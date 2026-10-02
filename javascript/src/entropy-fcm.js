/** Entropy-regularized fuzzy c-means, independently derived coordinate descent.
 * J(U,V) = sum U[i,c] ||X[i]-V[c]||^2 + tau sum U[i,c] log U[i,c].
 * This model uses linear U weights, simplex rows, tau > 0, and 0 log 0 = 0.
 */
import { normalizeInput, positiveInteger, finiteNumber, assertFinite, seededRandom,
  checkCancelled, progress, labelsFromMembership } from './core.js';
import { somStableMean } from './som-numerics.js';

// Exact dyadic sums/products extend the exponent range without rounding away
// cancellation. Only log/exp and final binary64 conversions are inexact.
// These are scalar arithmetic helpers, not SOM costs or update rules.
const view = new DataView(new ArrayBuffer(8));
function binary(value) {
  view.setFloat64(0, value, false);
  const hi = view.getUint32(0, false), lo = view.getUint32(4, false);
  const field = (hi >>> 20) & 2047;
  let integer = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  if (field) integer |= 1n << 52n;
  if (hi >>> 31) integer = -integer;
  return { integer, exponent: field ? field - 1075 : -1074 };
}
function add(sum, term) {
  if (term.integer === 0n) return sum;
  if (sum.integer === 0n) { sum.integer = term.integer; sum.exponent = term.exponent; }
  else if (term.exponent < sum.exponent) {
    sum.integer = (sum.integer << BigInt(sum.exponent - term.exponent)) + term.integer;
    sum.exponent = term.exponent;
  } else sum.integer += term.integer << BigInt(term.exponent - sum.exponent);
  return sum;
}
function multiply(a, b) { return { integer: a.integer * b.integer, exponent: a.exponent + b.exponent }; }
function subtract(a, b) { return add({ ...a }, { integer: -b.integer, exponent: b.exponent }); }
function zero() { return { integer: 0n, exponent: 0 }; }
function squaredCost(data, offset, centers, centerOffset, d) {
  const sum = zero();
  for (let f = 0; f < d; ++f) {
    const difference = subtract(binary(data[offset + f]), binary(centers[centerOffset + f]));
    add(sum, multiply(difference, difference));
  }
  return sum;
}
/** Correctly round an exact positive dyadic ratio, including subnormal output. */
function positiveRatio(integer, exponent, denominator = 1n) {
  if (integer === 0n) return 0;
  let power = integer.toString(2).length - denominator.toString(2).length;
  if (power >= 0 ? integer < (denominator << BigInt(power)) : (integer << BigInt(-power)) < denominator) --power;
  const magnitude = power + exponent;
  if (magnitude > 1023) return Infinity;
  if (magnitude < -1075) return 0;
  const unit = Math.max(-1074, magnitude - 52), shift = exponent - unit;
  const numerator = shift >= 0 ? integer << BigInt(shift) : integer;
  const divisor = shift >= 0 ? denominator : denominator << BigInt(-shift);
  let quotient = numerator / divisor;
  const remainder = numerator % divisor;
  if (2n * remainder > divisor || (2n * remainder === divisor && (quotient & 1n))) ++quotient;
  return Number(quotient) * 2 ** unit;
}
function objectiveResult(sum) {
  const sign = sum.integer < 0n ? -1 : sum.integer > 0n ? 1 : 0;
  const magnitude = sign < 0 ? -sum.integer : sum.integer;
  const objective = sign * positiveRatio(magnitude, sum.exponent);
  const bits = magnitude.toString(2).length, shift = Math.max(0, bits - 53);
  const objectiveLogAbs = sign ? Math.log(Number(magnitude >> BigInt(shift))) + (sum.exponent + shift) * Math.LN2 : -Infinity;
  const objectiveRepresentation = !Number.isFinite(objective) ? 'overflow' : objective === 0 && sign ? 'underflow' : 'finite';
  return { objective, objectiveRepresentation, objectiveSign: sign, objectiveLogAbs };
}

function typedArray(value, length, name) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView || typeof value[0] === 'bigint' || value.length !== length)
    throw new RangeError(`${name} must be a numeric TypedArray of ${length} values`);
  assertFinite(value, name);
}
function parameters(input, options) {
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
  // history, and conservative per-cluster dyadic workspace. Not a JS heap cap.
  const estimatedBytes = 16 * n * d + 16 * n * k + 16 * k * d + 4 * n + 2048 * k +
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
function membershipRow(data, centers, row, d, k, temperature, u, costs) {
  let minimum = 0;
  for (let c = 0; c < k; ++c) {
    costs[c] = squaredCost(data, row * d, centers, c * d, d);
    if (c && subtract(costs[c], costs[minimum]).integer < 0n) minimum = c;
  }
  let units = 0, tail = 0, correction = 0;
  for (let c = 0; c < k; ++c) {
    const difference = subtract(costs[c], costs[minimum]);
    const ratio = positiveRatio(difference.integer, difference.exponent - temperature.exponent, temperature.integer);
    const value = Math.exp(-ratio); u[row * k + c] = value;
    if (value === 1) ++units;
    else { const next = tail + value; correction += Math.abs(tail) >= Math.abs(value) ? (tail - next) + value : (value - next) + tail; tail = next; }
  }
  const total = units + (tail + correction);
  for (let c = 0; c < k; ++c) u[row * k + c] /= total;
}
function objectiveRow(u, row, k, temperature, sum, costs) {
  for (let c = 0; c < k; ++c) {
    const member = u[row * k + c];
    if (member === 0) continue; // 0 log 0 = 0, including huge squared costs.
    const weight = binary(member);
    // costs belongs to the immediately preceding membershipRow call. There is
    // no yield or callback between producing and consuming this one-row cache.
    add(sum, multiply(weight, costs[c]));
    add(sum, multiply(temperature, multiply(weight, binary(Math.log(member)))));
  }
}

export function entropyFcm(input, options = {}) {
  const iterator = entropyFcmSteps(input, options);
  for (;;) { const next = iterator.next(); if (next.done) return next.value; }
}

/** Cooperative row blocks; no stateful/session integration in this release. */
export function* entropyFcmSteps(input, options = {}) {
  const { x, n, d, k, tau, maxIterations, blockRows, tolerance } = parameters(input, options);
  const data = new Float64Array(x.data), centers = new Float64Array(k * d), temperature = binary(tau);
  let u = new Float64Array(n * k), nextU = new Float64Array(n * k);
  const costs = new Array(k), history = [];
  // somStableMean only supplies a scalar robust convex mean. The objective
  // and every clustering update here are independent of SOM/SOM-OLP.
  for (let f = 0; f < d; ++f) {
    checkCancelled(options);
    const mean = somStableMean(data, f, d, n);
    for (let c = 0; c < k; ++c) centers[c * d + f] = mean;
  }
  if (options.initMembership != null) { u.set(options.initMembership); normalizeMembership(u, n, k); }
  else if (options.initCenters != null) {
    const initialCenters = new Float64Array(options.initCenters);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      const end = Math.min(n, start + blockRows);
      for (let i = start; i < end; ++i) membershipRow(data, initialCenters, i, d, k, temperature, u, costs);
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
      let positive = false;
      for (let i = 0; i < n && !positive; ++i) positive = u[i * k + c] > 0;
      if (positive) for (let f = 0; f < d; ++f) centers[c * d + f] = somStableMean(data, f, d, n, u, c, k);
      yield { phase: 'centers', iteration: iterations, completedRows: c + 1, totalRows: k };
    }
    delta = 0;
    const objectiveSum = zero();
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      const end = Math.min(n, start + blockRows);
      for (let i = start; i < end; ++i) {
        membershipRow(data, centers, i, d, k, temperature, nextU, costs);
        for (let c = 0; c < k; ++c) delta = Math.hypot(delta, nextU[i * k + c] - u[i * k + c]);
        objectiveRow(nextU, i, k, temperature, objectiveSum, costs);
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
    backend: 'javascript-float64', numericalMode: 'exact-binary-reference', stoppingRule: 'membership-frobenius',
    ...(options.returnHistory ? { objectiveHistory: Float64Array.from(history) } : {}) };
}
