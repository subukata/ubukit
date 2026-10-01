import { fcmCenterBlock, fcmMembershipBlock } from './iteration-kernels.js';
import { checkpoint, SESSION_WARM_CENTERS } from './session-hooks.js';
import { normalizeInput, positiveInteger, finiteNumber, assertFinite, seededRandom, checkCancelled, progress, squaredDistance, translatedData, restoreCenters, initializeCenters, labelsFromMembership } from './core.js';

const MIN_NORMAL = 2.2250738585072014e-308;
function parameters(input, options, fuzzy = false) {
  const x = normalizeInput(input), n = x.nSamples, d = x.nFeatures;
  const inferred = options.initCenters?.length / d || (fuzzy && options.initMembership?.length / n);
  const k = positiveInteger(options.nClusters ?? inferred, 'nClusters');
  if (k > 2147483647) throw new RangeError('nClusters exceeds Int32 label range');
  const maxIterations = positiveInteger(options.maxIterations ?? 100, 'maxIterations');
  const blockRows = positiveInteger(options.blockRows ?? 512, 'blockRows');
  const tolerance = finiteNumber(options.tolerance ?? (fuzzy ? 1e-5 : 0), 'tolerance', 0);
  if (!Number.isSafeInteger(n * k) || !Number.isSafeInteger(k * d)) throw new RangeError('output dimensions exceed safe integer range');
  const estimatedBytes = 16 * n * d + (fuzzy ? 16 * n * k : 8 * n) + 24 * k * d + 32 * k;
  const maxMemoryBytes = finiteNumber(options.maxMemoryBytes ?? 512 * 1024 ** 2, 'maxMemoryBytes', 1);
  if (estimatedBytes > maxMemoryBytes) throw new RangeError(`estimated primary arrays ${estimatedBytes} bytes exceed maxMemoryBytes=${maxMemoryBytes}`);
  checkCancelled(options);
  return { x, n, d, k, maxIterations, blockRows, tolerance };
}
export function consumeSteps(iterator) {
  let step; do { step = iterator.next(); } while (!step.done); return step.value;
}
export function kmeans(input, options = {}) { return consumeSteps(kmeansSteps(input, options)); }
export function fcm(input, options = {}) { return consumeSteps(fcmSteps(input, options)); }
export function exrcm(input, options = {}) { return consumeSteps(exrcmSteps(input, options)); }
export function rcm(input, options = {}) {
  if (options.p != null && options.p !== 1) throw new RangeError('rcm fixes p=1; use exrcm for other p');
  return consumeSteps(exrcmSteps(input, { ...options, p: 1, _algorithm: 'rcm' }));
}

/** Lloyd iterations, first-index ties, retained empty centers, final label pass. */
export function* kmeansSteps(input, options = {}) {
  if (options.tolerance != null && options.tolerance !== 0) throw new RangeError('kmeans uses exact label convergence; nonzero tolerance is unsupported');
  const { x, n, d, k, maxIterations, blockRows } = parameters(input, options);
  const shifted = translatedData(x), data = shifted.data;
  const centers = initializeCenters({ ...x, data }, k, options, shifted.origin);
  const labels = new Int32Array(n).fill(-1), sums = new Float64Array(k * d), counts = new Float64Array(k);
  let iterations = 0, converged = false;
  for (iterations = 1; iterations <= maxIterations; ++iterations) {
    sums.fill(0); counts.fill(0); let changed = 0;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      for (let i = start; i < Math.min(n, start + blockRows); ++i) {
        const io = i * d; let best = 0, bestDistance = Infinity;
        for (let c = 0; c < k; ++c) {
          const s = squaredDistance(data, io, centers, c * d, d, bestDistance);
          if (s < bestDistance) { best = c; bestDistance = s; }
        }
        if (labels[i] !== best) ++changed;
        labels[i] = best; ++counts[best];
        for (let f = 0; f < d; ++f) sums[best * d + f] += data[io + f];
      }
      yield { phase: 'assignment', iteration: iterations, completedRows: Math.min(n, start + blockRows), totalRows: n };
    }
    for (let c = 0; c < k; ++c) if (counts[c] > 0) for (let f = 0; f < d; ++f) centers[c * d + f] = sums[c * d + f] / counts[c];
    assertFinite(centers, 'centers');
    converged = changed === 0;
    checkpoint(options, { algorithm: 'kmeans', centers, labels, coreLabels: labels, iterations, converged, nSamples: n, nFeatures: d, nClusters: k, objective: null, inertia: null, labelContract: 'iteration assignment that produced centers; nearest-final-centers only after finalization' }, shifted.origin);
    const event = { algorithm: 'kmeans', iteration: iterations, maxIterations, changed, converged };
    progress(options, event); yield event;
    if (converged) break;
  }
  iterations = Math.min(iterations, maxIterations);
  const coreLabels = labels.slice(); restoreCenters(centers, shifted.origin);
  let inertia = 0;
  for (let i = 0; i < n; ++i) {
    if (i % blockRows === 0) { checkCancelled(options); yield { phase: 'finalize', completedRows: i, totalRows: n }; }
    let best = 0, bestDistance = Infinity;
    for (let c = 0; c < k; ++c) { const s = squaredDistance(x.data, i * d, centers, c * d, d, bestDistance); if (s < bestDistance) { best = c; bestDistance = s; } }
    labels[i] = best; inertia += bestDistance;
  }
  finiteNumber(inertia, 'inertia', 0);
  return { algorithm: 'kmeans', centers, labels, coreLabels, inertia, objective: inertia, iterations, converged, nSamples: n, nFeatures: d, nClusters: k, backend: 'javascript-float64', labelContract: 'nearest-final-centers; direct-distance first-index ties' };
}

function normalizeMembership(u, n, k) {
  for (let i = 0; i < n; ++i) {
    const off = i * k; let max = 0;
    for (let c = 0; c < k; ++c) { finiteNumber(u[off + c], 'initMembership', 0); max = Math.max(max, u[off + c]); }
    if (!(max > 0)) throw new RangeError('every initMembership row must have a positive sum');
    let sum = 0; for (let c = 0; c < k; ++c) { u[off + c] /= max; sum += u[off + c]; }
    for (let c = 0; c < k; ++c) u[off + c] /= sum;
  }
}
/** Inverse powers normalized by row minimum prevent overflow near m=1. */
export function membershipsFromSquaredDistances(distances, nClusters, m = 2) {
  positiveInteger(nClusters, 'nClusters');
  if (!Number.isFinite(m) || m <= 1) throw new RangeError('m must be finite and > 1');
  if (distances.length % nClusters) throw new RangeError('distance length must be divisible by nClusters');
  const out = new Float64Array(distances.length), temp = new Float64Array(nClusters);
  for (let i = 0; i < distances.length; i += nClusters) {
    for (let c = 0; c < nClusters; ++c) temp[c] = finiteNumber(distances[i + c], 'squared distance', 0);
    fuzzyRow(temp, out, i, nClusters, m);
  }
  return out;
}
function fuzzyRow(dist, out, offset, k, m) {
  let minimum = Infinity; for (let c = 0; c < k; ++c) minimum = Math.min(minimum, dist[c]);
  let sum = 0;
  if (minimum === 0) {
    for (let c = 0; c < k; ++c) { const w = dist[c] === 0 ? 1 : 0; out[offset + c] = w; sum += w; }
  } else if (m === 2) {
    for (let c = 0; c < k; ++c) { const w = minimum / dist[c]; out[offset + c] = w; sum += w; }
  } else {
    const power = 1 / (m - 1);
    for (let c = 0; c < k; ++c) { const ratio = minimum / dist[c]; const w = ratio < MIN_NORMAL ? Math.exp((Math.log(minimum) - Math.log(dist[c])) * power) : ratio ** power; out[offset + c] = w; sum += w; }
  }
  for (let c = 0; c < k; ++c) out[offset + c] /= sum;
}
function objectiveTerm(u, distance, m) {
  if (u === 0 || distance === 0) return 0;
  const weight = m === 2 ? u * u : u ** m;
  return weight < MIN_NORMAL ? Math.exp(m * Math.log(u) + Math.log(distance)) : weight * distance;
}
function initialMembership(x, centers, k, options, m) {
  const { data, nSamples: n, nFeatures: d } = x, u = new Float64Array(n * k);
  if (options.initMembership != null) {
    if (!ArrayBuffer.isView(options.initMembership) || options.initMembership.length !== u.length) throw new RangeError('initMembership must be a TypedArray of nSamples * nClusters values');
    u.set(options.initMembership);
  } else if (options.initCenters != null) {
    const dist = new Float64Array(k);
    for (let i = 0; i < n; ++i) { for (let c = 0; c < k; ++c) dist[c] = squaredDistance(data, i * d, centers, c * d, d); fuzzyRow(dist, u, i * k, k, m); }
    return u;
  } else {
    const random = seededRandom(options.seed ?? 0);
    for (let j = 0; j < u.length; ++j) u[j] = random() || Number.EPSILON;
  }
  normalizeMembership(u, n, k); return u;
}
/** Standard FCM. Returned centers come from U_old; returned U is U_new. */
export function* fcmSteps(input, options = {}) {
  const { x, n, d, k, maxIterations, blockRows, tolerance } = parameters(input, options, true);
  const m = options.m ?? 2;
  if (!Number.isFinite(m) || m <= 1) throw new RangeError('m must be finite and > 1');
  if (options.initCenters != null && options.initMembership != null) throw new RangeError('choose initCenters or initMembership, not both');
  const shifted = translatedData(x), data = shifted.data, centers = new Float64Array(k * d);
  // Empty initial fuzzy clusters retain the data mean, as in the Python oracle.
  for (let f = 0; f < d; ++f) { let sum = 0; for (let i = 0; i < n; ++i) sum += data[i * d + f] / n; for (let c = 0; c < k; ++c) centers[c * d + f] = sum; }
  if (options[SESSION_WARM_CENTERS]) {
    const old = options[SESSION_WARM_CENTERS];
    for (let j = 0; j < centers.length; ++j) centers[j] = old[j] - shifted.origin[j % d];
  }
  let warm = centers;
  if (options.initCenters != null) warm = initializeCenters({ ...x, data }, k, options, shifted.origin);
  let u = initialMembership({ ...x, data }, warm, k, options, m), unew = new Float64Array(n * k);
  const sums = new Float64Array(k), maxima = new Float64Array(k), newcenters = new Float64Array(k * d), dist = new Float64Array(k), history = [];
  const accum = { delta2: 0, objective: 0 };
  let iterations = 0, delta = Infinity, objective = Infinity, converged = false;
  for (iterations = 1; iterations <= maxIterations; ++iterations) {
    sums.fill(0); newcenters.fill(0);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      fcmCenterBlock(data, u, sums, newcenters, start, Math.min(n, start + blockRows), d, k, m);
      yield { phase: 'centers', iteration: iterations, completedRows: Math.min(n, start + blockRows), totalRows: n };
    }
    for (let c = 0; c < k; ++c) {
      if (sums[c] < MIN_NORMAL) {
        let maximum = 0;
        for (let i = 0; i < n; ++i) maximum = Math.max(maximum, u[i * k + c]);
        maxima[c] = maximum;
      }
      if (sums[c] < MIN_NORMAL && maxima[c] > 0) {
        sums[c] = 0; newcenters.fill(0, c * d, (c + 1) * d);
        for (let i = 0; i < n; ++i) { const weight = (u[i * k + c] / maxima[c]) ** m; sums[c] += weight; for (let f = 0; f < d; ++f) newcenters[c * d + f] += weight * data[i * d + f]; }
      }
      if (sums[c] > 0) for (let f = 0; f < d; ++f) centers[c * d + f] = newcenters[c * d + f] / sums[c];
    }
    assertFinite(centers, 'centers');
    accum.delta2 = 0; accum.objective = 0;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      fcmMembershipBlock(data, centers, u, unew, dist, start, Math.min(n, start + blockRows), d, k, m, accum);
      yield { phase: 'membership', iteration: iterations, completedRows: Math.min(n, start + blockRows), totalRows: n };
    }
    [u, unew] = [unew, u]; objective = accum.objective; delta = Math.sqrt(accum.delta2); converged = delta < tolerance;
    if (options.returnHistory) history.push(objective);
    checkpoint(options, { algorithm: 'fcm', centers, membership: u, membershipLayout: 'samples-clusters', iterations, converged, nSamples: n, nFeatures: d, nClusters: k, m, objective, delta, ...(options.returnHistory ? { objectiveHistory: Float64Array.from(history) } : {}) }, shifted.origin);
    const event = { algorithm: 'fcm', iteration: iterations, maxIterations, objective, delta, converged };
    progress(options, event); yield event;
    if (converged) break;
  }
  iterations = Math.min(iterations, maxIterations); restoreCenters(centers, shifted.origin);
  // Objective is recomputed on exactly the published centers, including rounding.
  objective = 0; let fpc = 0;
  for (let i = 0; i < n; ++i) {
    if (i % blockRows === 0) { checkCancelled(options); yield { phase: 'finalize', completedRows: i, totalRows: n }; }
    for (let c = 0; c < k; ++c) { const v = u[i * k + c]; objective += objectiveTerm(v, squaredDistance(x.data, i * d, centers, c * d, d), m); fpc += v * v; }
  }
  finiteNumber(objective, 'objective', 0);
  return { algorithm: 'fcm', centers, membership: u, membershipLayout: 'samples-clusters', labels: labelsFromMembership(u, n, k), objective, fpc: fpc / n, delta, iterations, converged, nSamples: n, nFeatures: d, nClusters: k, m, backend: 'javascript-float64', ...(options.returnHistory ? { objectiveHistory: Float64Array.from(history) } : {}) };
}

/** Compare powers without forming alpha^p, beta^p, or large d^p. */
export function roughAdmissible(distance, minimum, alpha, beta, p) {
  const a = alpha * minimum;
  if (beta === 0) return distance <= a;
  if (p === 1) return distance <= a + beta;
  if (!Number.isFinite(a)) return true; // The radius exceeds every finite distance.
  const scale = Math.max(a, beta);
  if (scale === 0) return distance === 0;
  if (distance <= scale) return true;
  const small = Math.min(a, beta);
  if (small === 0) return false;
  const smallScaled = small / scale, leftScaled = distance / scale;
  if (p >= .25 && p <= 64 && smallScaled >= MIN_NORMAL && Number.isFinite(leftScaled)) return leftScaled ** p <= 1 + smallScaled ** p;
  // Log space preserves tiny ratios raised to a fractional p. log1p keeps
  // near-boundary differences that distance/scale could round down to one.
  const relative = (distance - scale) / scale;
  const logRatio = Number.isFinite(relative) ? Math.log1p(relative) : Math.log(distance) - Math.log(scale);
  const smallRatio = small / scale;
  const logSmall = smallRatio >= MIN_NORMAL ? Math.log(smallRatio) : Math.log(small) - Math.log(scale);
  return p * logRatio <= Math.log1p(Math.exp(p * logSmall));
}
function sameMask(a, b) { if (!b) return false; for (let j = 0; j < a.length; ++j) if (a[j] !== b[j]) return false; return true; }
function* assignRough(data, centers, n, d, k, alpha, beta, p, options, blockRows, iteration, out, mask) {
  const distances = new Float64Array(k), squared = new Float64Array(k);
  const alpha2 = alpha * alpha, beta2 = beta * beta;
  const useSquared = p === 2 && Number.isFinite(alpha2) && (beta === 0 || beta2 > 0);
  for (let start = 0; start < n; start += blockRows) {
    checkCancelled(options);
    for (let i = start; i < Math.min(n, start + blockRows); ++i) {
      let minimum = Infinity;
      for (let c = 0; c < k; ++c) { squared[c] = squaredDistance(data, i * d, centers, c * d, d); distances[c] = Math.sqrt(squared[c]); minimum = Math.min(minimum, distances[c]); }
      let minSquared = Infinity; for (let c = 0; c < k; ++c) minSquared = Math.min(minSquared, squared[c]);
      const thresholdSquared = alpha2 * minSquared + beta2;
      let count = 0;
      for (let c = 0; c < k; ++c) { const accepted = useSquared ? squared[c] <= thresholdSquared : roughAdmissible(distances[c], minimum, alpha, beta, p); const value = accepted ? 1 : 0; mask[i * k + c] = value; count += value; }
      if (count === 0) throw new RangeError('no admissible rough cluster; numerical range unsupported');
      for (let c = 0; c < k; ++c) out[i * k + c] = mask[i * k + c] / count;
    }
    yield { phase: 'membership', iteration, completedRows: Math.min(n, start + blockRows), totalRows: n };
  }
}
/** Generalized rough c-means; p=1 is the RCM API. */
export function* exrcmSteps(input, options = {}) {
  const { x, n, d, k, maxIterations, blockRows } = parameters(input, options);
  const alpha = finiteNumber(options.alpha ?? 1.1, 'alpha', 1), beta = finiteNumber(options.beta ?? 0, 'beta', 0), p = finiteNumber(options.p ?? 1, 'p', Number.MIN_VALUE);
  const cycleWindow = options.cycleWindow ?? 16;
  if (!Number.isSafeInteger(cycleWindow) || cycleWindow < 0) throw new RangeError('cycleWindow must be a nonnegative integer');
  const memoryEstimate = 16 * n * d + (9 + Math.max(1, cycleWindow)) * n * k + (24 + 8 * Math.max(1, cycleWindow)) * k * d + 16 * k;
  if (memoryEstimate > (options.maxMemoryBytes ?? 512 * 1024 ** 2)) throw new RangeError('rough membership/cycle arrays exceed maxMemoryBytes');
  const shifted = translatedData(x), data = shifted.data, centers = initializeCenters({ ...x, data }, k, options, shifted.origin);
  const u = new Float64Array(n * k), mask = new Uint8Array(n * k), previous = [], sums = new Float64Array(k), newcenters = new Float64Array(k * d);
  let iterations = 0, converged = false, stopReason = 'maxIterations';
  const algorithm = options._algorithm ?? 'exrcm';
  for (iterations = 1; iterations <= maxIterations; ++iterations) {
    yield* assignRough(data, centers, n, d, k, alpha, beta, p, options, blockRows, iterations, u, mask);
    const stableMask = sameMask(mask, previous.at(-1)?.mask);
    sums.fill(0); newcenters.fill(0);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      for (let i = start; i < Math.min(n, start + blockRows); ++i) for (let c = 0; c < k; ++c) {
        const weight = u[i * k + c]; if (weight === 0) continue; sums[c] += weight;
        for (let f = 0; f < d; ++f) newcenters[c * d + f] += weight * data[i * d + f];
      }
      yield { phase: 'centers', iteration: iterations, completedRows: Math.min(n, start + blockRows), totalRows: n };
    }
    let exact = true;
    for (let c = 0; c < k; ++c) if (sums[c] > 0) for (let f = 0; f < d; ++f) { const value = newcenters[c * d + f] / sums[c]; if (value !== centers[c * d + f]) exact = false; centers[c * d + f] = value; }
    assertFinite(centers, 'centers');
    const stableState = stableMask && sameMask(centers, previous.at(-1)?.centers);
    const cycle = cycleWindow > 0 && previous.slice(0, -1).some(old => sameMask(mask, old.mask) && sameMask(centers, old.centers));
    previous.push({ mask: mask.slice(), centers: centers.slice() }); if (previous.length > Math.max(1, cycleWindow)) previous.shift();
    converged = exact || stableState;
    checkpoint(options, { algorithm, centers, membership: u, mask, membershipLayout: 'samples-clusters', iterations, converged, stopReason: converged ? 'fixedPoint' : cycle ? 'membershipCycle' : null, nSamples: n, nFeatures: d, nClusters: k, alpha, beta, p, cycleWindow, membershipContract: 'iteration membership that produced centers; re-evaluated at final centers after finalization' }, shifted.origin);
    const event = { algorithm, iteration: iterations, maxIterations, converged };
    progress(options, event); yield event;
    if (converged) { stopReason = 'fixedPoint'; break; }
    if (cycle) { stopReason = 'membershipCycle'; break; }
  }
  iterations = Math.min(iterations, maxIterations);
  restoreCenters(centers, shifted.origin);
  // Published memberships are always evaluated at published final centers.
  yield* assignRough(x.data, centers, n, d, k, alpha, beta, p, options, blockRows, iterations, u, mask);
  if (sameMask(mask, previous.at(-1)?.mask)) { converged = true; stopReason = 'fixedPoint'; }
  return { algorithm, centers, membership: u, mask, membershipLayout: 'samples-clusters', labels: labelsFromMembership(u, n, k), iterations, converged, stopReason, nSamples: n, nFeatures: d, nClusters: k, alpha, beta, p, distance: 'euclidean', backend: 'javascript-float64', cycleWindow };
}
export function* rcmSteps(input, options = {}) {
  if (options.p != null && options.p !== 1) throw new RangeError('rcm fixes p=1; use exrcm for other p');
  return yield* exrcmSteps(input, { ...options, p: 1, _algorithm: 'rcm' });
}
