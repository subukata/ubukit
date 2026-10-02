import { kmeansWasmWorkspace, assignWasmBlock, finalizeWasmBlock } from './kmeans-wasm.js';
import { roughCenterBlock, fcmCenterBlock, fcmWeightedCenterBlock, fcmMembershipBlock, fullSquaredDistance, fullSquaredDistancesRow, addLogObjectiveTerm, logObjectiveValue, kmeansAssignment2dBlock, kmeansGroupedAssignmentBlock, kmeansFinalize2dBlock, kmeansFinalizeGroupedBlock } from './iteration-kernels.js';
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
  const estimatedBytes = 16 * n * d + (fuzzy ? 16 * n * k + d : 8 * n) + 24 * k * d + 32 * k;
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
  const kernelBackend = options.kernelBackend ?? 'javascript';
  if (!['javascript','wasm'].includes(kernelBackend)) throw new RangeError("kernelBackend must be 'javascript' or 'wasm'");
  const shifted = translatedData(x), data = shifted.data;
  const centers = initializeCenters({ ...x, data }, k, options, shifted.origin);
  const labels = new Int32Array(n).fill(-1), sums = new Float64Array(k * d), counts = new Float64Array(k);
  let iterations = 0, converged = false;
  const assignmentBlock = d === 2 ? kmeansAssignment2dBlock : kmeansGroupedAssignmentBlock;
  const estimatedBytes = 16 * n * d + 8 * n + 24 * k * d + 32 * k;
  const wasm = kernelBackend === 'wasm' ? kmeansWasmWorkspace(n,d,k,blockRows,(options.maxMemoryBytes ?? 512 * 1024 ** 2)-estimatedBytes) : null;
  const execution = {requested:kernelBackend,actual:wasm?'wasm':'javascript',allocatedWorkspaceBytes:wasm?.allocatedBytes??0,wasmAssignmentRows:0,wasmFinalizationRows:0,javascriptFallbackAssignmentRows:0,javascriptFallbackFinalizationRows:0};
  for (iterations = 1; iterations <= maxIterations; ++iterations) {
    sums.fill(0); counts.fill(0); let changed = 0;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      const end = Math.min(n, start + blockRows);
      changed += wasm
        ? assignWasmBlock(wasm, data, centers, labels, sums, counts, start, end, d, k, assignmentBlock, execution)
        : assignmentBlock(data, centers, labels, sums, counts, start, end, d, k);
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
  const finalizeBlock = d === 2 ? kmeansFinalize2dBlock : kmeansFinalizeGroupedBlock;
  for (let start = 0; start < n; start += blockRows) {
    checkCancelled(options); yield { phase: 'finalize', completedRows: start, totalRows: n };
    const end = Math.min(n, start + blockRows);
    inertia = wasm
      ? finalizeWasmBlock(wasm, x.data, centers, labels, start, end, d, k, inertia, finalizeBlock, execution)
      : finalizeBlock(x.data, centers, labels, start, end, d, k, inertia);
  }
  finiteNumber(inertia, 'inertia', 0);
  if(!wasm && kernelBackend==='wasm'){execution.javascriptFallbackAssignmentRows=n*iterations;execution.javascriptFallbackFinalizationRows=n;}
  if(wasm && execution.javascriptFallbackAssignmentRows+execution.javascriptFallbackFinalizationRows>0)execution.actual='mixed';
  execution.estimatedPeakPrimaryBytes=estimatedBytes+execution.allocatedWorkspaceBytes;
  return { algorithm: 'kmeans', centers, labels, coreLabels, inertia, objective: inertia, iterations, converged, nSamples: n, nFeatures: d, nClusters: k, backend: 'javascript-float64', labelContract: 'nearest-final-centers; direct-distance first-index ties', ...(options.kernelBackend != null ? {kernel:execution} : {}) };
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
    const power = 1 / (m - 1), nearOne = m - 1 < 1e-4;
    for (let c = 0; c < k; ++c) { const ratio = minimum / dist[c]; const w = nearOne ? (dist[c] - minimum <= minimum * (1024 * (m - 1)) ? Math.exp(-Math.log1p((dist[c] - minimum) / minimum) * power) : 0) : ratio < MIN_NORMAL ? Math.exp((Math.log(minimum) - Math.log(dist[c])) * power) : power === 0.5 ? Math.sqrt(ratio) : power === 2 ? ratio * ratio : ratio ** power; out[offset + c] = w; sum += w; }
  }
  for (let c = 0; c < k; ++c) out[offset + c] /= sum;
}
function objectiveTerm(u, distance, m, state, weight) {
  if (u === 0 || distance === 0) return 0;
  if (weight < MIN_NORMAL) {
    const term = m * Math.log(u) + Math.log(distance);
    if (state) { addLogObjectiveTerm(state, term); return 0; }
    return Math.exp(term);
  }
  return weight * distance;
}
function cacheMembershipPowers(u, powers, start, stop, m) {
  for (let j = start; j < stop; ++j) powers[j] = u[j] ** m;
}
function initialMembership(x, centers, k, options, m) {
  const { data, nSamples: n, nFeatures: d } = x, u = new Float64Array(n * k);
  if (options.initMembership != null) {
    if (!ArrayBuffer.isView(options.initMembership) || options.initMembership.length !== u.length) throw new RangeError('initMembership must be a TypedArray of nSamples * nClusters values');
    u.set(options.initMembership);
  } else if (options.initCenters != null) {
    const dist = new Float64Array(k);
    for (let i = 0; i < n; ++i) { for (let c = 0; c < k; ++c) dist[c] = fullSquaredDistance(data, i * d, centers, c * d, d); fuzzyRow(dist, u, i * k, k, m); }
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
  let minimumCoordinate = 1;
  const nonzeroCoordinate = new Uint8Array(d);
  for (let f = 0; f < d; ++f) {
    let sum = 0;
    for (let i = 0; i < n; ++i) {
      const value = data[i * d + f]; sum += value / n;
      if (value !== 0) { nonzeroCoordinate[f] = 1; minimumCoordinate = Math.min(minimumCoordinate, Math.abs(value)); }
    }
    for (let c = 0; c < k; ++c) centers[c * d + f] = sum;
  }
  // A normal column mass can still have underflowing weighted coordinates.
  // Scale only uniformly tiny columns; preserve ordinary reduction arithmetic.
  const centerWeightFloor = MIN_NORMAL / minimumCoordinate;
  if (options[SESSION_WARM_CENTERS]) {
    const old = options[SESSION_WARM_CENTERS];
    for (let j = 0; j < centers.length; ++j) centers[j] = old[j] - shifted.origin[j % d];
  }
  let warm = centers;
  if (options.initCenters != null) warm = initializeCenters({ ...x, data }, k, options, shifted.origin);
  let u = initialMembership({ ...x, data }, warm, k, options, m), unew = new Float64Array(n * k);
  const sums = new Float64Array(k), maxima = new Float64Array(k), newcenters = new Float64Array(k * d), dist = new Float64Array(k), history = [];
  const accum = { delta2: 0, objective: 0 }, objectiveState = { weakMax: -Infinity, weakSum: 0 };
  const cacheWeights = m !== 2;
  let iterations = 0, delta = Infinity, objective = Infinity, converged = false;
  for (iterations = 1; iterations <= maxIterations; ++iterations) {
    sums.fill(0); newcenters.fill(0);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      if (cacheWeights) {
        // The spare membership buffer can also hold the first block's powers.
        // Computing them here preserves the original cooperative boundary.
        if (iterations === 1) {
          const stop = Math.min(n, start + blockRows) * k;
          cacheMembershipPowers(u, unew, start * k, stop, m);
        }
        fcmWeightedCenterBlock(data, unew, sums, newcenters, start, Math.min(n, start + blockRows), d, k, m);
      } else fcmCenterBlock(data, u, sums, newcenters, start, Math.min(n, start + blockRows), d, k, m);
      yield { phase: 'centers', iteration: iterations, completedRows: Math.min(n, start + blockRows), totalRows: n };
    }
    for (let c = 0; c < k; ++c) {
      let rescale = sums[c] < MIN_NORMAL;
      // Heterogeneous columns may put their largest weight on a zero coordinate.
      // Inspect only endangered numerator coordinates, skipping constant zeros.
      if (!rescale) for (let f = 0; f < d && !rescale; ++f) {
        if (!nonzeroCoordinate[f] || Math.abs(newcenters[c * d + f]) >= MIN_NORMAL) continue;
        for (let i = 0; i < n; ++i) {
          const value = data[i * d + f], member = u[i * k + c];
          if (value === 0 || member === 0) continue;
          const weight = cacheWeights ? unew[i * k + c] : member * member;
          if (Math.abs(weight * value) < MIN_NORMAL) { rescale = true; break; }
        }
      }
      if (rescale || sums[c] <= n * centerWeightFloor) {
        let maximum = 0;
        for (let i = 0; i < n; ++i) maximum = Math.max(maximum, u[i * k + c]);
        maxima[c] = maximum;
        rescale ||= maximum ** m < centerWeightFloor;
      }
      if (rescale && maxima[c] > 0) {
        sums[c] = 0; newcenters.fill(0, c * d, (c + 1) * d);
        for (let i = 0; i < n; ++i) { const weight = (u[i * k + c] / maxima[c]) ** m; sums[c] += weight; for (let f = 0; f < d; ++f) newcenters[c * d + f] += weight * data[i * d + f]; }
      }
      if (sums[c] > 0) for (let f = 0; f < d; ++f) centers[c * d + f] = newcenters[c * d + f] / sums[c];
    }
    assertFinite(centers, 'centers');
    accum.delta2 = 0; accum.objective = 0; objectiveState.weakMax = -Infinity; objectiveState.weakSum = 0;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      fcmMembershipBlock(data, centers, u, unew, dist, start, Math.min(n, start + blockRows), d, k, m, accum, objectiveState, cacheWeights);
      yield { phase: 'membership', iteration: iterations, completedRows: Math.min(n, start + blockRows), totalRows: n };
    }
    [u, unew] = [unew, u]; objective = accum.objective + logObjectiveValue(objectiveState); delta = Math.sqrt(accum.delta2); converged = delta < tolerance;
    if (options.returnHistory) history.push(objective);
    checkpoint(options, { algorithm: 'fcm', centers, membership: u, membershipLayout: 'samples-clusters', iterations, converged, nSamples: n, nFeatures: d, nClusters: k, m, objective, delta, ...(options.returnHistory ? { objectiveHistory: Float64Array.from(history) } : {}) }, shifted.origin);
    const event = { algorithm: 'fcm', iteration: iterations, maxIterations, objective, delta, converged };
    progress(options, event); yield event;
    if (converged) break;
  }
  iterations = Math.min(iterations, maxIterations); restoreCenters(centers, shifted.origin);
  // Objective is recomputed on exactly the published centers, including rounding.
  // Cached U**m remains exact after the final swap and is independent of centers.
  objective = 0; objectiveState.weakMax = -Infinity; objectiveState.weakSum = 0; let fpc = 0;
  for (let i = 0; i < n; ++i) {
    if (i % blockRows === 0) { checkCancelled(options); yield { phase: 'finalize', completedRows: i, totalRows: n }; }
    if (d >= 16 && k >= 4) fullSquaredDistancesRow(x.data, i * d, centers, d, k, dist);
    else for (let c = 0; c < k; ++c) dist[c] = fullSquaredDistance(x.data, i * d, centers, c * d, d);
    for (let c = 0; c < k; ++c) { const v = u[i * k + c]; objective += objectiveTerm(v, dist[c], m, objectiveState, cacheWeights ? unew[i * k + c] : v * v); fpc += v * v; }
  }
  objective += logObjectiveValue(objectiveState);
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
  let coincident = d >= 16 && n * k * d >= 262144;
  if (coincident) for (let c = 1; c < k && coincident; ++c) for (let f = 0; f < d; ++f) {
    if (centers[c * d + f] !== centers[f]) { coincident = false; break; }
  }
  if (coincident) {
    // Every center has the same checked distance, so every mask bit qualifies.
    const value = 1 / k;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      const end = Math.min(n, start + blockRows);
      for (let i = start; i < end; ++i) fullSquaredDistance(data, i * d, centers, 0, d);
      out.fill(value, start * k, end * k); mask.fill(1, start * k, end * k);
      yield { phase: 'membership', iteration, completedRows: end, totalRows: n };
    }
    return n * k;
  }
  const squared = new Float64Array(k);
  let selectedTotal = 0;
  const alpha2 = alpha * alpha, beta2 = beta * beta;
  const useSquared = p === 2 && Number.isFinite(alpha2) && (beta === 0 || beta2 > 0);
  const simple = beta === 0 || p === 1;
  // A fast radius is only a filter: near-boundary values are re-evaluated
  // with the original canonical Float64 comparison, without widening it.
  const radiusFast = simple || (p >= .25 && p <= 64);
  const guard = 128 * Number.EPSILON * (1 + 1 / p);
  for (let start = 0; start < n; start += blockRows) {
    checkCancelled(options);
    for (let i = start; i < Math.min(n, start + blockRows); ++i) {
      let minSquared = Infinity;
      const io = i * d, off = i * k;
      if (d >= 16 && k >= 4) minSquared = fullSquaredDistancesRow(data, io, centers, d, k, squared);
      else for (let c = 0; c < k; ++c) {
        const distance = fullSquaredDistance(data, io, centers, c * d, d);
        squared[c] = distance;
        if (distance < minSquared) minSquared = distance;
      }
      const minimum = useSquared ? 0 : Math.sqrt(minSquared);
      let thresholdSquared;
      if (useSquared) thresholdSquared = alpha2 * minSquared + beta2;
      else if (radiusFast) {
        const a = alpha * minimum;
        let radius;
        if (simple) radius = beta === 0 ? a : a + beta;
        else {
          const scale = Math.max(a, beta), small = Math.min(a, beta);
          radius = scale === 0 ? 0 : scale * Math.exp(Math.log1p((small / scale) ** p) / p);
        }
        thresholdSquared = radius * radius;
      }
      let count = 0;
      for (let c = 0; c < k; ++c) {
        const sq = squared[c];
        let accepted;
        if (useSquared) accepted = sq <= thresholdSquared;
        else if (radiusFast && Number.isFinite(thresholdSquared) && thresholdSquared > 0
                 && Math.abs(sq - thresholdSquared) > guard * Math.max(sq, thresholdSquared)) accepted = sq <= thresholdSquared;
        else accepted = roughAdmissible(Math.sqrt(sq), minimum, alpha, beta, p);
        const value = accepted ? 1 : 0; mask[off + c] = value; count += value;
      }
      if (count === 0) throw new RangeError('no admissible rough cluster; numerical range unsupported');
      selectedTotal += count;
      for (let c = 0; c < k; ++c) out[off + c] = mask[off + c] / count;
    }
    yield { phase: 'membership', iteration, completedRows: Math.min(n, start + blockRows), totalRows: n };
  }
  return selectedTotal;
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
    const selectedTotal = yield* assignRough(data, centers, n, d, k, alpha, beta, p, options, blockRows, iterations, u, mask);
    const denseCenters = d >= 8 && selectedTotal >= 0.5 * n * k;
    const stableMask = sameMask(mask, previous.at(-1)?.mask);
    sums.fill(0); newcenters.fill(0);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      if (denseCenters) roughCenterBlock(data, u, sums, newcenters, start, Math.min(n, start + blockRows), d, k);
      else for (let i = start; i < Math.min(n, start + blockRows); ++i) for (let c = 0; c < k; ++c) {
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
