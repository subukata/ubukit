import { validateSOM } from './som.js';
import { needsStableFCM } from './fcm-stable.js';
import { SESSION_FCM_STATE } from './session-hooks.js';
import { normalizeInput, positiveInteger, finiteNumber, seededRandom, assertFinite } from './core.js';

export const sessionAlgorithms = Object.freeze(['kmeans', 'fcm', 'rcm', 'exrcm', 'rmcm', 'som-olp', 'som', 'som_batch']);
export function cloneOwned(value, seen = new Map()) {
  if (value == null || typeof value !== 'object') return value;
  if (seen.has(value)) return seen.get(value);
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    // Call the intrinsic TypedArray slice: Node Buffer.slice() aliases storage.
    const copy = Uint8Array.prototype.slice.call(value); seen.set(value, copy); return copy;
  }
  if (value instanceof ArrayBuffer) return value.slice(0);
  // AbortSignal remains a live cancellation control, not numerical state.
  if (typeof value.addEventListener === 'function' && 'aborted' in value) return value;
  const copy = Array.isArray(value) ? [] : {}; seen.set(value, copy);
  for (const key of Object.keys(value)) {
    // Defining own data avoids inherited setters such as Object.prototype.__proto__.
    Object.defineProperty(copy, key, {
      value: cloneOwned(value[key], seen), enumerable: true, writable: true, configurable: true,
    });
  }
  return copy;
}
export function sameArray(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let j = 0; j < a.length; ++j) if (a[j] !== b[j]) return false;
  return true;
}
function array(value, size, name, { nonnegative = false, probabilities = false } = {}) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView || typeof value[0] === 'bigint' || value.length !== size) throw new RangeError(`${name} must be a numeric TypedArray of length ${size}`);
  assertFinite(value, name);
  if (nonnegative || probabilities) for (const x of value) if (x < 0) throw new RangeError(`${name} must be nonnegative`);
}
function nonnegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a nonnegative safe integer`);
  return value;
}
// Argument validation only. Numerical over/underflow may still fail while stepping.
export function validateSession(algorithm, input, options, warmFCM = null) {
  if (!sessionAlgorithms.includes(algorithm)) throw new RangeError(`Stateful fitting requires ${sessionAlgorithms.join(', ')}; neighborhood is a one-shot metric`);
  if (algorithm === 'som' || algorithm === 'som_batch') {
    const cfg = validateSOM(input, options, algorithm === 'som_batch');
    let optionBytes = 0;
    for (const value of Object.values(options)) if (ArrayBuffer.isView(value)) optionBytes += value.byteLength;
    const reserveBytes = 8 * cfg.n * cfg.d + optionBytes + 2 * (8 * cfg.k * cfg.d + 20 * cfg.n);
    const kernelMemoryBytes = cfg.maxMemoryBytes - reserveBytes;
    if (!Number.isSafeInteger(reserveBytes) || cfg.estimatedPrimaryBytes > kernelMemoryBytes) throw new RangeError('Session owned input and checkpoint state exceed maxMemoryBytes');
    return { ...cfg, reserveBytes, kernelMemoryBytes };
  }
  const x = normalizeInput(input), n = x.nSamples, d = x.nFeatures;
  seededRandom(options.seed ?? 0);
  positiveInteger(options.blockRows ?? (algorithm === 'som-olp' || algorithm === 'rmcm' ? 128 : 512), 'blockRows');
  const maxIterations = algorithm === 'som-olp' ? nonnegativeInteger(options.maxIterations ?? 100, 'maxIterations') : positiveInteger(options.maxIterations ?? 100, 'maxIterations');
  const maxMemoryBytes = algorithm === 'som-olp'
    ? positiveInteger(options.maxMemoryBytes ?? 512 * 1024 ** 2, 'maxMemoryBytes')
    : finiteNumber(options.maxMemoryBytes ?? 512 * 1024 ** 2, 'maxMemoryBytes', 1);
  finiteNumber(options.tolerance ?? 0, 'tolerance', 0);
  if (algorithm === 'kmeans' && options.wasmCenterCache != null && typeof options.wasmCenterCache !== 'boolean') throw new TypeError('wasmCenterCache must be a boolean');
  // Validate execution options before an update abandons a committed revision.
  // These checks must also run when an existing prepared graph is reused.
  if (['kmeans', 'som-olp'].includes(algorithm) && !['javascript', 'wasm'].includes(options.kernelBackend ?? 'javascript')) throw new RangeError("kernelBackend must be 'javascript' or 'wasm'");
  let k, q = 0, grid = null;
  if (algorithm === 'som-olp') {
    grid = normalizeInput(options.grid); k = grid.nSamples; q = grid.nFeatures;
    finiteNumber(options.gamma ?? 1, 'gamma', 0);
    finiteNumber(options.lambda ?? options.lam ?? .1, 'lambda', Number.MIN_VALUE);
    positiveInteger(options.maxScratchBytes ?? 32 * 1024 ** 2, 'maxScratchBytes');
    positiveInteger(options.pcaMaxDimension ?? 128, 'pcaMaxDimension');
    positiveInteger(options.pcaMaxSweeps ?? 80, 'pcaMaxSweeps');
    finiteNumber(options.pcaTolerance ?? 1e-13, 'pcaTolerance', 0);
    finiteNumber(options.pcaScale ?? 2, 'pcaScale');
    const w = options.initialPrototypes ?? options.initCenters;
    const p = options.initialMemberships ?? options.initMembership;
    if (w != null) array(w, k * d, 'initialPrototypes');
    else {
      if (!['pca', 'sample'].includes(options.initializer ?? 'pca')) throw new RangeError("initializer must be 'pca' or 'sample'");
      if ((options.initializer ?? 'pca') === 'pca') {
        if (Math.min(q, d) > Math.min(n, d)) throw new RangeError('Too few samples for PCA grid rank');
        if (Math.min(n, d) > (options.pcaMaxDimension ?? 128)) throw new RangeError('PCA dimension exceeds pcaMaxDimension');
      }
    }
    if (p != null) {
      if (w == null) throw new RangeError('initialMemberships requires initialPrototypes');
      array(p, n * k, 'initialMemberships', { nonnegative: true });
      for (let i = 0; i < n; ++i) { let sum = 0; for (let c = 0; c < k; ++c) sum += p[i * k + c]; if (Math.abs(sum - 1) > 1e-8) throw new RangeError('Each initialMemberships row must sum to one within 1e-8'); }
    }
  } else {
    k = positiveInteger(options.nClusters ?? (options.initCenters?.length / d || (algorithm === 'fcm' && options.initMembership?.length / n)), 'nClusters');
    if (k > 0x7fffffff) throw new RangeError('nClusters exceeds Int32 range');
    if (options.initCenters != null) array(options.initCenters, k * d, 'initCenters');
    if (algorithm !== 'fcm' && options.initCenters == null && k > n) throw new RangeError('sample initialization requires nClusters <= nSamples');
    if (algorithm === 'kmeans' && (options.tolerance ?? 0) !== 0) throw new RangeError('kmeans requires tolerance=0');
    if (algorithm === 'fcm') {
      finiteNumber(options.m ?? 2, 'm', 1 + Number.EPSILON);
      if (options.initCenters != null && options.initMembership != null) throw new RangeError('choose initCenters or initMembership, not both');
      if (options.initMembership != null) {
        const u = options.initMembership; array(u, n * k, 'initMembership', { nonnegative: true });
        for (let i = 0; i < n; ++i) { let max = 0; for (let c = 0; c < k; ++c) max = Math.max(max, u[i * k + c]); if (!(max > 0)) throw new RangeError('every initMembership row needs a positive sum'); }
      }
    }
    if (algorithm === 'rcm' || algorithm === 'exrcm') {
      finiteNumber(options.alpha ?? 1.1, 'alpha', 1); finiteNumber(options.beta ?? 0, 'beta', 0);
      finiteNumber(options.p ?? 1, 'p', Number.MIN_VALUE);
      if (algorithm === 'rcm' && options.p != null && options.p !== 1) throw new RangeError('rcm fixes p=1');
      nonnegativeInteger(options.cycleWindow ?? 16, 'cycleWindow');
    }
    if (algorithm === 'rmcm') {
      for (const key of ['m', 'alpha', 'beta', 'p', 'excludeSelf', 'includeSelf', 'initMembership']) if (Object.hasOwn(options, key)) throw new RangeError(`${key} is not an RMCM parameter`);
      finiteNumber(options.delta, 'delta', 0);
      if ((options.tolerance ?? 0) !== 0) throw new RangeError('RMCM requires tolerance=0');
      if (k > n) throw new RangeError('RMCM nClusters cannot exceed nSamples');
      if (!['adjoint', 'reference'].includes(options.backend ?? 'adjoint')) throw new RangeError('Unknown RMCM backend');
      if (!['scalar', 'wasm-simd', 'grid'].includes(options.graphBackend ?? 'scalar')) throw new RangeError("graphBackend must be 'scalar', 'wasm-simd', or 'grid'");
      nonnegativeInteger(options.cycleWindow ?? 32, 'cycleWindow');
      const maxEdges = positiveInteger(options.maxEdges ?? 10_000_000, 'maxEdges');
      if (maxEdges > 0xffffffff || n > maxEdges || n > 0xffffffff) throw new RangeError('RMCM edge/index limit invalid');
      positiveInteger(options.graphBatchPairs ?? 4096, 'graphBatchPairs');
      if (options.returnMembership != null && typeof options.returnMembership !== 'boolean') throw new TypeError('returnMembership must be boolean');
      const bound = Math.sqrt(Number.MAX_VALUE / d) / 4;
      for (const a of [x.data, options.initCenters]) if (a) for (const v of a) if (Math.abs(v) > bound) throw new RangeError('RMCM coordinates exceed safe float64 range');
    }
  }
  if (![n * k, k * d, n * q].every(Number.isSafeInteger)) throw new RangeError('Session shape exceeds safe integer range');
  // Reserve owned X/options and one stable committed checkpoint, in addition to
  // the kernel's own checked arrays. Caller-owned snapshot copies are excluded.
  let optionBytes = 0;
  for (const value of Object.values(options)) {
    if (ArrayBuffer.isView(value)) optionBytes += value.byteLength;
    else if (value?.data && ArrayBuffer.isView(value.data)) optionBytes += value.data.byteLength;
  }
  // Match the actual warm kernel's initialization when selecting its memory
  // estimate. Keep the configured options above for owned-storage accounting.
  const executionOptions = algorithm === 'fcm' && warmFCM?.membership
    ? { ...options, initCenters: undefined, initMembership: warmFCM.membership, [SESSION_FCM_STATE]: warmFCM[SESSION_FCM_STATE] }
    : options;
  const stableFCM = algorithm === 'fcm' && needsStableFCM(x, executionOptions, options.m ?? 2);
  const checkpointBytes = (algorithm === 'fcm' ? 8*n*k+8*n : 0) + 8 * k * d + 8 * n * k + 8 * n * q + 12 * n + 8 * maxIterations;
  const reserveBytes = 8 * n * d + optionBytes + checkpointBytes * 2;
  const kernelMemoryBytes = maxMemoryBytes - reserveBytes;
  const minimumKernelBytes = 16 * n * d + 16 * n * k + 32 * k * d;
  if (!Number.isSafeInteger(reserveBytes) || kernelMemoryBytes < minimumKernelBytes) throw new RangeError('Session owned input and checkpoint state exceed maxMemoryBytes');
  let primary;
  if (algorithm === 'som-olp') {
    const train = 8 * (k * d + 2 * k);
    const pcaDim = Math.min(n, d), hasW = (options.initialPrototypes ?? options.initCenters) != null;
    const pca = !hasW && (options.initializer ?? 'pca') === 'pca' ? 16 * pcaDim * pcaDim + 8 * (2 * d + 2 * q) + 4 * pcaDim : 0;
    if (Math.max(train, pca) > (options.maxScratchBytes ?? 32 * 1024 ** 2)) throw new RangeError('SOM scratch exceeds maxScratchBytes');
    primary = 8 * (n * d + k * q + k * d + n * k + (maxIterations > 0 ? n * q : 0) + maxIterations) + 4 * n + Math.max(train, pca);
  } else if (algorithm === 'rmcm') {
    const adjoint = (options.backend ?? 'adjoint') === 'adjoint';
    const fixed = 8 * n * d + 12 * n + 4 + (adjoint ? 8 * n * d + 8 * n : 0) + 8 * d;
    const histories = Math.min(options.cycleWindow ?? 32, maxIterations) + 1;
    // Minimum mandatory self graph; actual edge-dependent bound is checked as
    // the graph is built. A dense graph can still fail operationally in step().
    primary = fixed + 4 * n + 8 * n + 24 * k * d + 8 * k + histories * (4 * n + 8 * k * d) + ((!adjoint || (options.returnMembership ?? true)) ? 8 * n * k : 0);
  } else if (algorithm === 'rcm' || algorithm === 'exrcm') {
    const window = Math.max(1, options.cycleWindow ?? 16);
    primary = 16 * n * d + (9 + window) * n * k + (24 + 8 * window) * k * d + 16 * k;
  } else primary = 16 * n * d + (algorithm === 'fcm' ? 16 * n * k + d : 8 * n) + 24 * k * d + 32 * k;
  if (stableFCM) primary = 16*n*d+24*n*k+328*n+32*k*d+48*k+16*d+8*maxIterations;
  if (!Number.isSafeInteger(primary) || primary > kernelMemoryBytes) throw new RangeError('Session kernel arrays exceed maxMemoryBytes after checkpoint reserve');
  return { n, d, k, q, grid, maxIterations, reserveBytes, kernelMemoryBytes };
}
