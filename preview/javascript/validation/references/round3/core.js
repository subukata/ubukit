/** Small shared helpers. No dependency or hidden global configuration. */
export function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
  return value;
}
export function finiteNumber(value, name, minimum = -Infinity) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum) throw new RangeError(`${name} must be finite and >= ${minimum}`);
  return value;
}
export function normalizeInput(input) {
  if (!input || typeof input !== 'object') throw new TypeError('input must be { data, nSamples, nFeatures }');
  const nSamples = positiveInteger(input.nSamples, 'nSamples');
  const nFeatures = positiveInteger(input.nFeatures, 'nFeatures');
  if (!Number.isSafeInteger(nSamples * nFeatures)) throw new RangeError('input dimensions exceed safe integer range');
  if (!ArrayBuffer.isView(input.data) || input.data instanceof DataView || typeof input.data[0] === 'bigint') throw new TypeError('data must be a numeric TypedArray in row-major order');
  if (input.data.length !== nSamples * nFeatures) throw new RangeError('data.length does not match nSamples * nFeatures');
  const data = input.data instanceof Float64Array ? input.data : Float64Array.from(input.data);
  assertFinite(data, 'data');
  return { data, nSamples, nFeatures };
}
export function assertFinite(data, name = 'array') {
  for (let i = 0; i < data.length; ++i) if (!Number.isFinite(data[i])) throw new RangeError(`${name}[${i}] must be finite; rescale if arithmetic overflowed`);
  return data;
}
/** Mulberry32, reproducible within JavaScript. Not NumPy's random generator. */
export function seededRandom(seed = 0) {
  if (!Number.isSafeInteger(seed)) throw new RangeError('seed must be a safe integer');
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function abortError(message = 'Computation cancelled') {
  const error = new Error(message); error.name = 'AbortError'; return error;
}
export function checkCancelled(options = {}) {
  if (options.signal?.aborted || options.shouldCancel?.()) throw abortError();
}
export function progress(options, event) {
  checkCancelled(options);
  if (typeof options.onProgress === 'function') options.onProgress(event);
}
export function squaredDistance(x, xo, y, yo, dimensions, cutoff = Infinity) {
  let s = 0, nonzero = false;
  // Direct differences retain the ||x||² term required by soft membership.
  for (let f = 0; f < dimensions; ++f) {
    const d = x[xo + f] - y[yo + f]; nonzero ||= d !== 0; s += d * d;
    if (s > cutoff) return s;
  }
  if (s > 0 && s < 2.2250738585072014e-308) throw new RangeError('squared distance is subnormal; rescale input');
  if (s === 0 && nonzero) throw new RangeError('squared distance underflow; rescale input');
  if (!Number.isFinite(s)) throw new RangeError('squared distance overflow; rescale input');
  return s;
}
/** Translation reduces loss in weighted means; no caller-owned array changes. */
export function translatedData(input) {
  const { data, nSamples, nFeatures } = input;
  const origin = data.slice(0, nFeatures), shifted = new Float64Array(data.length);
  for (let i = 0; i < nSamples; ++i) for (let f = 0; f < nFeatures; ++f) shifted[i * nFeatures + f] = data[i * nFeatures + f] - origin[f];
  assertFinite(shifted, 'translated data');
  return { data: shifted, origin };
}
export function restoreCenters(centers, origin) {
  for (let j = 0; j < centers.length; ++j) centers[j] += origin[j % origin.length];
  return assertFinite(centers, 'centers');
}
export function initializeCenters(input, nClusters, options, shiftedOrigin) {
  const { data, nSamples: n, nFeatures: d } = input;
  const centers = new Float64Array(nClusters * d);
  if (options.initCenters != null) {
    const init = options.initCenters;
    if (!ArrayBuffer.isView(init) || init instanceof DataView || init.length !== centers.length) throw new RangeError('initCenters must be a TypedArray of nClusters * nFeatures values');
    assertFinite(init, 'initCenters');
    for (let j = 0; j < init.length; ++j) centers[j] = init[j] - (shiftedOrigin?.[j % d] ?? 0);
  } else {
    if (nClusters > n) throw new RangeError('random sample initialization requires nClusters <= nSamples; provide initCenters for duplicates');
    const rng = seededRandom(options.seed ?? 0), indices = Uint32Array.from({ length: n }, (_, i) => i);
    for (let c = 0; c < nClusters; ++c) {
      const q = c + Math.floor(rng() * (n - c));
      const tmp = indices[c]; indices[c] = indices[q]; indices[q] = tmp;
      for (let f = 0; f < d; ++f) centers[c * d + f] = data[indices[c] * d + f];
    }
  }
  return assertFinite(centers, 'initial centers');
}
export function labelsFromMembership(u, n, k) {
  const labels = new Int32Array(n);
  for (let i = 0; i < n; ++i) {
    let best = 0;
    for (let c = 1; c < k; ++c) if (u[i * k + c] > u[i * k + best]) best = c;
    labels[i] = best;
  }
  return labels;
}
export function collectTransferables(value, seen = new Set()) {
  if (!value || typeof value !== 'object') return [...seen];
  if (value instanceof ArrayBuffer) seen.add(value);
  else if (ArrayBuffer.isView(value)) { if (value.buffer instanceof ArrayBuffer) seen.add(value.buffer); }
  else for (const child of Object.values(value)) collectTransferables(child, seen);
  return [...seen];
}
