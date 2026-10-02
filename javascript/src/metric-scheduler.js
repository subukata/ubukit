import { neighborhoodSteps } from './neighborhood.js';
import { abortError, assertFinite, finiteNumber, positiveInteger } from './core.js';
import { cloneOwned, sameArray } from './session-validation.js';
function equal(a, b) {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) return ArrayBuffer.isView(a) && ArrayBuffer.isView(b) && a.constructor === b.constructor && sameArray(a, b);
  const ak = Object.keys(a), bk = Object.keys(b);
  return ak.length === bk.length && ak.every(k => Object.hasOwn(b, k) && equal(a[k], b[k]));
}
// Numerical storage copied by cloneOwned; JS objects/runtime overhead are not
// a heap cap. A separately cloned input/embedding is charged separately.
function ownedBytes(value, seen = new Set()) {
  if (!value || typeof value !== 'object') return 0;
  if (value instanceof ArrayBuffer) return value.byteLength;
  if (seen.has(value)) return 0;
  seen.add(value);
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) return value.byteLength;
  if (typeof value.addEventListener === 'function' && 'aborted' in value) return 0;
  return Object.values(value).reduce((sum, child) => sum + ownedBytes(child, seen), 0);
}
// Match normalizeInput validation without allocating a Float64 conversion before
// the scheduler has checked its reserve and released superseded storage.
function inspectInput(input) {
  if (!input || typeof input !== 'object') throw new TypeError('input must be { data, nSamples, nFeatures }');
  const nSamples = positiveInteger(input.nSamples, 'nSamples');
  const nFeatures = positiveInteger(input.nFeatures, 'nFeatures');
  if (!Number.isSafeInteger(nSamples * nFeatures)) throw new RangeError('input dimensions exceed safe integer range');
  if (!ArrayBuffer.isView(input.data) || input.data instanceof DataView || typeof input.data[0] === 'bigint') throw new TypeError('data must be a numeric TypedArray in row-major order');
  if (input.data.length !== nSamples * nFeatures) throw new RangeError('data.length does not match nSamples * nFeatures');
  assertFinite(input.data, 'data');
  return { data: input.data, nSamples, nFeatures };
}
function ownInput(value) {
  return { ...value, data: new Float64Array(value.data) };
}
/** Exact one-shot neighborhood metrics: debounce, cooperative cancellation, and
 * one content-validated result cache. No invented iterative fitting semantics.
 */
export function createMetricScheduler({ debounceMs = 50, timeBudgetMs = 8, maxChunks = 8, maxMemoryBytes = 512 * 1024 ** 2 } = {}) {
  finiteNumber(debounceMs, 'debounceMs', 0); finiteNumber(timeBudgetMs, 'timeBudgetMs', 0);
  positiveInteger(maxChunks, 'maxChunks'); finiteNumber(maxMemoryBytes, 'maxMemoryBytes', 1);
  let revision = 0, current = null, executing = null, cache = null, disposed = false;
  function release(request) {
    request.iterator?.return?.(); request.iterator = null;
    request.input = null; request.options = null; request.reservedBytes = 0;
  }
  function stop(request, error) {
    if (!request || request.settled) return;
    request.settled = true; clearTimeout(request.timer); if (!request.running) release(request); request.signal?.removeEventListener('abort', request.abort);
    if (current === request) current = null;
    request.reject(error);
  }
  function pump(request) {
    if (current !== request || request.settled) return;
    const start = performance.now(); let chunks = 0;
    try {
      do {
        if (request.signal?.aborted) throw abortError();
        let next;
        request.running = true; executing = request;
        try { next = request.iterator.next(); } finally { request.running = false; executing = null; }
        ++chunks;
        if (request.settled || current !== request) { release(request); return; }
        if (next.done) {
          cache = { input: request.input, options: request.options, bytes: request.ownedBytes, result: cloneOwned(next.value) };
          request.settled = true; current = null; request.signal?.removeEventListener('abort', request.abort);
          release(request);
          request.resolve({ requestId: request.id, cacheHit: false, result: next.value }); return;
        }
      } while (chunks < maxChunks && performance.now() - start < timeBudgetMs);
      request.timer = setTimeout(() => pump(request), 0);
    } catch (error) { stop(request, error); }
  }
  return {
    get status() { return { revision, busy: current != null, cached: cache != null, disposed }; },
    run(input, options = {}) {
      if (disposed) return Promise.reject(new Error('Metric scheduler is disposed'));
      const { signal, onProgress, shouldCancel, ...values } = options;
      if (signal?.aborted) return Promise.reject(abortError());
      let x, owned, bytes, reservedBytes;
      try {
        const embeddingKey = values.embedding != null ? 'embedding' : 'coordinates';
        const sourceY = values[embeddingKey];
        const normalized = inspectInput(input), y = inspectInput(sourceY);
        const otherValues = { ...values }; delete otherValues[embeddingKey];
        const scratchBytes = positiveInteger(values.maxScratchBytes ?? 32 * 1024 ** 2, 'maxScratchBytes');
        bytes = 8 * (normalized.data.length + y.data.length) + ownedBytes(otherValues);
        reservedBytes = bytes + scratchBytes;
        // A generator executing an application callback cannot yet be closed.
        // Charge it even if cancel() has already cleared `current`.
        const retainedBytes = executing?.reservedBytes ?? 0;
        if (!Number.isSafeInteger(reservedBytes + retainedBytes) || reservedBytes + retainedBytes > maxMemoryBytes) throw new RangeError('Metric cache/input reserve exceeds maxMemoryBytes');
        stop(current, abortError('Superseded by newer metric input'));
        if (cache && cache.bytes + reservedBytes + retainedBytes > maxMemoryBytes) cache = null;
        // Normalize and own each selected input once, after the reserve check.
        x = ownInput(normalized);
        owned = cloneOwned(otherValues); owned[embeddingKey] = ownInput(y);
      } catch (error) { return Promise.reject(error); }
      const id = ++revision;
      if (cache && equal(cache.input, x) && equal(cache.options, owned)) return Promise.resolve({ requestId: id, cacheHit: true, result: cloneOwned(cache.result) });
      return new Promise((resolve, reject) => {
        const request = { id, input: x, options: owned, ownedBytes: bytes, reservedBytes, signal, resolve, reject, settled: false, timer: null, iterator: null };
        current = request; request.abort = () => stop(request, abortError());
        signal?.addEventListener('abort', request.abort, { once: true });
        request.iterator = neighborhoodSteps(x, { ...owned, signal, shouldCancel: () => current !== request || shouldCancel?.(),
          onProgress: event => { if (current === request) onProgress?.({ ...event, requestId: id }); } });
        request.timer = setTimeout(() => pump(request), debounceMs);
      });
    },
    cancel() { stop(current, abortError()); },
    clearCache() { cache = null; },
    dispose() { if (disposed) return; disposed = true; stop(current, abortError('Metric scheduler disposed')); cache = null; }
  };
}
