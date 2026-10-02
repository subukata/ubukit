import { neighborhoodSteps } from './neighborhood.js';
import { abortError, finiteNumber, normalizeInput, positiveInteger } from './core.js';
import { cloneOwned, sameArray } from './session-validation.js';
function equal(a, b) {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) return ArrayBuffer.isView(a) && ArrayBuffer.isView(b) && a.constructor === b.constructor && sameArray(a, b);
  const ak = Object.keys(a), bk = Object.keys(b);
  return ak.length === bk.length && ak.every(k => Object.hasOwn(b, k) && equal(a[k], b[k]));
}
/** Exact one-shot neighborhood metrics: debounce, cooperative cancellation, and
 * one content-validated result cache. No invented iterative fitting semantics.
 */
export function createMetricScheduler({ debounceMs = 50, timeBudgetMs = 8, maxChunks = 8, maxMemoryBytes = 512 * 1024 ** 2 } = {}) {
  finiteNumber(debounceMs, 'debounceMs', 0); finiteNumber(timeBudgetMs, 'timeBudgetMs', 0);
  positiveInteger(maxChunks, 'maxChunks'); finiteNumber(maxMemoryBytes, 'maxMemoryBytes', 1);
  let revision = 0, current = null, cache = null, disposed = false;
  function stop(request, error) {
    if (!request || request.settled) return;
    request.settled = true; clearTimeout(request.timer); if (!request.running) request.iterator?.return?.(); request.signal?.removeEventListener('abort', request.abort);
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
        request.running = true;
        try { next = request.iterator.next(); } finally { request.running = false; }
        ++chunks;
        if (request.settled || current !== request) { request.iterator.return?.(); return; }
        if (next.done) {
          cache = { input: request.input, options: request.options, result: cloneOwned(next.value) };
          request.settled = true; current = null; request.signal?.removeEventListener('abort', request.abort);
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
      let x, owned;
      try {
        const normalized = normalizeInput(input), y = normalizeInput(values.embedding ?? values.coordinates);
        // Two cached/pending owned pairs plus row scratch reserve. Returned
        // snapshots retained by the caller are outside this internal budget.
        const bytes = 16 * (normalized.data.byteLength + y.data.byteLength) / 8 + (values.maxScratchBytes ?? 32 * 1024 ** 2);
        if (bytes > maxMemoryBytes) throw new RangeError('Metric cache/input reserve exceeds maxMemoryBytes');
        x = cloneOwned(normalized); owned = cloneOwned(values);
      } catch (error) { return Promise.reject(error); }
      const id = ++revision;
      stop(current, abortError('Superseded by newer metric input'));
      if (cache && equal(cache.input, x) && equal(cache.options, owned)) return Promise.resolve({ requestId: id, cacheHit: true, result: cloneOwned(cache.result) });
      return new Promise((resolve, reject) => {
        const request = { id, input: x, options: owned, signal, resolve, reject, settled: false, timer: null, iterator: null };
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
