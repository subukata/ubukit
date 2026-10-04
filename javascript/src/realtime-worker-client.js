import { abortError } from './core.js';
import { cloneOwned } from './session-validation.js';

/** Persistent Worker, one dispatched request + one coalesced latest input.
 * Superseded promises reject AbortError; stale progress/results are never sent
 * to application callbacks. run() accepts a complete option set, not a patch.
 */
export function createRealtimeWorkerClient({ workerFactory, workerUrl = new URL('./realtime-worker.js', import.meta.url) } = {}) {
  let worker = null, active = null, pending = null, sequence = 0, disposed = false;
  const factory = workerFactory ?? (() => {
    if (typeof Worker === 'undefined') throw new Error('Web Worker unavailable');
    return new Worker(workerUrl, { type: 'module', name: 'ubukit-realtime' });
  });
  const subscribe = (target, type, callback) => {
    if (typeof target.addEventListener === 'function') { target.addEventListener(type, callback); return () => target.removeEventListener(type, callback); }
    target.on(type, callback); return () => target.off(type, callback);
  };
  let removers = [];
  function settle(request, error, result) {
    if (!request || request.settled) return;
    request.settled = true; request.signal?.removeEventListener('abort', request.abort);
    if (error) request.reject(error); else request.resolve(result);
  }
  function destroy(error) {
    const old = worker; worker = null;
    for (const remove of removers) remove(); removers = [];
    settle(active, error); settle(pending, error); active = null; pending = null; old?.terminate();
  }
  function dispatch() {
    if (active || !pending || disposed) return;
    active = pending; pending = null;
    try {
      if (!worker) {
        worker = factory();
        const current = worker;
        removers.push(subscribe(current, 'message', event => {
          if (worker !== current) return;
          const message = typeof current.addEventListener === 'function' ? event.data : event;
          if (!message || message.id !== active?.id) return;
          if (message.type === 'progress') {
            if (!active.settled && active.id === sequence) {
              const request = active;
              try { request.onProgress?.(message.snapshot); }
              catch (error) {
                settle(request, error);
                if (worker === current && active === request && !request.cancelSent) {
                  request.cancelSent = true;
                  try { current.postMessage({ type: 'cancel', id: request.id }); }
                  catch (postError) { destroy(postError); }
                }
              }
            }
            return;
          }
          const request = active; active = null;
          if (message.type === 'result' && request.id === sequence) settle(request, null, message.snapshot);
          else if (message.type === 'error') { const e = new Error(message.error.message); e.name = message.error.name; settle(request, e); }
          else settle(request, abortError('Computation superseded or cancelled'));
          dispatch();
        }));
        removers.push(subscribe(current, 'error', e => destroy(new Error(e.message ?? 'Realtime Worker failed'))));
        if (typeof current.addEventListener !== 'function') removers.push(subscribe(current, 'exit', code => { if (worker === current) destroy(new Error(`Realtime Worker exited (${code})`)); }));
      }
      const request = active;
      worker.postMessage({ type: 'run', id: request.id, algorithm: request.algorithm, input: request.input, options: request.options, controls: request.controls, reportProgress: request.onProgress != null });
      // postMessage cloned the payload; do not keep an extra active input copy.
      request.input = null; request.options = null;
    } catch (error) { destroy(error); }
  }
  function cancelActive(reason) {
    if (!active) return;
    settle(active, abortError(reason));
    if (!active.cancelSent) { active.cancelSent = true; worker?.postMessage({ type: 'cancel', id: active.id }); }
  }
  return {
    get busy() { return active != null || pending != null; },
    get revision() { return sequence; },
    run(algorithm, input, options = {}, controls = {}) {
      if (disposed) return Promise.reject(new Error('Realtime Worker client is disposed'));
      const { signal, onProgress, shouldCancel, transferInput, ...serializable } = options;
      if (shouldCancel != null || transferInput) return Promise.reject(new TypeError('Realtime requests own input snapshots; use AbortSignal, not shouldCancel/transferInput'));
      if (signal?.aborted) return Promise.reject(abortError());
      let owned;
      try { owned = { input: cloneOwned(input), options: cloneOwned(serializable), controls: cloneOwned(controls) }; }
      catch (error) { return Promise.reject(error); }
      const id = ++sequence;
      return new Promise((resolve, reject) => {
        settle(pending, abortError('Superseded by a newer input')); pending = null;
        cancelActive('Superseded by a newer input');
        const request = { id, algorithm, ...owned, resolve, reject, onProgress, signal, settled: false, cancelSent: false };
        request.abort = () => {
          if (pending === request) { pending = null; settle(request, abortError()); }
          else if (active === request) cancelActive('Computation cancelled');
        };
        signal?.addEventListener('abort', request.abort, { once: true });
        pending = request; dispatch();
      });
    },
    cancel() { settle(pending, abortError()); pending = null; cancelActive('Computation cancelled'); },
    dispose() { if (disposed) return; disposed = true; destroy(abortError('Realtime Worker client disposed')); }
  };
}
