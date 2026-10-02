import { abortError, collectTransferables } from './core.js';
/** One job at a time. Abort terminates its worker immediately and permits reuse. */
export function createWorkerClient({ workerFactory, workerUrl = new URL('./worker.js', import.meta.url) } = {}) {
  let worker = null, active = null, sequence = 0, disposed = false;
  const factory = workerFactory ?? (() => {
    if (typeof Worker === 'undefined') throw new Error('Web Worker unavailable; use runAsync or provide workerFactory');
    return new Worker(workerUrl, { type: 'module', name: 'ubukit-js' });
  });
  function stop() { const old = worker; worker = null; old?.terminate(); }
  function subscribe(target, type, callback) {
    if (typeof target.addEventListener === 'function') { target.addEventListener(type, callback); return () => target.removeEventListener(type, callback); }
    target.on(type, callback); return () => target.off(type, callback);
  }
  return {
    get busy() { return active != null; },
    run(algorithm, input, options = {}) {
      if (disposed) return Promise.reject(new Error('Worker client is disposed'));
      if (active) return Promise.reject(new Error('Worker client is busy; await or cancel the current job'));
      const { signal, onProgress, shouldCancel, transferInput = false, ...serializable } = options;
      if (shouldCancel != null) return Promise.reject(new TypeError('Worker callbacks cannot be serialized; use AbortSignal'));
      if (signal?.aborted) return Promise.reject(abortError());
      return new Promise((resolve, reject) => {
        const id = ++sequence, removers = [];
        let settled = false;
        const finish = (error, result, terminate = false) => {
          if (settled) return; settled = true;
          for (const remove of removers) remove();
          signal?.removeEventListener('abort', cancel); active = null;
          if (terminate) stop();
          if (error) reject(error); else resolve(result);
        };
        const cancel = () => finish(abortError(), null, true);
        active = { cancel };
        try {
          worker ??= factory();
          removers.push(subscribe(worker, 'message', event => {
            const message = typeof worker?.addEventListener === 'function' ? event.data : event;
            if (!message || message.id !== id) return;
            if (message.type === 'progress') { try { onProgress?.(message.progress); } catch (error) { finish(error, null, true); } }
            else if (message.type === 'result') finish(null, message.result);
            else if (message.type === 'error') { const error = new Error(message.error.message); error.name = message.error.name; error.stack = message.error.stack; finish(error); }
          }));
          removers.push(subscribe(worker, 'error', error => finish(new Error(error.message ?? 'Worker failed'), null, true)));
          if (typeof worker.addEventListener !== 'function') removers.push(subscribe(worker, 'exit', code => finish(new Error(`Worker exited before replying (code ${code})`), null, true)));
          signal?.addEventListener('abort', cancel, { once: true });
          // Only input buffers are transferred. Initialization/options stay usable.
          const transfers = transferInput ? collectTransferables(input) : [];
          worker.postMessage({ id, algorithm, input, options: serializable }, transfers);
        } catch (error) { finish(error, null, true); }
      });
    },
    cancel() { active?.cancel(); },
    dispose() { disposed = true; active?.cancel(); stop(); }
  };
}
