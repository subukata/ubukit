import { createSession } from './session.js';
import { collectTransferables, finiteNumber, positiveInteger } from './core.js';
let listen, send;
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  listen = fn => self.addEventListener('message', event => fn(event.data));
  send = (message, transfers) => self.postMessage(message, transfers);
} else {
  const { parentPort } = await import('node:worker_threads');
  if (!parentPort) throw new Error('realtime-worker.js must execute in a Worker');
  listen = fn => parentPort.on('message', fn); send = (message, transfers) => parentPort.postMessage(message, transfers);
}
let session = null, algorithm = null, job = null, timer = null;
function postSnapshot(type, target) {
  const snapshot = { ...session.snapshot(), requestId: target.id };
  send({ id: target.id, type, snapshot }, collectTransferables(snapshot));
}
function pump() {
  timer = null;
  const current = job;
  if (!current) return;
  try {
    session.step(current.iterationsPerSlice, { timeBudgetMs: current.timeBudgetMs, maxChunks: current.maxChunks });
    if (session.status.done) { postSnapshot('result', current); job = null; return; }
    if (current.reportProgress) {
      const now = performance.now();
      if (session.status.iteration > current.lastIteration && now - current.lastProgress >= current.progressIntervalMs) {
        postSnapshot('progress', current); current.lastProgress = now; current.lastIteration = session.status.iteration;
      }
    }
    timer = setTimeout(pump, 0);
  } catch (error) {
    send({ id: current.id, type: 'error', error: { name: error.name, message: error.message } }); job = null;
  }
}
listen(message => {
  if (!message || typeof message !== 'object') return;
  if (message.type === 'cancel') {
    if (job?.id === message.id) { if (timer != null) clearTimeout(timer); timer = null; job = null; }
    // Keep the last complete state and prepared graph. A subsequent configure
    // abandons any partial iteration before applying the new immutable revision.
    send({ id: message.id, type: 'cancelled' }); return;
  }
  if (message.type !== 'run') return;
  const { id, input, options = {}, controls = {} } = message;
  try {
    const timeBudgetMs = finiteNumber(controls.timeBudgetMs ?? 8, 'timeBudgetMs', 0);
    const maxChunks = positiveInteger(controls.maxChunks ?? 128, 'maxChunks');
    const iterationsPerSlice = positiveInteger(controls.iterationsPerSlice ?? 1, 'iterationsPerSlice');
    const progressIntervalMs = finiteNumber(controls.progressIntervalMs ?? 30, 'progressIntervalMs', 0);
    const warmStart = controls.warmStart ?? true;
    if (typeof warmStart !== 'boolean') throw new TypeError('warmStart must be boolean');
    if (session && algorithm === message.algorithm) session.configure(input, options, { warmStart });
    else { const next = createSession(message.algorithm, input, options); session?.dispose(); session = next; algorithm = message.algorithm; }
    // Older direct worker clients omit the flag and retain progress by default.
    job = { id, reportProgress: message.reportProgress !== false, timeBudgetMs, maxChunks, iterationsPerSlice, progressIntervalMs, lastProgress: -Infinity, lastIteration: 0 };
    if (timer != null) clearTimeout(timer); timer = setTimeout(pump, 0);
  } catch (error) { send({ id, type: 'error', error: { name: error.name, message: error.message } }); }
});
