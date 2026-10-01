import { run } from './index.js';
import { collectTransferables } from './core.js';
let listen, send;
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  listen = callback => self.addEventListener('message', event => callback(event.data));
  send = (message, transfers) => self.postMessage(message, transfers);
} else {
  // Built-in Node module only; browsers do not execute this branch.
  const { parentPort } = await import('node:worker_threads');
  if (!parentPort) throw new Error('worker.js must run as a worker');
  listen = callback => parentPort.on('message', callback);
  send = (message, transfers) => parentPort.postMessage(message, transfers);
}
listen(({ id, algorithm, input, options }) => {
  try {
    let lastProgress = -Infinity;
    const result = run(algorithm, input, { ...options, onProgress: event => {
      const now = performance.now();
      if (now - lastProgress >= 30 || event.converged || event.iteration === event.maxIterations) { send({ id, type: 'progress', progress: event }); lastProgress = now; }
    } });
    send({ id, type: 'result', result }, collectTransferables(result));
  } catch (error) { send({ id, type: 'error', error: { name: error.name, message: error.message, stack: error.stack } }); }
});
