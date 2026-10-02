// Run from any directory: node javascript/examples/realtime.js
import { createSession, createRealtimeWorkerClient, createMetricScheduler } from '../../javascript/src/index.js';
import { Worker } from 'node:worker_threads';
const x = { data: Float64Array.of(0, 0, 0, 1, 1, 0, 8, 8, 8, 9, 9, 8), nSamples: 6, nFeatures: 2 };
const options = { nClusters: 2, m: 2, seed: 7, maxIterations: 8, tolerance: 0, blockRows: 2 };
const session = createSession('fcm', x, options);
const drain = async () => {
  while (!session.status.done) {
    session.step(1, { timeBudgetMs: 2, maxChunks: 8 });
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return session.snapshot();
};
try {
  await drain();
  session.updateParameters({ m: 2.5 });
  await drain();
  // Remove a point and move the remaining rows. Membership is reinitialized;
  // centers are retained, and no assumption about stable row identity is made.
  session.updateData({ data: Float64Array.of(.1, 0, .1, 1, 8.1, 8, 8.1, 9, 9.1, 8), nSamples: 5, nFeatures: 2 });
  const result = await drain();
  console.log('session', result.revision, result.result.centers, result.result.membership.length);
} finally { session.dispose(); }
const client = createRealtimeWorkerClient({
  workerFactory: () => new Worker(new URL('../../javascript/src/realtime-worker.js', import.meta.url), { type: 'module' })
});
try {
  const old = client.run('fcm', x, options).catch(error => error.name);
  const latest = client.run('fcm', x, { ...options, m: 3 });
  console.log('superseded', await old);
  const snapshot = await latest;
  console.log('worker', snapshot.requestId, snapshot.result.m);
} finally { client.dispose(); }
const metrics = createMetricScheduler({ debounceMs: 0 });
try {
  const { result } = await metrics.run(x, { embedding: x, k: 1 });
  console.log('identity metrics', result.qualities);
} finally { metrics.dispose(); }
