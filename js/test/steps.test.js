import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X } from './helpers.js';

test('progress results are the run stopped at that iteration', () => {
  // result() at iteration t equals the run with maxIter = t, even when called
  // after the loop has moved on (spreading the generator runs it to the end).
  // Four clusters for three blobs, so that every method takes several iterations.
  const converging = {
    kmeans: n => ub.steps.kmeans(X, 4, { seed: 1, maxIter: n }),
    fcm: n => ub.steps.fcm(X, 4, { seed: 1, maxIter: n }),
    efcm: n => ub.steps.efcm(X, 4, { seed: 1, tau: 0.5, maxIter: n }),
    rcm: n => ub.steps.rcm(X, 4, { seed: 1, maxIter: n }),
    rmcm: n => ub.steps.rmcm(X, 4, 0.5, { seed: 1, maxIter: n }),
    somOlp: n => ub.steps.somOlp(X, [3, 3], { lam: 0.5, gamma: 1, maxIter: n }),
  };
  for (const [name, start] of Object.entries(converging)) {
    const results = [...start(1000)].map(p => p.result());
    assert.ok(results.length > 1, name);
    for (const t of new Set([1, 2, results.length])) assert.deepEqual(results[t - 1], ub.run(start(t)), `${name} at ${t}`);
    assert.deepEqual(results.at(-1), ub.run(start(1000)), name);
  }
  // Scheduled maps: the last epoch is the result; earlier ones are not complete.
  for (const start of [() => ub.steps.som(X, [3, 3], { epochs: 3, seed: 2 }), () => ub.steps.batchSom(X, [3, 3], { epochs: 4 })]) {
    const results = [...start()].map(p => p.result());
    assert.deepEqual(results.at(-1), ub.run(start()));
    assert.ok(results.slice(0, -1).every(r => !r.converged) && results.at(-1).converged);
  }
});

test('runAsync reports progress and honors abort', async () => {
  const seen = [];
  const r = await ub.runAsync(ub.steps.fcm(X, 3, { seed: 0 }), { onProgress: p => seen.push(p.iteration) });
  assert.equal(seen.length, r.nIter);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(ub.runAsync(ub.steps.kmeans(X, 3), { signal: controller.signal }), { name: 'AbortError' });
});
