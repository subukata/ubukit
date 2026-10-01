// Run from the repository root: node examples/javascript.mjs
import assert from 'node:assert/strict';
import { run } from '../javascript/src/index.js';

// Row-major flat storage: (N=6, D=2).
const input = {
  data: Float64Array.of(0, 0, 0, 1, 1, 0, 8, 8, 8, 9, 9, 8),
  nSamples: 6,
  nFeatures: 2,
};
const initCenters = Float64Array.of(0, 0, 8, 8);  // (K=2, D=2)
const common = { nClusters: 2, maxIterations: 100 };

const km = run('kmeans', input, { ...common, initCenters });
console.log('kmeans:', km.labels, km.inertia);
assert.deepEqual(Array.from(km.labels), [0, 0, 0, 1, 1, 1]);

const fcm = run('fcm', input, { ...common, initCenters, m: 2, tolerance: 1e-5 });
console.log('fcm:', fcm.membership, fcm.converged);
assert.equal(fcm.membership.length, 12);

const rcm = run('rcm', input, { ...common, initCenters, alpha: 1.1, beta: 0.5 });
console.log('rcm:', rcm.membership, rcm.stopReason);
assert.equal(rcm.p, 1);

const exrcm = run('exrcm', input, {
  ...common, initCenters, p: 2, alpha: 1.1, beta: 0.5,
});
console.log('exrcm:', exrcm.mask, exrcm.stopReason);
assert.equal(exrcm.p, 2);

const som = run('som-olp', input, {
  grid: { data: Float64Array.of(0, 0, 0, 1, 1, 0, 1, 1), nSamples: 4, nFeatures: 2 },
  gamma: 0.5, lambda: 1, maxIterations: 50, tolerance: 1e-4,
});
console.log('som-olp:', som.embedding);
assert.equal(som.embedding.length, 12);

const quality = run('neighborhood', input, {
  embedding: { data: som.embedding, nSamples: 6, nFeatures: 2 }, ks: [1, 2],
});
console.log('neighborhood:', quality.qualities);
assert.deepEqual(quality.qualities.map(q => q.k), [1, 2]);
for (const q of quality.qualities) {
  assert.ok(q.trustworthiness >= 0 && q.trustworthiness <= 1);
  assert.ok(q.continuity >= 0 && q.continuity <= 1);
}
