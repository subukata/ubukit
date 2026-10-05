import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X } from './helpers.js';

test('matrix input accepts {data, rows, cols} and validates', () => {
  const m = ub.matrix(X);
  assert.deepEqual(ub.kmeans(m, 3, { seed: 2 }).centers, ub.kmeans(X, 3, { seed: 2 }).centers);
  assert.throws(() => ub.kmeans([[1, NaN]], 1), RangeError);
  assert.throws(() => ub.kmeans([[1, 2], [3]], 1), RangeError);
  assert.throws(() => ub.matrix({ data: new Float64Array(5), rows: 2.5, cols: 2 }), RangeError);
  assert.throws(() => ub.kmeans(X.map(row => row.map(v => v * 1e200)), 3), /scale/);
  assert.throws(() => ub.kmeans(X.map(row => row.map(v => v * 1e-170)), 3), /scale/);
  assert.deepEqual(Array.from(ub.fcm([[1, 1], [1, 1], [1, 1]], 2, { seed: 0 }).centers.data), [1, 1, 1, 1]);
  assert.throws(() => ub.rmcm(X, 3, 0, { maxEdges: 10 }), /maxEdges/);
  assert.throws(() => ub.somOlp(X, [3, 3], { lam: 1, gamma: 1, pcaScale: 0 }), RangeError);
  assert.throws(() => ub.kmeans(X, 0), RangeError);
  assert.throws(() => ub.fcm(X, 3, { m: 1 }), RangeError);
  assert.throws(() => ub.rmcm(X, 3, 100, { maxEdges: 1000 }), /maxEdges/);
  assert.throws(() => ub.kmeans(X, 3, { init: 'random' }), RangeError);
  assert.throws(() => ub.somOlp(X, [3, 3], { lam: 0, gamma: 1 }), RangeError);
});
