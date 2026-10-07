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
  assert.throws(() => ub.som(X, [3, 3], { init: 'random' }), /init/);
  assert.throws(() => ub.somOlp(X, [3, 3], { lam: 0, gamma: 1 }), RangeError);
});

test('toRows takes only a matrix whose data hold its rows', () => {
  assert.throws(() => ub.toRows({ data: new Float64Array(0), rows: 2 ** 40, cols: 0 }), TypeError);
  assert.throws(() => ub.toRows({ data: new Float64Array(3), rows: 2, cols: 2 }), TypeError);
  assert.deepEqual(ub.toRows(ub.matrix([[1, 2], [3, 4]])), [[1, 2], [3, 4]]);
});

test('rows are checked before the matrix is allocated', () => {
  // An object claiming a huge length is refused as a row (TypeError) before
  // any copy is allocated; allocating first raised RangeError from the typed
  // array constructor, after asking for 2^40 elements.
  assert.throws(() => ub.matrix([{ length: 2 ** 40 }]), TypeError);
  assert.throws(() => ub.matrix([[1, 2], 'ab']), TypeError);
  assert.throws(() => ub.matrix([null]), TypeError);
  assert.deepEqual(ub.matrix([Float64Array.of(1, 2), [3, 4]]).data, Float64Array.of(1, 2, 3, 4));
});

test('row entries must be numbers', () => {
  // The copy into a Float64Array read null as 0 (Python rejects None), true
  // as 1 and '3' as 3; a BigInt made it throw an unrelated error.
  for (const v of [null, true, '3', 1n]) assert.throws(() => ub.matrix([[v], [2]]), /rows must be arrays of numbers/);
  assert.throws(() => ub.kmeans([[0], [null], [2]], 2), TypeError);
});
