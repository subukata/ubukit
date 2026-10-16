import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X, largeAllocations } from './helpers.js';

test('metrics edge cases', () => {
  assert.equal(ub.ari([0, 0, 1, 1], [5, 5, 3, 3]), 1);
  assert.equal(ub.ami(['a', 'b'], ['x', 'y']), 1);
  assert.equal(ub.ari([], []), 1);
  assert.throws(() => ub.ami([0, 1], [0, 1], { average: 'median' }), RangeError);
  // Names every object inherits are not averages either.
  for (const average of ['toString', 'constructor', 'hasOwnProperty', '__proto__']) {
    assert.throws(() => ub.ami([0, 1], [0, 1], { average }), RangeError, average);
  }
  assert.throws(() => ub.ami(5, 7), TypeError);
  assert.throws(() => ub.ari('ab', 'ab'), TypeError);
  assert.throws(() => ub.trustworthiness(X, X, 200), RangeError);
});

// Ties in both spaces: duplicated inputs and an integer grid embedding.
const P = X.map((x, i) => (i % 10 === 0 ? X[0] : x)), G = X.map(([a, b]) => [Math.round(a), Math.round(b)]);

test('several k give the values of separate calls, in their order', () => {
  for (const metric of [ub.trustworthiness, ub.continuity]) {
    const ks = [7, 2, 4], values = metric(P, G, ks);
    assert.ok(values instanceof Float64Array);
    assert.deepEqual(values, Float64Array.from(ks.map(k => metric(P, G, k))), metric.name);
  }
});

test('a Neighborhood gives the values of its points', () => {
  const nP = new ub.Neighborhood(P), nG = new ub.Neighborhood(G);
  for (const metric of [ub.trustworthiness, ub.continuity]) {
    const expected = metric(P, G, [5, 1, 9]);
    for (const [A, B] of [[nP, G], [P, nG], [nP, nG]]) assert.deepEqual(metric(A, B, [5, 1, 9]), expected, metric.name);
    assert.equal(metric(nP, G), metric(P, G));
  }
  // It keeps the distances it was built with, and cannot be changed.
  const rows = P.map(r => r.slice()), reference = new ub.Neighborhood(rows), before = ub.trustworthiness(reference, G, 5);
  rows.forEach(r => r.fill(0));
  assert.equal(ub.trustworthiness(reference, G, 5), before);
  assert.ok(Object.isFrozen(reference));
});

test('neighborhoods and lists of k are checked', () => {
  const reference = new ub.Neighborhood(X);
  assert.throws(() => ub.trustworthiness(reference, X.slice(1)), /same number of rows/);
  assert.throws(() => ub.continuity(X.slice(1), reference), /same number of rows/);
  assert.throws(() => ub.trustworthiness(reference, X, []), RangeError);
  assert.throws(() => ub.trustworthiness(reference, X, [5, 200]), /n \/ 2/);
  assert.throws(() => ub.trustworthiness(reference, X, [5, 2.5]), RangeError);
  assert.throws(() => ub.fcm(reference, 2), TypeError); // not data
});

test('labels are checked before anything is allocated for them', () => {
  // An object claiming a length is not a labeling (TypeError) and sizes no
  // allocation; the old check accepted it, asked for 2^40 codes (RangeError)
  // and scored two such objects as identical.
  for (const metric of [ub.ari, ub.ami]) {
    assert.throws(() => metric({ length: 2 ** 40 }, { length: 2 ** 40 }), TypeError);
    assert.throws(() => metric({ length: 3 }, { length: 3 }), TypeError);
  }
  assert.equal(ub.ari(Int32Array.of(0, 0, 1), [5, 5, 3]), 1);
});

test('labels are values, as in Python', () => {
  // Arrays as labels were compared by identity, each a cluster of its own,
  // so two labelings of nothing but distinct arrays scored 1; Python refuses
  // labels that are not 1-D.
  for (const metric of [ub.ari, ub.ami]) {
    assert.throws(() => metric([[0], [0], [1]], [[0], [0], [1]]), { name: 'TypeError', message: 'labels must hold numbers, strings or booleans' });
    assert.throws(() => metric([0, null], [0, 1]), TypeError);
    assert.throws(() => metric([0, 1], [{}, {}]), TypeError);
  }
  assert.equal(ub.ari([true, true, false], ['a', 'a', 'b']), 1);
});

test('sparse labels and lists of k are refused before anything is allocated for them', () => {
  // Their holes were skipped or checked late, so the encoding of the labels
  // and the copy of k were sized by the claimed length.
  for (const call of [() => ub.ari(new Array(5e6), new Array(5e6)), () => ub.ami(new Array(5e6), new Array(5e6))]) {
    assert.throws(call, { name: 'TypeError', message: 'labels must hold numbers, strings or booleans' });
    assert.deepEqual(largeAllocations(call), []);
  }
  assert.throws(() => ub.trustworthiness(P, G, new Array(2 ** 32 - 1)), /k must be an integer/);
});

test('any number of k', () => {
  // Math.max(...ks) overflowed the stack for 200,000 of them.
  const values = ub.trustworthiness(P, G, Array(200_000).fill(3)), one = ub.trustworthiness(P, G, 3);
  assert.equal(values.length, 200_000);
  assert.ok(values.every(v => v === one)); // one reference: computing it per entry took minutes
});
