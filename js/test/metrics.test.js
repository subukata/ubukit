import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';
import { X } from './helpers.js';

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
