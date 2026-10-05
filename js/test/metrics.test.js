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
