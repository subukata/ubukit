import test from 'node:test';
import assert from 'node:assert/strict';
import { createSession } from '../consumer/node_modules/ubukit-js/src/index.js';
import { cloneOwned } from '../consumer/node_modules/ubukit-js/src/session-validation.js';

const input = { data: Float64Array.of(0, 1, 9, 10), nSamples: 4, nFeatures: 1 };
const options = { nClusters: 2, maxIterations: 3, seed: 1 };
function finish(session) {
  for (let chunks = 0; !session.status.done; ++chunks) {
    assert.ok(chunks < 100, 'session must finish within bounded chunks');
    session.step(10, { timeBudgetMs: 1000, maxChunks: 100 });
  }
  return session.snapshot().result;
}
function unchangedGlobals(callback) {
  const object = Object.getOwnPropertyDescriptors(Object.prototype);
  const array = Object.getOwnPropertyDescriptors(Array.prototype);
  try { callback(); }
  finally {
    assert.deepEqual(Object.getOwnPropertyDescriptors(Object.prototype), object);
    assert.deepEqual(Object.getOwnPropertyDescriptors(Array.prototype), array);
  }
}
function ownData(object, key, value) {
  assert.deepEqual(Object.getOwnPropertyDescriptor(object, key), {
    value, writable: true, enumerable: true, configurable: true,
  });
}

test('cloneOwned preserves JSON __proto__ as owned data with an ordinary prototype', () => unchangedGlobals(() => {
  const source = JSON.parse('{"__proto__":{"signal":{"aborted":true},"injected":1},"value":2}');
  const copy = cloneOwned(source);
  assert.equal(Object.getPrototypeOf(copy), Object.prototype);
  assert.equal(copy.signal, undefined);
  assert.equal(copy.injected, undefined);
  ownData(copy, '__proto__', copy.__proto__);
  assert.deepEqual(copy, source);
  assert.notEqual(copy.__proto__, source.__proto__);
  assert.notEqual(copy.__proto__.signal, source.__proto__.signal);
  copy.__proto__.signal.aborted = false;
  assert.equal(source.__proto__.signal.aborted, true);
}));

for (const value of [null, 3, 'data']) {
  test(`cloneOwned preserves a ${JSON.stringify(value)} __proto__ value`, () => unchangedGlobals(() => {
    const source = JSON.parse(`{"__proto__":${JSON.stringify(value)}}`);
    const copy = cloneOwned(source);
    assert.equal(Object.getPrototypeOf(copy), Object.prototype);
    ownData(copy, '__proto__', value);
    assert.deepEqual(copy, source);
  }));
}

test('cloneOwned keeps nested constructor/prototype keys, array prototypes, aliases, and cycles', () => unchangedGlobals(() => {
  const source = JSON.parse('{"constructor":{"prototype":{"__proto__":{"injected":1}}},"prototype":{"nested":{"__proto__":{"injected":2}}},"items":[{"__proto__":{"injected":3}}]}');
  source.self = source;
  source.alias = source.constructor.prototype;
  source.constructor.prototype.owner = source;
  source.items.push(source.items, source);
  Object.defineProperty(source.items, '__proto__', {
    value: { owner: source.items, injected: 4 }, enumerable: true,
    writable: true, configurable: true,
  });
  const copy = cloneOwned(source);
  assert.deepEqual(copy, source);
  assert.equal(Object.getPrototypeOf(copy), Object.prototype);
  assert.equal(Object.getPrototypeOf(copy.constructor), Object.prototype);
  assert.equal(Object.getPrototypeOf(copy.constructor.prototype), Object.prototype);
  assert.equal(Object.getPrototypeOf(copy.prototype.nested), Object.prototype);
  assert.equal(Object.getPrototypeOf(copy.items), Array.prototype);
  assert.equal(Object.getPrototypeOf(copy.items[0]), Object.prototype);
  assert.ok(Array.isArray(copy.items));
  assert.equal(copy.self, copy);
  assert.equal(copy.alias, copy.constructor.prototype);
  assert.equal(copy.alias.owner, copy);
  assert.equal(copy.items[1], copy.items);
  assert.equal(copy.items[2], copy);
  assert.equal(copy.items.__proto__.owner, copy.items);
  for (const object of [copy.alias, copy.prototype.nested, copy.items, copy.items[0]]) {
    assert.equal(object.injected, undefined);
    ownData(object, '__proto__', object.__proto__);
  }
  ownData(copy, 'constructor', copy.constructor);
  ownData(copy, 'prototype', copy.prototype);
  copy.alias.__proto__.injected = 9;
  assert.equal(source.alias.__proto__.injected, 1);
}));

test('cloneOwned retains own-enumerable-only, numerical ownership, and live signal contracts', () => unchangedGlobals(() => {
  const controller = new AbortController();
  const source = Object.create({ inherited: 'not copied' });
  const callback = () => true;
  const values = Float64Array.of(1, 2, 3);
  Object.assign(source, { values, alias: values, bytes: Buffer.from([4, 5]),
    buffer: Uint8Array.of(6, 7).buffer, signal: controller.signal, callback });
  Object.defineProperty(source, 'hidden', { value: 1 });
  const copy = cloneOwned(source);
  assert.equal(Object.getPrototypeOf(copy), Object.prototype);
  assert.equal(copy.inherited, undefined);
  assert.equal(Object.hasOwn(copy, 'hidden'), false);
  assert.ok(copy.values instanceof Float64Array);
  assert.deepEqual(copy.values, values);
  assert.equal(copy.alias, copy.values);
  assert.notEqual(copy.values.buffer, values.buffer);
  copy.values[0] = 10;
  copy.bytes[0] = 11;
  new Uint8Array(copy.buffer)[0] = 12;
  assert.equal(values[0], 1);
  assert.equal(source.bytes[0], 4);
  assert.equal(new Uint8Array(source.buffer)[0], 6);
  assert.equal(copy.signal, controller.signal);
  assert.equal(copy.callback, callback);
  controller.abort();
  assert.equal(copy.signal.aborted, true);
}));

test('createSession ignores JSON prototype-injected cancellation and matches valid options', () => unchangedGlobals(() => {
  const poisoned = JSON.parse('{"nClusters":2,"maxIterations":3,"seed":1,"__proto__":{"signal":{"aborted":true}}}');
  assert.equal(poisoned.signal, undefined);
  const session = createSession('kmeans', input, poisoned);
  const reference = createSession('kmeans', input, options);
  try {
    assert.deepEqual(finish(session), finish(reference));
    assert.equal(session.status.phase, 'done');
    assert.equal(session.status.error, null);
    assert.equal(poisoned.__proto__.signal.aborted, true);
    assert.equal(Object.getPrototypeOf(poisoned), Object.prototype);
  } finally { session.dispose(); reference.dispose(); }
}));

for (const method of ['updateParameters', 'configure']) {
  test(`${method} retains safe cloned options across replacement and reset`, () => unchangedGlobals(() => {
    const session = createSession('kmeans', input, options);
    const reference = createSession('kmeans', input, options);
    try {
      const expected = finish(reference);
      finish(session);
      const patch = JSON.parse('{"nClusters":2,"maxIterations":3,"seed":1,"__proto__":{"signal":{"aborted":true}}}');
      if (method === 'configure') session.configure(input, patch, { warmStart: false });
      else session.updateParameters(patch, { warmStart: false });
      assert.deepEqual(finish(session), expected);
      session.updateData(input, { warmStart: false });
      assert.deepEqual(finish(session), expected);
      session.reset();
      assert.deepEqual(finish(session), expected);
    } finally { session.dispose(); reference.dispose(); }
  }));
}

test('createSession still honors an actual own live AbortSignal', () => unchangedGlobals(() => {
  const controller = new AbortController();
  const session = createSession('kmeans', input, { ...options, signal: controller.signal });
  try {
    controller.abort();
    assert.throws(() => session.step(), { name: 'AbortError', message: 'Computation cancelled' });
    assert.equal(session.status.phase, 'cancelled');
    assert.equal(session.snapshot().result, null);
  } finally { session.dispose(); }
}));
