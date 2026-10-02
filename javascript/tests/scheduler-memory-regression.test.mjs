import test from 'node:test';
import assert from 'node:assert/strict';
import { createMetricScheduler, run } from '../consumer/node_modules/ubukit-js/src/index.js';
const input = (n, d, Type = Float64Array) => ({ data: Type.from({ length: n * d }, (_, i) => (i * 7) % 31), nSamples: n, nFeatures: d });
const largePair = () => input(5, 12);
const scratchHeavyPair = () => input(32, 1);
const largeOptions = x => ({ embedding: x, k: 1, maxScratchBytes: 140, blockRows: 1 });
const scratchOptions = x => ({ embedding: x, k: 15, maxScratchBytes: 1076, blockRows: 1 });

test('evicts an oversized retained cache before admitting differently shaped owned buffers', async () => {
  const scheduler = createMetricScheduler({ maxMemoryBytes: 2200, debounceMs: 0, maxChunks: 1 });
  try {
    const a = largePair(), b = scratchHeavyPair();
    await scheduler.run(a, largeOptions(a));
    assert.equal(scheduler.status.cached, true);
    const next = scheduler.run(b, scratchOptions(b));
    const cachedWhilePending = scheduler.status.cached;
    const result = await next;
    // Old cache 960 + new input pair 512 + required scratch 1076 = 2548.
    assert.equal(cachedWhilePending, false, '2200-byte budget must evict the 960-byte old cache');
    assert.deepEqual(result.result, run('neighborhood', b, scratchOptions(b)));
    assert.equal((await scheduler.run(b, scratchOptions(b))).cacheHit, true);
  } finally { scheduler.dispose(); }
});

for (const cancelFirst of [false, true]) test(`callback replacement accounts for an executing generator${cancelFirst ? ' after explicit cancellation' : ''}`, async () => {
  const scheduler = createMetricScheduler({ maxMemoryBytes: 2300, debounceMs: 0, maxChunks: 1 });
  let replacement, triggered = false;
  try {
    const a = input(16, 1), b = scratchHeavyPair();
    const old = await scheduler.run(a, { embedding: a, k: 7, maxScratchBytes: 532, blockRows: 1,
      onProgress: () => {
        if (triggered) return; triggered = true;
        if (cancelFirst) scheduler.cancel();
        replacement = scheduler.run(b, scratchOptions(b)).catch(error => error);
      }
    }).catch(error => error);
    const next = await replacement;
    // Executing pair/scratch 256+532 plus replacement 512+1076 = 2376.
    assert.equal(next.name, 'RangeError', 'executing buffers remain live until the progress callback returns');
    if (cancelFirst) assert.equal(old.name, 'AbortError');
    else assert.ok(old.result, 'rejected replacement must not cancel the valid executing request');
    assert.equal(scheduler.status.busy, false);
    assert.ok((await scheduler.run(b, scratchOptions(b))).result, 'reuse succeeds after executing buffers are released');
  } finally { scheduler.dispose(); }
});

test('non-running debounced replacement releases the superseded request', async () => {
  const scheduler = createMetricScheduler({ maxMemoryBytes: 2200, debounceMs: 1, maxChunks: 1 });
  try {
    const x = scratchHeavyPair();
    const old = scheduler.run(x, scratchOptions(x)).catch(error => error);
    const next = scheduler.run(x, scratchOptions(x));
    assert.equal((await old).name, 'AbortError');
    assert.ok((await next).result);
    assert.equal(scheduler.status.busy, false);
  } finally { scheduler.dispose(); }
});

test('retains a small affordable cache and keeps returned results independent', async () => {
  const scheduler = createMetricScheduler({ maxMemoryBytes: 2200, debounceMs: 0 });
  try {
    const a = scratchHeavyPair(), b = largePair();
    await scheduler.run(a, scratchOptions(a));
    const next = scheduler.run(b, largeOptions(b));
    assert.equal(scheduler.status.cached, true, '512+960+140 fits the 2200-byte budget');
    const result = await next;
    result.result.qualities[0].trustworthiness = -999;
    const cached = await scheduler.run(b, largeOptions(b));
    assert.equal(cached.cacheHit, true);
    assert.notEqual(cached.result.qualities[0].trustworthiness, -999);
  } finally { scheduler.dispose(); }
});

for (const key of ['embedding', 'coordinates']) test(`${key}: Float32 inputs are owned and cache remains usable`, async () => {
  const scheduler = createMetricScheduler({ debounceMs: 1 });
  try {
    const x = input(8, 2, Float32Array), y = input(8, 1, Float32Array);
    const original = { ...x, data: x.data.slice() }, embedding = { ...y, data: y.data.slice() };
    const options = { [key]: y, k: 2 };
    const expected = run('neighborhood', original, { [key]: embedding, k: 2 });
    const pending = scheduler.run(x, options);
    x.data.fill(123); y.data.fill(-123);
    assert.deepEqual((await pending).result, expected);
    assert.equal((await scheduler.run(original, { [key]: embedding, k: 2 })).cacheHit, true);
  } finally { scheduler.dispose(); }
});

test('rejects impossible owned/scratch reservations without abandoning current work', async () => {
  const scheduler = createMetricScheduler({ maxMemoryBytes: 2200, debounceMs: 0 });
  try {
    const x = largePair();
    const old = scheduler.run(x, largeOptions(x));
    await assert.rejects(scheduler.run(x, { ...largeOptions(x), maxScratchBytes: 2200 }), /reserve/);
    assert.ok((await old).result);
    await assert.rejects(scheduler.run(x, { ...largeOptions(x), maxScratchBytes: -1 }), /maxScratchBytes/);
  } finally { scheduler.dispose(); }
});

test('rejects before allocating Float32-to-Float64 input conversions', async () => {
  const scheduler = createMetricScheduler({ maxMemoryBytes: 1 });
  const x = input(8, 2, Float32Array), OriginalArray = Float64Array;
  let conversions = 0, pending;
  try {
    globalThis.Float64Array = new Proxy(OriginalArray, {
      construct(target, args, newTarget) { ++conversions; return Reflect.construct(target, args, newTarget); }
    });
    pending = scheduler.run(x, { embedding: x, k: 1 });
  } finally { globalThis.Float64Array = OriginalArray; }
  try {
    await assert.rejects(pending, /reserve/);
    assert.equal(conversions, 0, 'reservation must precede either input conversion');
  } finally { scheduler.dispose(); }
});

test('caller-overridden TypedArray.slice cannot defeat numerical ownership', async () => {
  class AliasingArray extends Float64Array { slice() { return this; } }
  const scheduler = createMetricScheduler({ debounceMs: 1 });
  try {
    const x = input(8, 2, AliasingArray), y = input(8, 1, AliasingArray);
    const options = { embedding: y, k: 2 };
    const expected = run('neighborhood', x, options);
    const pending = scheduler.run(x, options);
    x.data.fill(123); y.data.fill(-123);
    assert.deepEqual((await pending).result, expected);
  } finally { scheduler.dispose(); }
});

test('caller-overridden iteration cannot change the inspected values or allocation size', async () => {
  class CustomIteration extends Float64Array { *[Symbol.iterator]() { yield 999; } }
  const scheduler = createMetricScheduler({ debounceMs: 1 });
  try {
    const x = input(8, 2, CustomIteration), y = input(8, 1, CustomIteration);
    const options = { embedding: y, k: 2 };
    const expected = run('neighborhood', x, options);
    assert.deepEqual((await scheduler.run(x, options)).result, expected);
  } finally { scheduler.dispose(); }
});
