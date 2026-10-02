import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createSession, run } from '../consumer/node_modules/ubukit-js/src/index.js';

const fixture = JSON.parse(readFileSync(new URL('../fixtures/warm-preflight-reference.json', import.meta.url), 'utf8'));
assert.equal(fixture.source_commit, 'b847c272bbbad5d152bf254467098a689c828f68');
const typed = { Float64Array, Float32Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array, Uint8ClampedArray };
function decode(value) {
  if (Array.isArray(value)) return value.map(decode);
  if (value && typeof value === 'object') {
    if (value.$undefined) return undefined;
    if (value.$number) return value.$number === '-0' ? -0 : Number(value.$number);
    if (value.$typed) { assert.ok(Object.hasOwn(typed, value.$typed)); return new typed[value.$typed](value.values.map(decode)); }
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, decode(child)]));
  }
  return value;
}
function referenceResult(key) { assert.ok(Object.hasOwn(fixture.results, key)); return decode(fixture.results[key]); }
const X = { data: Float64Array.of(0, 0, .1, -.1, .2, .1, 4, 4, 4.3, 4.1, 3.8, 4.2), nSamples: 6, nFeatures: 2 };
const options = { nClusters: 2, initCenters: Float64Array.of(0, 0, 4, 4), m: 64, maxMemoryBytes: 4000, maxIterations: 1, tolerance: 0 };
function finish(session) {
  for (let count = 0; !session.status.done; ++count) {
    assert.ok(count < 100, 'bounded session completion');
    session.step(10, { timeBudgetMs: 1000, maxChunks: 100 });
  }
  return session.snapshot().result;
}
function update(session, method, patch, controls = {}) {
  return method === 'updateParameters'
    ? session.updateParameters(patch, controls)
    : session.configure(X, { ...options, ...patch }, controls);
}

for (const method of ['updateParameters', 'configure']) {
  for (const phase of ['done', 'partial']) test(`${method}: insufficient stable-to-ordinary warm budget preserves ${phase} revision`, () => {
    const initial = { ...options, maxIterations: phase === 'partial' ? 2 : 1 };
    const session = createSession('fcm', X, initial);
    try {
      if (phase === 'done') { finish(session); }
      else {
        session.step(1, { timeBudgetMs: 1000, maxChunks: 100 });
        assert.equal(session.status.done, false);
      }
      const before = session.snapshot();
      assert.equal(before.result.numericalMode, 'log-domain');
      assert.throws(() => update(session, method, { m: 2, maxMemoryBytes: 2000, maxIterations: 1 }), { name: 'RangeError' });
      assert.deepEqual(session.snapshot(), before, 'failed preflight must preserve revision, checkpoint, and lifecycle status');
      assert.deepEqual(finish(session), referenceResult(`continued-${phase}`), 'unfinished original iteration must remain usable');
    } finally { session.dispose(); }
  });

  test(`${method}: retained anchor state is checked before any replacement iteration`, () => {
    const session = createSession('fcm', X, options);
    try {
      finish(session);
      update(session, method, { m: 2 });
      const before = session.snapshot();
      assert.equal(before.result, null); assert.equal(before.revision, 1);
      assert.throws(() => update(session, method, { m: 2, maxMemoryBytes: 2000 }), { name: 'RangeError' });
      assert.deepEqual(session.snapshot(), before);
      assert.deepEqual(finish(session), referenceResult('warm-64-2'));
    } finally { session.dispose(); }
  });

  test(`${method}: exact warm stable allocation boundary is transactional`, () => {
    const session = createSession('fcm', X, options);
    try {
      finish(session); const before = session.snapshot();
      // 2712 stable kernel + 832 session-owned/checkpoint reserve.
      assert.throws(() => update(session, method, { m: 2, maxMemoryBytes: 3543 }), { name: 'RangeError' });
      assert.deepEqual(session.snapshot(), before);
      update(session, method, { m: 2, maxMemoryBytes: 3544 });
      assert.equal(finish(session).numericalMode, 'log-domain');
    } finally { session.dispose(); }
  });

  for (const [oldM, newM] of [[64, 2], [64, 128], [2, 64]]) test(`${method}: valid m=${oldM} to m=${newM} warm transition preserves baseline results`, () => {
    const initial = { ...options, m: oldM };
    const session = createSession('fcm', X, initial);
    try {
      finish(session);
      update(session, method, { m: newM });
      assert.equal(session.status.invalidation.kind, 'membership-and-centers');
      assert.deepEqual(finish(session), referenceResult(`warm-${oldM}-${newM}`));
    } finally { session.dispose(); }
  });

  test(`${method}: ordinary-to-stable insufficient budget remains transactional`, () => {
    const session = createSession('fcm', X, { ...options, m: 2 });
    try {
      finish(session); const before = session.snapshot();
      assert.throws(() => update(session, method, { m: 64, maxMemoryBytes: 2000 }), { name: 'RangeError' });
      assert.deepEqual(session.snapshot(), before);
    } finally { session.dispose(); }
  });

  test(`${method}: cold restart does not inherit hidden stable budget requirements`, () => {
    const session = createSession('fcm', X, options);
    try {
      finish(session);
      update(session, method, { m: 2, maxMemoryBytes: 2000 }, { warmStart: false });
      assert.equal(session.status.invalidation.kind, 'cold');
      const result = finish(session);
      assert.deepEqual(result, run('fcm', X, { ...options, m: 2, maxMemoryBytes: 2000 }));
      assert.equal(result.numericalMode, undefined);
    } finally { session.dispose(); }
  });

  test(`${method}: explicit changed initialization discards hidden stable requirements`, () => {
    const session = createSession('fcm', X, options);
    const initial = Float64Array.of(.2, 0, 4.2, 4);
    try {
      finish(session);
      update(session, method, { m: 2, maxMemoryBytes: 2000, initCenters: initial });
      assert.equal(session.status.invalidation.kind, 'cold');
      assert.deepEqual(finish(session), run('fcm', X, { ...options, m: 2, maxMemoryBytes: 2000, initCenters: initial }));
    } finally { session.dispose(); }
  });
}

const tinyX = { data: Float64Array.of(0, 1e-100, 1), nSamples: 3, nFeatures: 1 };
const tinyOptions = { nClusters: 2, initCenters: Float64Array.of(0, 1), m: 2, maxIterations: 1, tolerance: 0, maxMemoryBytes: 2000 };
function updateTiny(session, method, patch, controls = {}) {
  return method === 'updateParameters'
    ? session.updateParameters(patch, controls)
    : session.configure(tinyX, { ...tinyOptions, ...patch }, controls);
}
for (const method of ['updateParameters', 'configure']) {
  for (const phase of ['done', 'partial']) test(`${method}: ordinary tiny retained membership preserves ${phase} revision on failed preflight`, () => {
    const session = createSession('fcm', tinyX, tinyOptions);
    try {
      if (phase === 'done') { finish(session); }
      else {
        session.step(1, { timeBudgetMs: 1000, maxChunks: 100 });
        assert.equal(session.status.done, false);
      }
      const before = session.snapshot();
      assert.equal(before.result.numericalMode, undefined, 'source must have no stable log state');
      assert.ok(before.result.membership.some(value => value > 0 && value < 1e-150));
      assert.throws(() => updateTiny(session, method, { m: 4, maxMemoryBytes: 1000 }), { name: 'RangeError' });
      assert.deepEqual(session.snapshot(), before);
      assert.deepEqual(finish(session), referenceResult('tiny-continued'));
    } finally { session.dispose(); }
  });
  test(`${method}: ordinary membership-triggered stable allocation boundary is transactional`, () => {
    const session = createSession('fcm', tinyX, tinyOptions);
    try {
      finish(session); const before = session.snapshot();
      // 1360 stable kernel + 400 session-owned/checkpoint reserve.
      assert.throws(() => updateTiny(session, method, { m: 4, maxMemoryBytes: 1759 }), { name: 'RangeError' });
      assert.deepEqual(session.snapshot(), before);
      updateTiny(session, method, { m: 4, maxMemoryBytes: 1760 });
      assert.equal(finish(session).numericalMode, 'log-domain');
    } finally { session.dispose(); }
  });
  test(`${method}: valid ordinary membership-triggered stable warm transition matches baseline`, () => {
    const session = createSession('fcm', tinyX, tinyOptions);
    try {
      finish(session);
      updateTiny(session, method, { m: 4 });
      assert.equal(session.status.invalidation.kind, 'membership-and-centers');
      const result = finish(session);
      assert.equal(result.numericalMode, 'log-domain');
      assert.deepEqual(result, referenceResult('tiny-warm-2-4'));
    } finally { session.dispose(); }
  });
  test(`${method}: ordinary cold restart ignores discarded tiny retained membership`, () => {
    const session = createSession('fcm', tinyX, tinyOptions);
    try {
      finish(session);
      updateTiny(session, method, { m: 4, maxMemoryBytes: 1000 }, { warmStart: false });
      assert.equal(session.status.invalidation.kind, 'cold');
      assert.deepEqual(finish(session), run('fcm', tinyX, { ...tinyOptions, m: 4, maxMemoryBytes: 1000 }));
    } finally { session.dispose(); }
  });
}
