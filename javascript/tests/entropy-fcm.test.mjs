import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import { entropyFcm, entropyFcmSteps, algorithms, run, runAsync, steps, sessionAlgorithms, createWorkerClient } from '../src/index.js';

const input = rows => ({ data: Float64Array.from(rows.flat()), nSamples: rows.length, nFeatures: rows[0].length });
function close(actual, expected, relative = 3e-13, absolute = 3e-14) {
  expected = expected.flat?.(Infinity) ?? expected;
  assert.equal(actual.length, expected.length);
  for (let j = 0; j < actual.length; ++j)
    assert.ok(Math.abs(actual[j] - expected[j]) <= absolute + relative * Math.abs(expected[j]), `${j}: ${actual[j]} versus ${expected[j]}`);
}
function consume(iterator) { for (;;) { const next = iterator.next(); if (next.done) return next.value; } }
const fixture = JSON.parse(readFileSync(new URL('../fixtures/entropy-fcm-reference.json', import.meta.url)));
for (const item of fixture.cases) test(`80-digit independent Decimal reference: ${item.name}`, () => {
  const result = entropyFcm(input(item.X), { initMembership: Float64Array.from(item.init.flat()), tau: item.tau,
    maxIterations: item.iterations, tolerance: 0, returnHistory: true });
  close(result.centers, item.centers); close(result.membership, item.membership);
  close(result.objectiveHistory, item.objective_history); close([result.objective], [item.objective]);
});

test('linear U center weights and coincident centers remain soft', () => {
  const x = input([[0], [2]]);
  const result = entropyFcm(x, { initMembership: Float64Array.of(.8, .2, .2, .8), tau: 2, maxIterations: 1 });
  close(result.centers, [.4, 1.6], 0, 3e-16);
  const p = 1 / (1 + Math.exp(-1.2)); close(result.membership, [p, 1 - p, 1 - p, p], 0, 3e-16);
  const coincident = entropyFcm(x, { initMembership: Float64Array.of(1, 0, 0, 1), tau: 2, maxIterations: 1 });
  assert.ok(coincident.membership[1] > .1);
  close([coincident.objective], [-4 * Math.log1p(Math.exp(-2))], 0, 8e-16);
});

test('objective evaluates returned arrays, simplex rows, first argmax and descent', () => {
  const x = input([[-2, 1], [-1, 2], [1, 1], [3, -1]]);
  for (const tau of [.02, .7, 10]) {
    const r = entropyFcm(x, { nClusters: 3, tau, seed: 19, maxIterations: 15, tolerance: 0, returnHistory: true });
    let objective = 0;
    for (let i = 0; i < x.nSamples; ++i) {
      let sum = 0, best = 0;
      for (let c = 0; c < r.nClusters; ++c) {
        const member = r.membership[i * r.nClusters + c]; sum += member;
        assert.ok(Number.isFinite(member) && member >= 0);
        if (member > r.membership[i * r.nClusters + best]) best = c;
        if (member > 0) {
          let cost = 0;
          for (let f = 0; f < x.nFeatures; ++f) cost += (x.data[i * x.nFeatures + f] - r.centers[c * x.nFeatures + f]) ** 2;
          objective += member * cost + tau * member * Math.log(member);
        }
      }
      assert.ok(Math.abs(sum - 1) < 4e-16); assert.equal(r.labels[i], best);
    }
    close([r.objective], [objective]);
    for (let j = 1; j < r.objectiveHistory.length; ++j) assert.ok(r.objectiveHistory[j] <= r.objectiveHistory[j - 1] + 1e-12);
    assert.equal(r.objectiveHistory.at(-1), r.objective);
  }
});

test('empty clusters, repeated points, single cluster, and zero tolerance', () => {
  const r = entropyFcm(input([[3, 2], [3, 2], [3, 2]]), { initMembership: Float64Array.of(1, 0, 0, 1, 0, 0, 1, 0, 0), tau: .5, maxIterations: 2 });
  assert.deepEqual([...r.centers], [3, 2, 3, 2, 3, 2]);
  assert.deepEqual([...r.membership], new Array(9).fill(1 / 3));
  assert.deepEqual([...r.labels], [0, 0, 0]);
  close([r.objective], [-1.5 * Math.log(3)]);
  const one = entropyFcm(input([[4, -5]]), { nClusters: 1, maxIterations: 3, tolerance: 0 });
  assert.deepEqual([...one.centers], [4, -5]); assert.equal(one.objective, 0);
  assert.equal(one.iterations, 3); assert.equal(one.converged, false);
  assert.equal(entropyFcm(input([[4]]), { nClusters: 1 }).iterations, 1);
});

test('owned arrays, deterministic seed, initial scaling and center initialization', () => {
  const x = input([[0], [1], [5]]), initial = Float64Array.of(.1, .9, .8, .2, .3, .7);
  const before = x.data.slice(), copy = initial.slice(), options = { nClusters: 2, seed: 5, maxIterations: 3, returnHistory: true };
  assert.deepEqual(entropyFcm(x, options), entropyFcm(x, options));
  const a = entropyFcm(x, { ...options, initMembership: initial });
  const b = entropyFcm(x, { ...options, initMembership: Float64Array.from(initial, (v, j) => v * [1e200, 1e-200, 5][Math.floor(j / 2)]) });
  close(a.membership, b.membership);
  for (const array of [a.centers, a.membership, a.labels, a.objectiveHistory]) array.fill(0);
  assert.deepEqual(x.data, before); assert.deepEqual(initial, copy);
  const c = entropyFcm(x, { initCenters: Float64Array.of(0, 5), maxIterations: 1 });
  assert.equal(c.nClusters, 2); assert.ok(c.membership.every(Number.isFinite));
});

test('finite positive tau, shape, membership, and resource validation', () => {
  const x = input([[0]]), base = { nClusters: 1 };
  for (const tau of [0, -1, NaN, Infinity, -Infinity, true, '1', [1]])
    assert.throws(() => entropyFcm(x, { ...base, tau }), RangeError);
  for (const option of [{ nClusters: 0 }, { maxIterations: 0 }, { maxIterations: true }, { tolerance: -1 },
    { initMembership: Float64Array.of(0) }, { initMembership: Float64Array.of(-1) },
    { initMembership: Float64Array.of(Infinity) }, { initMembership: new DataView(new ArrayBuffer(8)) },
    { initCenters: Float64Array.of(NaN) }, { initCenters: BigInt64Array.of(1n) },
    { initMembership: Float64Array.of(1), initCenters: Float64Array.of(0) },
    { maxMemoryBytes: 1 }, { blockRows: 0 }]) assert.throws(() => entropyFcm(x, { ...base, ...option }), RangeError);
  assert.throws(() => entropyFcm(input([[Infinity]]), base), RangeError);
  assert.throws(() => entropyFcm({ data: [0], nSamples: 1, nFeatures: 1 }, base), TypeError);
});

test('input copies bypass caller-overridden slice and iterator hooks', () => {
  const x = input([[0], [2]]), initialCenters = Float64Array.of(0, 2);
  x.data.slice = () => { throw new Error('caller slice must not run'); };
  x.data[Symbol.iterator] = () => { throw new Error('caller iterator must not run'); };
  initialCenters[Symbol.iterator] = () => { throw new Error('caller initializer iterator must not run'); };
  const result = entropyFcm(x, { initCenters: initialCenters, maxIterations: 1 });
  assert.ok(result.centers.every(Number.isFinite)); assert.ok(result.membership.every(Number.isFinite));
  assert.notEqual(result.centers.buffer, initialCenters.buffer); assert.notEqual(result.centers.buffer, x.data.buffer);
});

test('subnormal temperature and squared distances keep their real scale', () => {
  const hard = entropyFcm(input([[0], [2]]), { initMembership: Float64Array.of(1, 0, 0, 1), tau: Number.MIN_VALUE, maxIterations: 1 });
  assert.deepEqual([...hard.membership], [1, 0, 0, 1]); assert.equal(hard.objective, 0);
  const soft = entropyFcm(input([[0], [1e-160]]), { initMembership: Float64Array.of(1, 0, 0, 1), tau: 1e-320, maxIterations: 1 });
  assert.ok(soft.membership[0] > .7 && soft.membership[0] < .75);
  assert.ok(soft.membership.every(Number.isFinite));
});

test('extreme coordinates, signed objective overflow and nonzero underflow', () => {
  const x = input([[-1e308], [1e308]]);
  const hard = entropyFcm(x, { initMembership: Float64Array.of(1, 0, 0, 1), tau: 1e308, maxIterations: 1 });
  assert.deepEqual([...hard.centers], [-1e308, 1e308]); assert.deepEqual([...hard.membership], [1, 0, 0, 1]); assert.equal(hard.objective, 0);
  const positive = entropyFcm(x, { nClusters: 1, maxIterations: 1 });
  assert.equal(positive.objective, Infinity); assert.equal(positive.objectiveRepresentation, 'overflow'); assert.equal(positive.objectiveSign, 1);
  const negative = entropyFcm(input([[0], [0]]), { nClusters: 8, tau: 1e308, maxIterations: 1 });
  assert.equal(negative.objective, -Infinity); assert.equal(negative.objectiveRepresentation, 'overflow'); assert.equal(negative.objectiveSign, -1);
  const tiny = entropyFcm(input([[0], [Number.MIN_VALUE]]), { nClusters: 1, maxIterations: 1 });
  assert.equal(tiny.objective, 0); assert.equal(tiny.objectiveRepresentation, 'underflow'); assert.equal(tiny.objectiveSign, 1);
  for (const r of [positive, negative, tiny]) assert.ok(Number.isFinite(r.objectiveLogAbs));
});

test('weighted means retain cancellation residuals and tiny input coordinates', () => {
  for (const [values, expected] of [[[8e307, 1, -8e307], 1 / 3], [[-1e308, 1e308, 1e-320], 1e-320 / 3]]) {
    const r = entropyFcm(input(values.map(value => [value])), { nClusters: 1, maxIterations: 1 });
    assert.equal(r.centers[0], expected);
  }
  const r = entropyFcm(input([[1], [0], [1e-200]]), { initMembership: Float64Array.of(0, 1, 1, 0, 1, 0), maxIterations: 1 });
  close(r.centers, [5e-201, 1], 3e-15, 0);
});

test('registry, generator, cooperative API and progress agree; sessions are excluded', async () => {
  const x = input([[0], [1], [5]]), options = { nClusters: 2, seed: 4, maxIterations: 3, tolerance: 0, blockRows: 1 };
  const expected = entropyFcm(x, options), events = [];
  assert.ok(algorithms.includes('entropy-fcm')); assert.ok(!sessionAlgorithms.includes('entropy-fcm'));
  assert.deepEqual(run('entropy-fcm', x, options), expected);
  assert.deepEqual(consume(entropyFcmSteps(x, options)), expected);
  assert.deepEqual(consume(steps('entropy-fcm', x, options)), expected);
  assert.deepEqual(await runAsync('entropy-fcm', x, { ...options, timeBudgetMs: 0, onProgress: event => events.push(event) }), expected);
  assert.deepEqual(events.map(event => event.iteration), [1, 2, 3]);
  assert.equal(events.at(-1).objective, expected.objective);
});

test('cancellation checks run before initialization and between yielded blocks', () => {
  const x = input([[0], [1], [2]]), controller = new AbortController(); controller.abort();
  assert.throws(() => entropyFcm(x, { nClusters: 2, signal: controller.signal }), { name: 'AbortError' });
  let checks = 0;
  assert.throws(() => entropyFcm(x, { nClusters: 2, blockRows: 1, shouldCancel: () => ++checks > 3 }), { name: 'AbortError' });
  const options = { nClusters: 2, blockRows: 1 }, iterator = entropyFcmSteps(x, options);
  assert.equal(iterator.next().done, false); options.shouldCancel = () => true;
  assert.throws(() => iterator.next(), { name: 'AbortError' });
});

test('one-shot Node Worker preserves entropy FCM output and caller buffers', async () => {
  const x = input([[0], [1], [5]]), options = { nClusters: 2, seed: 17, maxIterations: 3, tolerance: 0, returnHistory: true };
  const expected = entropyFcm(x, options), before = x.data.slice();
  const client = createWorkerClient({ workerFactory: () => new Worker(new URL('../src/worker.js', import.meta.url), { type: 'module' }) });
  try { assert.deepEqual(await client.run('entropy-fcm', x, options), expected); assert.deepEqual(x.data, before); }
  finally { client.dispose(); }
});
