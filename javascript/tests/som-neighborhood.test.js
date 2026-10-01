import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { somOlp, somOlpSteps } from '../src/som-olp.js';
import { neighborhood, neighborhoodSteps } from '../src/neighborhood.js';
const fixture = JSON.parse(readFileSync(new URL('../fixtures/som-neighborhood.json', import.meta.url), 'utf8'));
const input = z => ({ ...z, data: Float64Array.from(z.data) });
function close(actual, expected, abs = 2e-12, rel = 2e-12) {
  if (expected === null) { assert.equal(actual, null); return; }
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i] - expected[i]) <= abs + rel * Math.abs(expected[i]), `index ${i}: ${actual[i]} != ${expected[i]}`);
}
function somArgs(f) { return { ...f.options, grid: input(f.grid), initialPrototypes: Float64Array.from(f.initialPrototypes), initialMemberships: Float64Array.from(f.initialMemberships) }; }
for (const f of fixture.som) test(`SOM prepared-init Python parity: ${f.name}`, () => {
  const x = input(f.input), options = somArgs(f), beforeX = x.data.slice(), beforeP = options.initialMemberships.slice(), beforeW = options.initialPrototypes.slice();
  const output = somOlp(x, { ...options, blockRows: 3 });
  const atol = f.name === 'large-offset' ? 3e-6 : 2e-12;
  for (const key of ['W', 'P', 'V', 'history']) close(output[key], f.expected[key], atol, 2e-12);
  assert.equal(output.iterations, f.iterations);
  assert.deepEqual(x.data, beforeX); assert.deepEqual(options.initialMemberships, beforeP); assert.deepEqual(options.initialPrototypes, beforeW);
  assert.strictEqual(output.W, output.centers); assert.strictEqual(output.P, output.membership);
  assert.equal(output.initialization.kind, 'provided-prototypes-and-memberships');
  for (let i = 0; i < f.input.nSamples; i++) {
    let sum = 0; for (let j = 0; j < f.grid.nSamples; j++) sum += output.P[i * f.grid.nSamples + j];
    assert.ok(Math.abs(sum - 1) < 2e-15);
  }
});
for (const f of fixture.pca) test(`PCA compared to sign-canonicalized Python SVD: ${f.name}`, () => {
  const result = somOlp(input(f.input), { grid: input(f.grid), lambda: f.lambda, maxIterations: 0 });
  close(result.W, f.W, 2e-11, 2e-11); close(result.P, f.P, 2e-11, 2e-11);
  assert.equal(result.initialization.numpySvdParity, false);
  assert.equal(result.initialization.kind, 'pca-jacobi');
});
test('SOM retains upstream final V timing rather than silently recomputing P@R', () => {
  const f = fixture.som[0], options = somArgs(f);
  const result = somOlp(input(f.input), { ...options, maxIterations: 1 });
  close(result.V, f.firstV);
  const recomputed = new Float64Array(result.V.length);
  for (let i = 0; i < result.nSamples; i++) for (let j = 0; j < result.nUnits; j++) for (let h = 0; h < result.nComponents; h++) recomputed[i * result.nComponents + h] += result.P[i * result.nUnits + j] * options.grid.data[j * result.nComponents + h];
  assert.ok(recomputed.some((v, i) => Math.abs(v - result.V[i]) > 1e-3));
});
test('SOM sample initialization is explicit and reproducible', () => {
  const f = fixture.som[0], x = input(f.input), options = { grid: input(f.grid), initializer: 'sample', seed: 314, maxIterations: 2 };
  const a = somOlp(x, options), b = somOlp(x, options);
  assert.deepEqual(a.P, b.P); assert.deepEqual(a.W, b.W);
  assert.equal(a.initialization.kind, 'sample-with-replacement');
});
test('SOM tile boundaries do not change reductions or outputs', () => {
  const f = fixture.som[0], x = input(f.input), options = somArgs(f);
  const a = somOlp(x, { ...options, blockRows: 1 }), b = somOlp(x, { ...options, blockRows: 99 });
  for (const key of ['P', 'W', 'V', 'history']) assert.deepEqual(a[key], b[key]);
});
test('SOM generator progress and cooperative cancellation', () => {
  const f = fixture.som[0]; let events = 0, cancel = false;
  const generator = somOlpSteps(input(f.input), { ...somArgs(f), blockRows: 1, onProgress: () => events++, shouldCancel: () => cancel });
  assert.equal(generator.next().done, false); assert.equal(events, 1);
  cancel = true; assert.throws(() => generator.next(), { name: 'AbortError' });
});
test('SOM validates shapes, memberships, hyperparameters, and scratch', () => {
  const f = fixture.som[0], x = input(f.input), good = somArgs(f);
  for (const patch of [{lambda: 0}, {gamma: -1}, {maxIterations: -1}, {tolerance: NaN}, {blockRows: 0}, {maxScratchBytes: 1}, {maxMemoryBytes: 1}, {initialPrototypes: new Float64Array(1)}, {initialMemberships: new Float64Array(good.initialMemberships.length)}]) assert.throws(() => somOlp(x, {...good, ...patch}));
  assert.throws(() => somOlp(x, {grid: good.grid, pcaMaxDimension: 1}), /PCA eigensystem/);
  assert.throws(() => somOlp(x, {grid: good.grid, initialMemberships: good.initialMemberships}), /requires initialPrototypes/);
});
for (const f of fixture.neighborhood) test(`Exact metric Python index-tie oracle: ${f.name}`, () => {
  const x = input(f.input), y = input(f.embedding);
  const result = neighborhood(x, {embedding: y, ks: f.ks, blockRows: 3});
  assert.deepEqual(result.qualities, f.expected);
  assert.equal(result.stats.distanceMatrixAllocated, false);
  assert.equal(result.stats.rankMatrixAllocated, false);
  assert.equal(result.stats.tiePolicy, 'distance-then-sample-index');
  if (f.name === 'ordinary') assert.deepEqual(result.qualities, f.sklearn);
});
test('Neighborhood documents rather than hides sklearn tied-case differences', () => {
  const tied = fixture.neighborhood.find(f => f.name === 'duplicate-ties');
  assert.notDeepEqual(tied.expected, tied.sklearn);
});
test('Neighborhood identity, reverse symmetry, multi-k order, and no mutation', () => {
  const f = fixture.neighborhood[0], x = input(f.input), y = input(f.embedding), before = x.data.slice();
  const own = neighborhood(x, {embedding: x, ks: [5, 1, 5]});
  assert.deepEqual(own.qualities.map(q => [q.k, q.trustworthiness, q.continuity]), [[5, 1, 1], [1, 1, 1]]);
  const a = neighborhood(x, {embedding: y, ks: [1, 3]}), b = neighborhood(y, {embedding: x, ks: [1, 3]});
  for (let z = 0; z < a.qualities.length; z++) { assert.equal(a.qualities[z].trustworthiness, b.qualities[z].continuity); assert.equal(a.qualities[z].continuity, b.qualities[z].trustworthiness); }
  assert.deepEqual(x.data, before);
});
test('Neighborhood progress, cancellation, scratch cap, and invalid k', () => {
  const f = fixture.neighborhood[0], x = input(f.input), options = { embedding: input(f.embedding), k: 2 };
  for (const k of [0, -1, 1.5, x.nSamples / 2, Infinity]) assert.throws(() => neighborhood(x, {...options, k}), /Each k/);
  assert.throws(() => neighborhood(x, {...options, maxScratchBytes: 1}), /scratch/);
  let events = 0, cancel = false;
  const steps = neighborhoodSteps(x, {...options, blockRows: 2, onProgress: () => events++, shouldCancel: () => cancel});
  steps.next(); assert.equal(events, 1); cancel = true; assert.throws(() => steps.next(), {name: 'AbortError'});
  assert.throws(() => neighborhood({ ...x, data: Float64Array.from(x.data, () => NaN) }, options), /finite/);
});
test('Neighborhood heap/histogram agrees with brute full ranks on tiny randomized tied cases', () => {
  let state = 721;
  const next = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  for (let trial = 0; trial < 35; trial++) {
    const n = 5 + trial % 14;
    const x = {data: Float64Array.from({length: 3 * n}, () => Math.floor(next() * 4)), nSamples: n, nFeatures: 3};
    const y = {data: Float64Array.from({length: 2 * n}, () => Math.floor(next() * 4)), nSamples: n, nFeatures: 2};
    function full(z) {
      const orders = [], ranks = [];
      for (let i = 0; i < n; i++) {
        const ds = Array.from({length: n}, (_, j) => {
          let sum = 0; for (let f = 0; f < z.nFeatures; f++) { const delta = z.data[i * z.nFeatures + f] - z.data[j * z.nFeatures + f]; sum += delta * delta; }
          return Math.sqrt(sum);
        });
        const order = Array.from({length: n}, (_, j) => j).filter(j => j !== i).sort((a, b) => ds[a] - ds[b] || a - b);
        const rank = new Uint32Array(n); order.forEach((j, a) => rank[j] = a + 1); orders.push(order); ranks.push(rank);
      }
      return {orders, ranks};
    }
    const a = full(x), b = full(y), ks = Array.from({length: Math.floor((n - 1) / 2)}, (_, j) => j + 1);
    const actual = neighborhood(x, {embedding: y, ks}).qualities;
    for (let z = 0; z < ks.length; z++) {
      const k = ks[z]; let pt = 0, pc = 0;
      for (let i = 0; i < n; i++) for (let h = 0; h < k; h++) {pt += Math.max(0, a.ranks[i][b.orders[i][h]] - k); pc += Math.max(0, b.ranks[i][a.orders[i][h]] - k);}
      assert.equal(actual[z].trustworthinessPenalty, pt); assert.equal(actual[z].continuityPenalty, pc);
    }
  }
});
