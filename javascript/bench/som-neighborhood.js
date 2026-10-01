/** Matched, intentionally straightforward JavaScript baselines plus timing harness.
 * No public speed claim is valid until this harness is executed on named hardware.
 * This file introduces no package-wide license grant. Upstream SOM attribution is
 * retained in LICENSES/SOM-OLP-MIT.txt and docs/SOM_AND_NEIGHBORHOOD.md.
 */
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { normalizeInput, checkCancelled, labelsFromMembership, seededRandom } from '../src/core.js';
import { somOlp } from '../src/som-olp.js';
import { neighborhood } from '../src/neighborhood.js';

function squared(a, ao, b, bo, d) {
  let sum = 0;
  for (let f = 0; f < d; f++) { const delta = a[ao + f] - b[bo + f]; sum += delta * delta; }
  if (!Number.isFinite(sum)) throw new RangeError('distance overflow');
  return sum;
}
function probabilities(cost, n, m, lambda) {
  const p = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    const offset = i * m; let min = Infinity, sum = 0;
    for (let j = 0; j < m; j++) min = Math.min(min, cost[offset + j]);
    for (let j = 0; j < m; j++) { p[offset + j] = Math.exp(-(cost[offset + j] - min) / lambda); sum += p[offset + j]; }
    for (let j = 0; j < m; j++) p[offset + j] /= sum;
  }
  return p;
}

/** Full intermediate matrices, separate V/W/cost/softmax/objective passes.
 * Same O(iterations*N*M*(D+Q)) arithmetic as the production kernel. The baseline
 * is dependency-free scalar JS, not Python, LAPACK, GPU, or a fabricated sleep.
 */
export function naiveSomOlp(input, options) {
  const x = normalizeInput(input), grid = normalizeInput(options.grid);
  const { data, nSamples: n, nFeatures: d } = x;
  const { data: r, nSamples: m, nFeatures: q } = grid;
  if (!options.initialPrototypes || !options.initialMemberships) throw new Error('Benchmark baseline requires prepared initialization');
  const w = Float64Array.from(options.initialPrototypes);
  let p = Float64Array.from(options.initialMemberships), v = null;
  const history = [], gamma = options.gamma, lambda = options.lambda;
  for (let iteration = 0; iteration < options.maxIterations; iteration++) {
    checkCancelled(options);
    v = new Float64Array(n * q);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const value = p[i * m + j];
      for (let h = 0; h < q; h++) v[i * q + h] += value * r[j * q + h];
    }
    const numerator = new Float64Array(m * d), denominator = new Float64Array(m);
    for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) {
      const value = p[i * m + j]; denominator[j] += value;
      for (let f = 0; f < d; f++) numerator[j * d + f] += value * data[i * d + f];
    }
    for (let j = 0; j < m; j++) if (denominator[j] > 0) for (let f = 0; f < d; f++) w[j * d + f] = numerator[j * d + f] / denominator[j];
    const cost = new Float64Array(n * m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) cost[i * m + j] = squared(data, i * d, w, j * d, d) + gamma * squared(v, i * q, r, j * q, q);
    p = probabilities(cost, n, m, lambda);
    let distortion = 0, entropy = 0;
    for (let z = 0; z < p.length; z++) { distortion += p[z] * cost[z]; if (p[z] > 0) entropy += p[z] * Math.log(p[z]); }
    const objective = distortion + lambda * entropy;
    history.push(objective);
    if (iteration > 0 && Math.abs(objective - history[iteration - 1]) / Math.max(1, Math.abs(history[iteration - 1])) <= options.tolerance) break;
  }
  return { W: w, P: p, V: v, history: Float64Array.from(history), iterations: history.length,
    labels: labelsFromMembership(p, n, m) };
}

/** Full N² distance matrices, row sorts, dense inverse ranks in both spaces.
 * Rooted direct-difference Float64, self excluded, distance/index ties exactly
 * match the production metric. This is a natural full-sort baseline, not the
 * different version-dependent NumPy/sklearn tie contract.
 */
export function naiveNeighborhood(input, options) {
  const x = normalizeInput(input), y = normalizeInput(options.embedding), n = x.nSamples;
  if (y.nSamples !== n) throw new RangeError('row counts differ');
  const ks = [...new Set(typeof options.ks === 'number' ? [options.ks] : options.ks)];
  const full = z => {
    const distances = new Float64Array(n * n), orders = new Uint32Array(n * n), ranks = new Uint32Array(n * n);
    for (let i = 0; i < n; i++) {
      checkCancelled(options);
      const row = orders.subarray(i * n, (i + 1) * n);
      for (let j = 0; j < n; j++) { row[j] = j; distances[i * n + j] = i === j ? Infinity : Math.sqrt(squared(z.data, i * z.nFeatures, z.data, j * z.nFeatures, z.nFeatures)); }
      row.sort((a, b) => distances[i * n + a] - distances[i * n + b] || a - b);
      for (let rank = 0; rank < n; rank++) ranks[i * n + row[rank]] = rank + 1;
    }
    return { distances, orders, ranks };
  };
  const a = full(x), b = full(y);
  const qualities = ks.map(k => {
    let pt = 0, pc = 0;
    for (let i = 0; i < n; i++) for (let h = 0; h < k; h++) {
      pt += Math.max(0, a.ranks[i * n + b.orders[i * n + h]] - k);
      pc += Math.max(0, b.ranks[i * n + a.orders[i * n + h]] - k);
    }
    const factor = 2 / (n * k * (2 * n - 3 * k - 1));
    return { k, trustworthiness: 1 - pt * factor, continuity: 1 - pc * factor,
      trustworthinessPenalty: pt, continuityPenalty: pc };
  });
  return { qualities };
}

function matrix(n, d, random, discrete = false) {
  const data = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) {
    const value = (random() - 0.5) * 5 + Math.sin((i + 1) * (f + 1) * 0.13);
    data[i * d + f] = discrete ? Math.round(value) : value;
  }
  return {data, nSamples: n, nFeatures: d};
}
function grid(side) {
  const data = new Float64Array(side * side * 2);
  for (let a = 0; a < side; a++) for (let b = 0; b < side; b++) { data[2 * (a * side + b)] = a / Math.max(1, side - 1); data[2 * (a * side + b) + 1] = b / Math.max(1, side - 1); }
  return {data, nSamples: side * side, nFeatures: 2};
}

export const SOM_NEIGHBORHOOD_SIZES = Object.freeze({
  tiny: {somN: 17, somD: 3, side: 2, iterations: 3, metricN: 17, metricDX: 3, ks: [1, 3, 5]},
  demo: {somN: 400, somD: 8, side: 6, iterations: 8, metricN: 700, metricDX: 8, ks: [1, 5, 10]},
  larger: {somN: 2000, somD: 16, side: 8, iterations: 8, metricN: 1600, metricDX: 16, ks: [5, 10, 20]},
});

export function makeSomNeighborhoodCases(size = 'demo') {
  const shape = typeof size === 'string' ? SOM_NEIGHBORHOOD_SIZES[size] : size;
  if (!shape) throw new Error(`Unknown size ${size}`);
  const random = seededRandom(104729), x = matrix(shape.somN, shape.somD, random), r = grid(shape.side);
  const m = r.nSamples, d = x.nFeatures, w = new Float64Array(m * d), cost = new Float64Array(x.nSamples * m);
  // Prepared state is generated independently of either fit kernel, avoiding a
  // hidden production-kernel warmup in first-call measurements.
  for (let j = 0; j < m; j++) { const i = Math.floor(random() * x.nSamples); w.set(x.data.subarray(i * d, (i + 1) * d), j * d); }
  for (let i = 0; i < x.nSamples; i++) for (let j = 0; j < m; j++) {
    let sum = 0; for (let f = 0; f < d; f++) { const delta = x.data[i * d + f] - w[j * d + f]; sum += delta * delta; }
    cost[i * m + j] = sum;
  }
  const options = {grid: r, gamma: 0.5, lambda: 1.2, maxIterations: shape.iterations, tolerance: 0, blockRows: 64};
  // Separate fixture softmax avoids warming the baseline's probability helper.
  const fixtureP = new Float64Array(cost.length);
  for (let i = 0; i < x.nSamples; i++) {
    const offset = i * m; let min = Infinity, sum = 0;
    for (let j = 0; j < m; j++) min = Math.min(min, cost[offset + j]);
    for (let j = 0; j < m; j++) { fixtureP[offset + j] = Math.exp(-(cost[offset + j] - min) / options.lambda); sum += fixtureP[offset + j]; }
    for (let j = 0; j < m; j++) fixtureP[offset + j] /= sum;
  }
  const prepared = {...options, initialPrototypes: w, initialMemberships: fixtureP};
  const mx = matrix(shape.metricN, shape.metricDX, random), my = matrix(shape.metricN, 2, random);
  const tx = matrix(shape.metricN, Math.min(3, shape.metricDX), random, true), ty = matrix(shape.metricN, 2, random, true);
  return {shape, som: {input: x, options: prepared}, somFull: {input: x, options},
    metric: {input: mx, options: {embedding: my, ks: shape.ks, blockRows: 16}},
    metricTies: {input: tx, options: {embedding: ty, ks: shape.ks, blockRows: 16}}};
}

function initPlusKernel(caseData, implementation) {
  // Same public JS PCA stage for both candidates, including its overhead.
  const initial = somOlp(caseData.input, {...caseData.options, maxIterations: 0});
  const options = {...caseData.options, initialPrototypes: initial.W, initialMemberships: initial.P};
  return implementation(caseData.input, options);
}
export function somNeighborhoodJob(cases, algorithm, implementation) {
  const somImpl = implementation === 'baseline' ? naiveSomOlp : somOlp;
  const metricImpl = implementation === 'baseline' ? naiveNeighborhood : neighborhood;
  if (!['baseline', 'optimized'].includes(implementation)) throw new Error('implementation must be baseline or optimized');
  switch (algorithm) {
    case 'som-prepared': return () => somImpl(cases.som.input, cases.som.options);
    case 'som-init-plus-kernel': return () => initPlusKernel(cases.somFull, somImpl);
    case 'neighborhood': return () => metricImpl(cases.metric.input, cases.metric.options);
    case 'neighborhood-ties': return () => metricImpl(cases.metricTies.input, cases.metricTies.options);
    default: throw new Error(`Unknown benchmark algorithm ${algorithm}`);
  }
}
function maxDifference(a, b) {
  if (a === null || b === null) { if (a !== b) throw new Error('null output mismatch'); return 0; }
  if (a.length !== b.length) throw new Error('output shape mismatch');
  let max = 0;
  for (let i = 0; i < a.length; i++) { if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) throw new Error('nonfinite output'); max = Math.max(max, Math.abs(a[i] - b[i])); }
  return max;
}
export function checkSomNeighborhoodParity(algorithm, baseline, optimized, expectedIterations) {
  if (algorithm.startsWith('som')) {
    if (baseline.iterations !== expectedIterations || optimized.iterations !== expectedIterations) throw new Error(`Expected ${expectedIterations} fixed iterations; actual baseline=${baseline.iterations}, optimized=${optimized.iterations}`);
    const errors = Object.fromEntries(['W', 'P', 'V', 'history', 'labels'].map(key => [key, maxDifference(baseline[key], optimized[key])]));
    if (Object.values(errors).some(error => error > 2e-11)) throw new Error(`SOM mismatch ${JSON.stringify(errors)}`);
    return {passed: true, errors, iterations: baseline.iterations};
  }
  if (JSON.stringify(baseline.qualities) !== JSON.stringify(optimized.qualities)) throw new Error('Exact metric result/penalty mismatch');
  return {passed: true, exactQualitiesAndIntegerPenalties: true};
}
function summary(samples) {
  const sorted = [...samples].sort((a, b) => a - b), n = sorted.length;
  return {samplesMs: samples, medianMs: n % 2 ? sorted[n >> 1] : (sorted[(n >> 1) - 1] + sorted[n >> 1]) / 2,
    minMs: sorted[0], maxMs: sorted[n - 1]};
}
function environment() {
  return {node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch,
    timing: 'performance.now wall-clock execution only; process/import/data preparation excluded',
    execution: 'single Node main-thread JavaScript; no Web Worker or browser timing'};
}
const algorithmNames = ['som-prepared', 'som-init-plus-kernel', 'neighborhood', 'neighborhood-ties'];

/** The parent must hold the shared benchmark lease before calling this. */
export function runSomNeighborhoodBenchmarks({size = 'demo', warmups = 3, repetitions = 7, algorithms = algorithmNames} = {}) {
  if (!Number.isSafeInteger(warmups) || warmups < 0 || !Number.isSafeInteger(repetitions) || repetitions < 1) throw new Error('Invalid repeat counts');
  const cases = makeSomNeighborhoodCases(size), results = [];
  for (const algorithm of algorithms) {
    const baseline = somNeighborhoodJob(cases, algorithm, 'baseline'), optimized = somNeighborhoodJob(cases, algorithm, 'optimized');
    const parity = checkSomNeighborhoodParity(algorithm, baseline(), optimized(), cases.shape.iterations);
    for (let i = 0; i < warmups; i++) { baseline(); optimized(); }
    const bSamples = [], oSamples = [];
    for (let i = 0; i < repetitions; i++) {
      const measured = {};
      // Alternate order, with every result validated outside its timing window.
      for (const key of i % 2 ? ['optimized', 'baseline'] : ['baseline', 'optimized']) {
        const start = performance.now(); const result = (key === 'baseline' ? baseline : optimized)(); const elapsed = performance.now() - start;
        (key === 'baseline' ? bSamples : oSamples).push(elapsed); measured[key] = result;
      }
      checkSomNeighborhoodParity(algorithm, measured.baseline, measured.optimized, cases.shape.iterations);
    }
    const b = summary(bSamples), o = summary(oSamples);
    results.push({algorithm, mode: 'warm', warmupsPlusUntimedParity: warmups + 1, parity, baseline: b, optimized: o, baselineOverOptimized: b.medianMs / o.medianMs});
  }
  return {environment: environment(), size, shape: cases.shape, results};
}

/** Invoke once in a fresh process for each algorithm+implementation pair.
 * The opposite implementation is executed ONLY AFTER the timed first call.
 */
export function runSomNeighborhoodFirstCall({size = 'demo', algorithm = 'som-prepared', implementation = 'optimized'} = {}) {
  const cases = makeSomNeighborhoodCases(size), job = somNeighborhoodJob(cases, algorithm, implementation);
  const start = performance.now(); const measured = job(); const elapsedMs = performance.now() - start;
  const otherKey = implementation === 'baseline' ? 'optimized' : 'baseline';
  const other = somNeighborhoodJob(cases, algorithm, otherKey)();
  const baseline = implementation === 'baseline' ? measured : other, optimized = implementation === 'optimized' ? measured : other;
  const parity = checkSomNeighborhoodParity(algorithm, baseline, optimized, cases.shape.iterations);
  return {environment: environment(), size, shape: cases.shape, algorithm, implementation, mode: 'first-call-in-fresh-process', elapsedMs, parity};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2), get = (key, fallback) => { const index = args.indexOf(key); return index < 0 ? fallback : args[index + 1]; };
  const size = get('--size', 'demo'), mode = get('--mode', 'warm');
  let report;
  if (mode === 'verify') {
    const cases = makeSomNeighborhoodCases('tiny');
    report = {kind: 'tiny parity self-test; not benchmark evidence', results: algorithmNames.map(algorithm => ({algorithm, ...checkSomNeighborhoodParity(algorithm, somNeighborhoodJob(cases, algorithm, 'baseline')(), somNeighborhoodJob(cases, algorithm, 'optimized')(), cases.shape.iterations)}))};
  } else if (mode === 'first-call') report = runSomNeighborhoodFirstCall({size, algorithm: get('--algorithm', 'som-prepared'), implementation: get('--implementation', 'optimized')});
  else if (mode === 'warm') report = runSomNeighborhoodBenchmarks({size, warmups: Number(get('--warmups', '3')), repetitions: Number(get('--repetitions', '7'))});
  else throw new Error('mode must be verify, first-call, or warm');
  process.stdout.write(JSON.stringify(report, null, 2) + '\n');
}
