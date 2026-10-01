/** Reusable latency benchmark; prints JSON to stdout, writes no repository files.
 * node --expose-gc bench/realtime.js [--quick] [--baseline=/path/to/old/src/index.js]
 * CPU affinity and exclusive-machine policy belong to the caller, not this file.
 */
import { performance } from 'node:perf_hooks';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { createSession, run, seededRandom } from '../src/index.js';
const quick = process.argv.includes('--quick');
const repeats = quick ? 5 : 21, coldRepeats = quick ? 3 : 9;
const cases = [
  { algorithm: 'kmeans', n: 4096, d: 4, k: 8 },
  { algorithm: 'fcm', n: 4096, d: 4, k: 8 },
  { algorithm: 'exrcm', n: 2048, d: 2, k: 6 },
  { algorithm: 'rmcm', n: 384, d: 2, k: 4 },
  { algorithm: 'som-olp', n: 2048, d: 4, k: 8 },
  { algorithm: 'som-olp', n: 512, d: 16, k: 8, pca: true }
];
function setup(c) {
  const rng = seededRandom(24), data = new Float64Array(c.n * c.d), centers = new Float64Array(c.k * c.d);
  for (let j = 0; j < c.k; ++j) for (let f = 0; f < c.d; ++f) centers[j * c.d + f] = Math.sin(j * 2.7 + f) * 3;
  for (let i = 0; i < c.n; ++i) for (let f = 0; f < c.d; ++f) data[i * c.d + f] = centers[(i % c.k) * c.d + f] + (rng() - .5) * 4;
  const input = { data, nSamples: c.n, nFeatures: c.d }, options = { nClusters: c.k, initCenters: centers, maxIterations: 8, tolerance: 0, blockRows: 64 };
  if (c.algorithm === 'rmcm') Object.assign(options, { delta: .3, graphBatchPairs: 256 });
  if (c.algorithm === 'som-olp') {
    options.grid = { data: Float64Array.from({ length: c.k * 2 }, (_, i) => i % 2 ? Math.floor(i / 2) % 2 : Math.floor(i / 4)), nSamples: c.k, nFeatures: 2 };
    if (c.pca) delete options.initCenters;
  }
  return { input, options };
}
function measure(fn) { const t = performance.now(); const value = fn(); return { ms: performance.now() - t, value }; }
function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return { count: values.length, p50Ms: sorted[Math.floor((sorted.length - 1) * .5)] ?? null,
    p95Ms: sorted[Math.ceil((sorted.length - 1) * .95)] ?? null, maxMs: sorted.at(-1) ?? null, samplesMs: values };
}
function fit(session, durations = [], snapshots = []) {
  let count = 0;
  while (!session.status.done) {
    const r = session.step(1, { timeBudgetMs: 2, maxChunks: 128 }); durations.push(r.elapsedMs);
    if (r.completedIterations > 0) snapshots.push(measure(() => session.snapshot()).ms);
    if (++count > 100000) throw new Error('Non-terminating session');
  }
  return session.snapshot().result;
}
const coldIndex = process.argv.find(a => a.startsWith('--cold='));
if (coldIndex) {
  const c = cases[Number(coldIndex.split('=')[1])], { input, options } = setup(c);
  const create = measure(() => createSession(c.algorithm, input, options)), session = create.value;
  const stepTimes = [], snapshotTimes = [], firstStep = measure(() => session.step(1, { timeBudgetMs: 2, maxChunks: 128 }));
  const rest = measure(() => fit(session, stepTimes, snapshotTimes));
  console.log(JSON.stringify({ createMs: create.ms, firstStepMs: firstStep.ms, fitIncludingCreateMs: create.ms + firstStep.ms + rest.ms, maxStepMs: Math.max(firstStep.ms, ...stepTimes) }));
} else {
  const baselineArg = process.argv.find(a => a.startsWith('--baseline='));
  const baseline = baselineArg ? await import(pathToFileURL(baselineArg.slice(11)).href) : null;
  const rows = [];
  for (let ci = 0; ci < cases.length; ++ci) {
    const c = cases[ci], { input, options } = setup(c);
    const expected = run(c.algorithm, input, options);
    assert.deepEqual(fit(createSession(c.algorithm, input, options)), expected);
    for (let w = 0; w < 5; ++w) { run(c.algorithm, input, options); const s = createSession(c.algorithm, input, options); fit(s); s.dispose(); }
    const oneShot = [], oldOneShot = [], creation = [], sessionTotal = [], stepTimes = [], snapshotTimes = [], dataUpdates = [], parameterUpdates = [], warmFit = [];
    for (let r = 0; r < repeats; ++r) {
      // Alternate baseline ordering to reduce systematic order bias.
      if (baseline && r % 2 === 0) oldOneShot.push(measure(() => baseline.run(c.algorithm, input, options)).ms);
      oneShot.push(measure(() => run(c.algorithm, input, options)).ms);
      if (baseline && r % 2) oldOneShot.push(measure(() => baseline.run(c.algorithm, input, options)).ms);
      const created = measure(() => createSession(c.algorithm, input, options)), session = created.value; creation.push(created.ms);
      const fitted = measure(() => fit(session, stepTimes, snapshotTimes)); sessionTotal.push(created.ms + fitted.ms);
      parameterUpdates.push(measure(() => session.updateParameters({ maxIterations: 8 })).ms);
      warmFit.push(measure(() => fit(session)).ms);
      const moved = { ...input, data: Float64Array.from(input.data, (x, i) => x + Math.sin(i + r) * .01) };
      dataUpdates.push(measure(() => session.updateData(moved)).ms); fit(session); session.dispose();
    }
    const cold = [];
    for (let r = 0; r < coldRepeats; ++r) {
      const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url), `--cold=${ci}`], { encoding: 'utf8' });
      if (child.status !== 0) throw new Error(child.stderr); cold.push(JSON.parse(child.stdout));
    }
    global.gc?.();
    const before = process.memoryUsage();
    const held = createSession(c.algorithm, input, options); fit(held);
    for (let r = 0; r < 50; ++r) { held.updateParameters({ maxIterations: 2 }); fit(held); }
    global.gc?.(); const after50Updates = process.memoryUsage(); held.dispose(); global.gc?.(); const afterDispose = process.memoryUsage();
    rows.push({ ...c, iterations: expected.iterations, maxIterations: 8, blockRows: 64, timeBudgetMs: 2,
      oneShot: stats(oneShot), ...(baseline ? { previousProductionOneShot: stats(oldOneShot) } : {}),
      creation: stats(creation), sessionFitIncludingSnapshots: stats(sessionTotal), chunksIncludingCheckpointCopies: stats(stepTimes), explicitSnapshot: stats(snapshotTimes),
      updateDataCopyCompareValidate: stats(dataUpdates), updateParameters: stats(parameterUpdates), warmStateFitDifferentWorkload: stats(warmFit),
      freshProcess: { create: stats(cold.map(x => x.createMs)), firstStep: stats(cold.map(x => x.firstStepMs)), total: stats(cold.map(x => x.fitIncludingCreateMs)), maximumChunk: stats(cold.map(x => x.maxStepMs)) },
      memoryObservation: { gcAvailable: !!global.gc, before, after50Updates, afterDispose } });
  }
  console.log(JSON.stringify({ schema: 1, runtime: process.version, repeats, coldRepeats,
    methodology: 'Exact same initialization/iterations for run vs segmented session. Session total includes constructor, all internal complete-checkpoint copies, and explicit per-iteration snapshot copies. Warm-state fits have different numerical starting states/work and are not speedup baselines. Soft budgets checked after generator chunks; initialization/finalization/checkpoint copies can exceed them. Cold samples use a fresh process per case; module import/process startup excluded. Raw values are observations, not hard latency guarantees.', rows }, null, 2));
}
