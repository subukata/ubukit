import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { Worker } from 'node:worker_threads';
import * as candidate from '../candidate_package/src/index.js';
import * as baseline from '../baseline_package/src/index.js';
import { SESSION_CHECKPOINT as candidateHook } from '../candidate_package/src/session-hooks.js';
import { SESSION_CHECKPOINT as baselineHook } from '../baseline_package/src/session-hooks.js';

const input = values => ({ data: Float64Array.from(values), nSamples: values.length, nFeatures: 1 });
const cases = [
  { name: 'ordinary FCM', algorithm: 'fcm', x: input([-2, -1, 0, 1, 3]), options: { nClusters: 2, initCenters: Float64Array.of(-1, 2), m: 2, maxIterations: 4, tolerance: 0, blockRows: 1, returnHistory: true } },
  { name: 'stable FCM', algorithm: 'fcm', x: input([-1, 1]), options: { initCenters: Float64Array.of(-.5, .5), m: 1e100, maxIterations: 4, tolerance: 0, blockRows: 1, returnHistory: true } },
  { name: 'retry FCM', algorithm: 'fcm', x: input([0, 1e-139]), options: { initMembership: Float64Array.of(1, 1e-10, 1, 1e-100), m: 2, maxIterations: 4, tolerance: 0, blockRows: 1, returnHistory: true } },
  { name: 'SOM-OLP', algorithm: 'som-olp', x: input([-2, -1, 0, 1, 3]), options: { grid: input([0, 1]), initialPrototypes: Float64Array.of(-1, 2), maxIterations: 4, tolerance: 0, blockRows: 1 } }
];

// Count only plain-array Float64Array.from calls from the history-producing
// kernels. Inputs/prototypes are typed arrays; this observes real allocations.
function countHistoryCopies(fn) {
  const original = Float64Array.from;
  let copies = 0;
  Float64Array.from = function(source, ...rest) {
    if (Array.isArray(source) && /(?:clustering|fcm-stable|som-olp)\.js/.test(new Error().stack)) copies++;
    return original.call(this, source, ...rest);
  };
  try { return { value: fn(), copies }; } finally { Float64Array.from = original; }
}
function trace(lib, symbol, c, mode = 'none') {
  const options = { ...c.options }, events = [], callbacks = [], histories = [];
  let it, yielded = 0, centerYields = 0;
  const hook = function(state, origin) {
    assert.equal(this, options, 'hook receiver remains the live options object');
    callbacks.push(structuredClone({ state, origin }));
    const history = state.objectiveHistory ?? state.history;
    histories.push(history);
    if (mode === 'mutate-diagnostic') state.membershipConvergenceOnly = 'hook-value';
    if (mode === 'late-remove') delete options[symbol];
  };
  if (mode === 'all' || mode === 'mutate-diagnostic') options[symbol] = hook;
  it = lib.steps(c.algorithm, c.x, options);
  for (;;) {
    const next = it.next();
    if (next.done) return { result: next.value, events, callbacks, histories };
    events.push(structuredClone(next.value));
    yielded++;
    if ((mode === 'late' || mode === 'late-remove') && yielded === 1) options[symbol] = hook;
    if (mode === 'late-iteration' && next.value.iteration === 1 && (next.value.phase === 'iteration' || next.value.algorithm === 'fcm' && next.value.phase == null)) options[symbol] = hook;
    // The fixture yields two ordinary center blocks, then fails in the
    // membership path; the third center block is the stable retry's first.
    if (next.value.phase === 'centers') centerYields++;
    if (mode === 'retry-late' && c.name === 'retry FCM' && centerYields === 3) options[symbol] = hook;
  }
}
for (const c of cases) {
  test(`${c.name}: no hook makes no intermediate history copies; final output and yields exact`, t => {
    const old = countHistoryCopies(() => trace(baseline, baselineHook, c));
    const next = countHistoryCopies(() => trace(candidate, candidateHook, c));
    assert.ok(old.copies > 1, 'baseline must exhibit unused intermediate history copies');
    assert.equal(next.copies, 1, 'only the required final history is copied');
    assert.deepEqual(next.value, old.value);
    t.diagnostic(`history-copy-count ${c.name}: baseline=${old.copies}, candidate=${next.copies}`);
  });
  for (const mode of ['all', 'late', 'late-iteration', 'late-remove']) test(`${c.name}: ${mode} hooks preserve checkpoints, ownership, yields and final output`, () => {
    const old = trace(baseline, baselineHook, c, mode), next = trace(candidate, candidateHook, c, mode);
    assert.ok(next.callbacks.length > 0);
    if (mode === 'late-remove') assert.equal(next.callbacks.length, 1);
    assert.deepEqual(next, old);
    const finalHistory = next.result.objectiveHistory ?? next.result.history;
    for (let i = 0; i < next.histories.length; i++) {
      assert.notEqual(next.histories[i], finalHistory);
      for (let j = i + 1; j < next.histories.length; j++) assert.notEqual(next.histories[i], next.histories[j]);
    }
    const saved = finalHistory.slice();
    next.histories[0].fill(12345);
    assert.deepEqual(finalHistory, saved, 'retained checkpoint histories never alias final history');
  });
  test(`${c.name}: cancellation checks and cooperative completion are unchanged`, () => {
    function cancelled(lib) {
      let checks = 0;
      try { lib.run(c.algorithm, c.x, { ...c.options, shouldCancel: () => ++checks > 8 }); }
      catch (error) { return { checks, name: error.name, message: error.message }; }
      assert.fail('test must cancel');
    }
    assert.deepEqual(cancelled(candidate), cancelled(baseline));
    const session = candidate.createSession(c.algorithm, c.x, c.options);
    let chunks = 0;
    while (!session.status.done) { session.step(1, { timeBudgetMs: 0, maxChunks: 1 }); assert.ok(++chunks < 1000); }
    assert.deepEqual(session.snapshot().result, candidate.run(c.algorithm, c.x, c.options));
    session.dispose();
  });
}
test('retry FCM: hook added after entering stable retry remains live', () => {
  const c = cases.find(c => c.name === 'retry FCM');
  const next = trace(candidate, candidateHook, c, 'retry-late');
  assert.ok(next.callbacks.length > 0);
  assert.deepEqual(next, trace(baseline, baselineHook, c, 'retry-late'));
});
test('stable FCM: checkpoint mutation of existing event diagnostic is retained', () => {
  const c = cases.find(c => c.name === 'stable FCM');
  const next = trace(candidate, candidateHook, c, 'mutate-diagnostic');
  assert.deepEqual(next, trace(baseline, baselineHook, c, 'mutate-diagnostic'));
  assert.ok(next.events.some(e => e.membershipConvergenceOnly === 'hook-value'));
});

// Execute the unmodified worker module text with mock dependencies to observe
// snapshot/transfer calls directly, without exposing test hooks in production.
async function workerHarness(reportProgress, cancelAfterFirst = false) {
  const timers = new Map(), messages = [];
  let listener, timerId = 0, steps = 0, snapshots = 0, transfers = 0, clocks = 0;
  const session = { status: { done: false, iteration: 0 }, step() { steps++; this.status.iteration++; this.status.done = steps === 3; }, snapshot() { snapshots++; return { result: { centers: Float64Array.of(7) }, iteration: this.status.iteration }; }, dispose() {} };
  const context = vm.createContext({
    self: { addEventListener(type, fn) { assert.equal(type, 'message'); listener = fn; }, postMessage(message) { messages.push(message); } },
    setTimeout(fn) { timers.set(++timerId, fn); return timerId; }, clearTimeout(id) { timers.delete(id); },
    performance: { now() { clocks++; return 100; } }
  });
  const module = new vm.SourceTextModule(readFileSync(new URL('../candidate_package/src/realtime-worker.js', import.meta.url), 'utf8'), { context });
  await module.link(specifier => {
    if (specifier === './session.js') return new vm.SyntheticModule(['createSession'], function() { this.setExport('createSession', () => session); }, { context });
    assert.equal(specifier, './core.js');
    return new vm.SyntheticModule(['collectTransferables', 'finiteNumber', 'positiveInteger'], function() {
      this.setExport('collectTransferables', () => { transfers++; return []; });
      this.setExport('finiteNumber', value => value); this.setExport('positiveInteger', value => value);
    }, { context });
  });
  await module.evaluate();
  listener({ data: { type: 'run', id: 11, algorithm: 'fcm', input: {}, controls: { progressIntervalMs: 0 }, ...(reportProgress === undefined ? {} : { reportProgress }) } });
  while (timers.size) {
    const [id, fn] = timers.entries().next().value; timers.delete(id); fn();
    if (cancelAfterFirst && steps === 1) listener({ data: { type: 'cancel', id: 11 } });
    assert.ok(steps <= 3);
  }
  return { steps, snapshots, transfers, clocks, messages };
}
test('realtime worker: no subscriber performs zero progress snapshots/transfers/clock reads', async () => {
  const result = await workerHarness(false);
  assert.equal(result.steps, 3); assert.equal(result.snapshots, 1); assert.equal(result.transfers, 1); assert.equal(result.clocks, 0);
  assert.deepEqual(result.messages.map(m => m.type), ['result']);
});
for (const flag of [true, undefined]) test(`realtime worker: requested/legacy progress is retained (${flag})`, async () => {
  const result = await workerHarness(flag);
  assert.equal(result.snapshots, 3); assert.equal(result.transfers, 3);
  assert.deepEqual(result.messages.map(m => m.type), ['progress', 'progress', 'result']);
});
test('realtime worker: cancellation without subscriber remains cooperative and snapshot-free', async () => {
  const result = await workerHarness(false, true);
  assert.equal(result.steps, 1); assert.equal(result.snapshots, 0); assert.equal(result.transfers, 0);
  assert.deepEqual(result.messages.map(m => m.type), ['cancelled']);
});
for (const subscribed of [false, true]) test(`real realtime client/worker: subscribed=${subscribed}, output/input ownership exact`, async () => {
  const c = cases[0], raw = [], callbacks = [], sent = [], ownedInput = c.x.data.slice();
  const client = candidate.createRealtimeWorkerClient({ workerFactory: () => {
    const worker = new Worker(new URL('../candidate_package/src/realtime-worker.js', import.meta.url), { type: 'module' });
    worker.on('message', message => raw.push(message));
    const post = worker.postMessage.bind(worker);
    worker.postMessage = function(message, ...rest) { sent.push(structuredClone(message)); return post(message, ...rest); };
    return worker;
  } });
  try {
    const result = await client.run(c.algorithm, c.x, { ...c.options, ...(subscribed ? { onProgress: snapshot => { callbacks.push(structuredClone(snapshot)); snapshot.result.centers.fill(1e50); } } : {}) }, { warmStart: false, timeBudgetMs: 0, maxChunks: 1, progressIntervalMs: 0 });
    assert.equal(sent.find(m => m.type === 'run').reportProgress, subscribed);
    assert.deepEqual(result.result, baseline.run(c.algorithm, c.x, c.options));
    assert.deepEqual(c.x.data, ownedInput);
    assert.equal(raw.filter(m => m.type === 'result').length, 1);
    if (subscribed) { assert.ok(callbacks.length > 0); assert.equal(raw.filter(m => m.type === 'progress').length, callbacks.length); }
    else { assert.equal(callbacks.length, 0); assert.equal(raw.filter(m => m.type === 'progress').length, 0); }
  } finally { client.dispose(); }
});

for (const c of cases) test(`${c.name}: checkpoint accessor is read once per commit; single-use hook survives`, () => {
  function capture(lib, symbol) {
    const options = { ...c.options }, callbacks = [];
    let reads = 0, armed = true;
    Object.defineProperty(options, symbol, { enumerable: true, configurable: true, get() {
      reads++;
      if (!armed) return undefined;
      armed = false;
      return function(state, origin) {
        assert.equal(this, options);
        callbacks.push(structuredClone({ state, origin }));
      };
    } });
    const result = lib.run(c.algorithm, c.x, options);
    return { result, reads, callbacks };
  }
  const old = capture(baseline, baselineHook), next = capture(candidate, candidateHook);
  assert.deepEqual(next, old);
  // In retry FCM the pre-existing option spread reads/consumes the one-shot
  // accessor before the retried checkpoint. Both implementations retain that.
  assert.equal(next.callbacks.length, c.name === 'retry FCM' ? 0 : 1);
});
test('retry FCM: a hook accessor armed after retry entry is read once per checkpoint', () => {
  function capture(lib, symbol) {
    const c = cases.find(c => c.name === 'retry FCM'), options = { ...c.options }, callbacks = [];
    let reads = 0, armed = false, installed = false, centerYields = 0;
    const iterator = lib.steps(c.algorithm, c.x, options);
    for (;;) {
      const next = iterator.next();
      if (next.done) return { result: next.value, callbacks, reads };
      if (next.value.phase === 'centers') centerYields++;
      if (!installed && centerYields === 3) {
        installed = true; armed = true;
        Object.defineProperty(options, symbol, { get() { reads++; if (!armed) return undefined; armed = false; return function(state, origin) { assert.equal(this, options); callbacks.push(structuredClone({ state, origin })); }; } });
      }
    }
  }
  const old = capture(baseline, baselineHook), next = capture(candidate, candidateHook);
  assert.deepEqual(next, old); assert.equal(next.callbacks.length, 1);
});
