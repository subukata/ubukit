import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';

const space = { x: ub.uniform(-5, 5), lr: ub.loguniform(1e-5, 1e-1), n: ub.integer(1, 20), kind: ub.choice('a', 'b', 'c') };
const f = p => (p.x - 1.5) ** 2 + Math.log10(p.lr / 1e-3) ** 2 + (p.n - 7) ** 2 + (p.kind !== 'b');

/** n trials of a search; the parameters asked. */
function askAndTell(tpe, objective, n) {
  const asked = [];
  for (let t = 0; t < n; t++) {
    const p = tpe.ask();
    tpe.tell(p, objective(p));
    asked.push(p);
  }
  return asked;
}

test('tpe minimizes a mixed objective', async () => {
  const best = [];
  for (let seed = 0; seed < 5; seed++) best.push((await ub.minimize(f, space, { nTrials: 80, seed })).bestValue);
  const rnd = [];
  for (let seed = 0; seed < 5; seed++) rnd.push((await ub.minimize(f, space, { nTrials: 80, seed, nStartup: 80 })).bestValue);
  const median = a => a.sort((x, y) => x - y)[2];
  assert.ok(median(best) < 0.25 * median(rnd), `${median(best)} vs ${median(rnd)}`);
  const tpe = new ub.TPE(space, { seed: 1 });
  for (let t = 0; t < 30; t++) {
    const p = tpe.ask();
    assert.ok(Number.isInteger(p.n) && p.n >= 1 && p.n <= 20 && p.lr >= 1e-5 && p.lr <= 1e-1);
    tpe.tell(p, f(p));
  }
  assert.throws(() => tpe.tell({ x: 0 }, 1), RangeError);
  // The keys must be the params' own: an inherited x is not a value for x.
  const inherited = Object.assign(Object.create({ x: 0.5 }), { y: 1 });
  assert.throws(() => new ub.TPE({ x: ub.uniform(0, 1) }).tell(inherited, 1), RangeError);
  const p = tpe.ask();
  for (const bad of [{ x: 99 }, { x: '0' }, { n: 2.5 }, { kind: 'z' }]) assert.throws(() => tpe.tell({ ...p, ...bad }, 1), RangeError);
  assert.throws(() => ub.uniform(1, 1), RangeError);
  assert.throws(() => ub.uniform(-1e308, 1e308), RangeError);
  assert.throws(() => ub.integer(0, 2 ** 60), /9007199254740991/);
  const many = new ub.TPE({ x: ub.uniform(0, 1) }, { seed: 0 });
  for (let t = 0; t < 200_000; t++) many.tell({ x: 0.5 }, 200_000 - t);
  assert.equal(many.result().bestValue, 1);
});

test('a search resumes from its trials', () => {
  // A new TPE with the seed, told the trials of a search, an external one and
  // tied values included, asks what the search asked next, whether it stops
  // before, at or after nStartup.
  const tied = p => Math.round(f(p));
  const tpe = new ub.TPE(space, { seed: 11, nStartup: 6 });
  tpe.tell({ x: 0, lr: 1e-3, n: 7, kind: 'a' }, 2);
  askAndTell(tpe, tied, 29);
  const done = tpe.result();
  assert.equal(new Set(done.params.map(p => p.x)).size, 30);  // each history has its own stream
  for (const k of [3, 6, 7, 18]) {
    const resumed = new ub.TPE(space, { seed: done.seed, nStartup: 6 });
    for (let i = 0; i < k; i++) resumed.tell(done.params[i], done.values[i]);
    assert.deepEqual(askAndTell(resumed, tied, 30 - k), done.params.slice(k), `${k}`);
  }
});

test('asks not told leave the next tell alone', () => {
  // Asks after a tell continue one stream; the next tell starts the stream of
  // the new history, however many asks were left untold.
  const a = new ub.TPE(space, { seed: 4, nStartup: 3 }), b = new ub.TPE(space, { seed: 4, nStartup: 3 });
  for (const tpe of [a, b]) askAndTell(tpe, f, 5);
  const pending = [a.ask(), a.ask(), a.ask()];
  assert.notDeepEqual(pending[1], pending[0]);
  assert.deepEqual(b.ask(), pending[0]);
  for (const tpe of [a, b]) tpe.tell({ x: 1, lr: 1e-3, n: 7, kind: 'b' }, 0.5);
  assert.deepEqual(askAndTell(a, f, 5), askAndTell(b, f, 5));
});

test('a failed tell and result leave the stream alone', () => {
  const a = new ub.TPE(space, { seed: 2, nStartup: 3 }), b = new ub.TPE(space, { seed: 2, nStartup: 3 });
  for (const tpe of [a, b]) askAndTell(tpe, f, 4);
  const params = a.ask();
  assert.deepEqual(b.ask(), params);
  b.result();
  assert.throws(() => b.tell({ ...params, x: 99 }, 1), RangeError);
  const next = a.ask();
  assert.notDeepEqual(next, params);
  assert.deepEqual(b.ask(), next);
  assert.deepEqual(askAndTell(a, f, 4), askAndTell(b, f, 4));
});

test('a drawn seed is kept', async () => {
  const tpe = new ub.TPE(space, { nStartup: 4 });
  assert.ok(Number.isInteger(tpe.seed) && tpe.seed >= 0 && tpe.seed < 2 ** 32);
  const asked = askAndTell(tpe, f, 8);
  assert.equal(tpe.result().seed, tpe.seed);
  assert.deepEqual(askAndTell(new ub.TPE(space, { seed: tpe.seed, nStartup: 4 }), f, 8), asked);
  const best = await ub.minimize(f, space, { nTrials: 8 });
  assert.deepEqual((await ub.minimize(f, space, { nTrials: 8, seed: best.seed })).params, best.params);
});

test('a space too narrow to search is refused', () => {
  // Its logarithms coincide, so the third ask, the first from the model, gave NaN.
  assert.throws(() => ub.loguniform(1000, 1000.0000000000001), /too narrow to search on a log scale/);
  assert.throws(() => new ub.TPE({ x: ub.uniform(0, 1) }).tell({ x: 0.5 }, Infinity), { message: 'value must be a finite number' });
});

test('a NaN option is told back as ask gave it', () => {
  // indexOf never finds NaN, so the TPE refused its own proposal; Python finds
  // the option by identity.
  const tpe = new ub.TPE({ x: ub.choice(NaN, 1) }, { seed: 0, nStartup: 2 });
  for (let t = 0; t < 6; t++) {
    const p = tpe.ask();
    tpe.tell(p, Number.isNaN(p.x) ? 0 : 1);
  }
  assert.ok(Number.isNaN(tpe.result().bestParams.x));
});

test('the settings of a TPE are read-only', () => {
  // They were plain properties: nCandidates = 0 set later skipped the check and
  // made the next proposal fail.
  const tpe = new ub.TPE({ x: ub.uniform(0, 1) }, { nStartup: 3, nCandidates: 5, gamma: 0.2 });
  assert.deepEqual([tpe.nStartup, tpe.nCandidates, tpe.gamma], [3, 5, 0.2]);
  for (const name of ['seed', 'nStartup', 'nCandidates', 'gamma']) assert.throws(() => { /** @type {any} */ (tpe)[name] = 0; }, TypeError, name);
});

test('a TPE keeps checked copies of its dimensions', () => {
  // Changing a dimension after the TPE was made changed its proposals; the
  // factories' dimensions are frozen, as Python's are, and a dimension written
  // by hand is copied and checked as its factory would.
  const made = ub.uniform(0, 1), options = ub.choice('a', 'b');
  assert.throws(() => { /** @type {any} */ (made).low = 10; }, TypeError);
  assert.throws(() => { options.options.push('c'); }, TypeError);
  const hand = { kind: 'numeric', low: 0, high: 1, log: false, integer: false };
  const tpe = new ub.TPE({ x: /** @type {any} */ (hand) }, { seed: 0, nStartup: 2 });
  Object.assign(hand, { low: 10, high: 20 });
  for (let t = 0; t < 5; t++) {
    const p = tpe.ask();
    assert.ok(p.x >= 0 && p.x <= 1, `${p.x}`);
    tpe.tell(p, Number(p.x));
  }
  assert.throws(() => new ub.TPE({ x: /** @type {any} */ ({ ...hand, low: 30 }) }), /high/);
  assert.throws(() => new ub.TPE({ x: /** @type {any} */ ({ kind: 'numeric', low: 1000, high: 1000.0000000000001, log: true, integer: false }) }), /too narrow/);
  // Its flags must be booleans, as the factories' are: a string 'false' is truthy.
  assert.throws(() => new ub.TPE({ x: /** @type {any} */ ({ ...hand, log: 'false' }) }), /space.x.log must be a boolean/);
  assert.throws(() => new ub.TPE({ x: /** @type {any} */ ({ kind: 'numeric', low: 0, high: 1, log: false }) }), /space.x.integer must be a boolean/);
});

test('flags must be booleans', () => {
  // integer(1, 8, { log: 'false' }) searched on a log scale, and shuffle: 'false' shuffled.
  assert.throws(() => ub.integer(1, 8, { log: /** @type {any} */ ('false') }), { name: 'RangeError', message: 'log must be a boolean' });
  assert.throws(() => ub.som([[0, 0], [1, 1], [2, 0]], [1, 2], { shuffle: /** @type {any} */ ('false') }), /shuffle must be a boolean/);
  assert.equal(ub.integer(1, 8, { log: true }).log, true);
});

test('a categorical dimension may have any number of options', () => {
  // Copying a dimension spread its options into arguments, which overflowed the stack at 200,000.
  const tpe = new ub.TPE({ k: /** @type {any} */ ({ kind: 'choice', options: Array.from({ length: 200_000 }, (_, i) => i) }) }, { seed: 0 });
  const p = tpe.ask();
  assert.ok(Number.isInteger(p.k) && p.k >= 0 && p.k < 200_000);
  tpe.tell(p, 1);
  // A value that is not an option is named in the error, a symbol too.
  assert.throws(() => new ub.TPE({ x: ub.choice(1, 2) }).tell({ x: Symbol('s') }, 1), { name: 'RangeError', message: 'Symbol(s) is not an option' });
});
