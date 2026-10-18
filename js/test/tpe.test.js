import assert from 'node:assert/strict';
import test from 'node:test';
import * as ub from '../src/index.js';

test('tpe minimizes a mixed objective', async () => {
  const space = { x: ub.uniform(-5, 5), lr: ub.loguniform(1e-5, 1e-1), n: ub.integer(1, 20), kind: ub.choice('a', 'b', 'c') };
  const f = p => (p.x - 1.5) ** 2 + Math.log10(p.lr / 1e-3) ** 2 + (p.n - 7) ** 2 + (p.kind !== 'b');
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
  for (const name of ['nStartup', 'nCandidates', 'gamma']) assert.throws(() => { /** @type {any} */ (tpe)[name] = 0; }, TypeError, name);
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
  assert.throws(() => new ub.TPE({ x: /** @type {any} */ ({ kind: 'numeric', low: 1000, high: 1000.0000000000001, log: true }) }), /too narrow/);
});
