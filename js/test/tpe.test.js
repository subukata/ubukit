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
