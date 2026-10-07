// The JavaScript port must have Python's public names and reproduce its reference
// results, both in fixtures/fixtures.json.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import * as ub from '../src/index.js';

const { api, data, cases } = JSON.parse(readFileSync(new URL('../../fixtures/fixtures.json', import.meta.url), 'utf8'));
const camel = s => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
const resolve = v => (typeof v === 'string' && v in data ? data[v] : v);

function close(actual, expected, label, tol = 1e-8) {
  const a = actual.data ?? actual, e = expected.flat();
  assert.equal(a.length, e.length, `${label}: length`);
  for (let i = 0; i < e.length; i++) {
    assert.ok(Math.abs(a[i] - e[i]) <= tol * (1 + Math.abs(e[i])), `${label}[${i}]: ${a[i]} vs ${e[i]}`);
  }
}

test('the public names are those of Python, in camelCase', () => {
  // Python's classes are types here (typing-api.js uses them); run, runAsync,
  // matrix and toRows are JavaScript's own (DESIGN.md, principle 4).
  const types = ['Progress', 'Result', 'TPEResult'], own = ['matrix', 'run', 'runAsync', 'toRows'];
  const python = api.ubukit.filter(n => !types.includes(n)).map(camel);
  assert.deepEqual(Object.keys(ub).filter(n => !own.includes(n)).sort(), python.sort());
  assert.deepEqual(Object.keys(ub.steps).sort(), api.steps.map(camel).sort());
});

for (const c of cases) {
  const options = Object.fromEntries(Object.entries(c.options).map(([k, v]) => [camel(k), resolve(v)]));
  test(`${c.method} ${JSON.stringify(c.options)}`, () => {
    const fn = ub[camel(c.method)];
    if (c.X) {
      const r = fn(data[c.X], ...c.args, options);
      close(r.centers, c.expect.centers, 'centers');
      assert.deepEqual(Array.from(r.labels), c.expect.labels);
      if (c.expect.membership) close(r.membership, c.expect.membership, 'membership');
      if (c.expect.embedding) close(r.embedding, c.expect.embedding, 'embedding');
    } else {
      close([fn(...c.args.map(resolve), options)], [c.expect], c.method, 1e-10);
    }
  });
}
