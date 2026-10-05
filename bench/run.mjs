// Runs the cases prepared by bench/run.py and prints timings and results as JSON.
import { readFileSync } from 'node:fs';
import * as ub from '../js/src/index.js';

const { data, cases, repeat } = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const camel = s => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
// Matrices are converted once, outside the timed region, as NumPy arrays are in Python.
const inputs = Object.fromEntries(Object.entries(data).map(([k, v]) => [k, Array.isArray(v[0]) ? ub.matrix(v) : v]));
const resolve = v => (typeof v === 'string' && v in inputs ? inputs[v] : v);
const local = {
  tpeSphere: (nTrials, { seed }) => ub.minimize(
    p => (p.x - 1) ** 2 + p.y ** 2, { x: ub.uniform(-5, 5), y: ub.uniform(-5, 5) }, { nTrials, seed },
  ),
};

const out = {};
for (const c of cases) {
  const fn = local[camel(c.method)] ?? ub[camel(c.method)];
  const args = [...(c.X === null ? [] : [inputs[c.X]]), ...c.args.map(resolve)];
  const options = Object.fromEntries(Object.entries(c.options).map(([k, v]) => [camel(k), resolve(v)]));
  let best = Infinity, result;
  for (let r = 0; r < repeat; r++) {
    const start = performance.now();
    result = await fn(...args, options);
    best = Math.min(best, (performance.now() - start) / 1000);
  }
  out[c.name] = { seconds: best, ...summary(result) };
}
console.log(JSON.stringify(out));

function summary(result) {
  if (typeof result === 'number') return { value: result };
  if ('bestValue' in result) return { value: result.bestValue };
  return { labels: Array.from(result.labels), centers: ub.toRows(result.centers) };
}
