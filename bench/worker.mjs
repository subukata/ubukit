// Run benchmark cases in one Node.js process; bench/run.py drives it.
// Usage: node bench/worker.mjs request.json  (results as JSON on stdout)
// The request names the source tree whose UbuKit to import, so --compare can
// time another commit with this same harness.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const request = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const ub = await import(pathToFileURL(join(request.src, 'js', 'src', 'index.js')).href);
const manifest = JSON.parse(readFileSync(join(request.data, 'manifest.json'), 'utf8'));
const camel = s => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
const local = {
  tpeSphere: (nTrials, { seed }) => ub.minimize(
    p => (p.x - 1) ** 2 + p.y ** 2, { x: ub.uniform(-5, 5), y: ub.uniform(-5, 5) }, { nTrials, seed },
  ),
};

// Fields are loaded once, as matrices or label arrays, outside the timed region.
const cache = new Map();
function resolve(name, value) {
  const spec = manifest[name]?.[value];
  if (typeof value !== 'string' || !spec) return value;
  const key = `${name}/${value}`;
  if (!cache.has(key)) {
    const bytes = readFileSync(join(request.data, name, `${value}.bin`));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length);
    const array = spec.dtype === 'int32' ? new Int32Array(buffer) : new Float64Array(buffer);
    cache.set(key, spec.shape.length === 2 ? { data: array, rows: spec.shape[0], cols: spec.shape[1] } : array);
  }
  return cache.get(key);
}

const results = {};
for (const c of request.cases) {
  const fn = local[camel(c.method)] ?? ub[camel(c.method)];
  if (!fn) continue;
  const args = c.args.map(a => resolve(c.data, a));
  const options = Object.fromEntries(Object.entries(c.options).map(([k, v]) => [camel(k), resolve(c.data, v)]));
  let variants = [options];
  if (request.seeds && c.seeded) {
    const { init, seed, ...fixed } = options;
    variants = request.seeds.map(s => ({ ...fixed, seed: s }));
  }
  try {
    await fn(...args, variants[0]); // warm-up for the JIT
    const seconds = [], outputs = [];
    for (const v of variants) {
      let result;
      for (let r = 0; r < (variants.length === 1 ? request.repeat : 1); r++) {
        const start = performance.now();
        result = await fn(...args, v);
        seconds.push((performance.now() - start) / 1000);
      }
      outputs.push(summary(result));
    }
    results[c.name] = { seconds, outputs };
  } catch (error) {
    results[c.name] = { error: `${error.name}: ${error.message}` };
  }
}
console.log(JSON.stringify({ env: { runtime: `Node ${process.version}`, module: request.src }, results }));

function summary(result) {
  if (typeof result === 'number') return { value: result };
  if ('bestValue' in result) return { value: result.bestValue };
  return { labels: Array.from(result.labels), centers: ub.toRows(result.centers), n_iter: result.nIter };
}
