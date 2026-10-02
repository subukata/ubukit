// Recreate expected public results from the pinned pre-fix JavaScript checkout.
// Usage: node javascript/tools/generate-warm-preflight-reference.mjs /path/to/old/javascript
// The supplied checkout must match every source hash in the fixture provenance.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const baseline = resolve(process.argv[2]);
const destination = new URL('../fixtures/warm-preflight-reference.json', import.meta.url);
const fixture = JSON.parse(await readFile(destination, 'utf8'));
assert.equal(fixture.source_commit, 'b847c272bbbad5d152bf254467098a689c828f68');
for (const row of fixture.source_files) {
  const bytes = await readFile(resolve(baseline, row.path));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), row.sha256, row.path);
}
const { createSession } = await import(pathToFileURL(resolve(baseline, 'src/session.js')).href);
const X = { data: Float64Array.of(0, 0, .1, -.1, .2, .1, 4, 4, 4.3, 4.1, 3.8, 4.2), nSamples: 6, nFeatures: 2 };
const options = { nClusters: 2, initCenters: Float64Array.of(0, 0, 4, 4), m: 64, maxMemoryBytes: 4000, maxIterations: 1, tolerance: 0 };
const tinyX = { data: Float64Array.of(0, 1e-100, 1), nSamples: 3, nFeatures: 1 };
const tinyOptions = { nClusters: 2, initCenters: Float64Array.of(0, 1), m: 2, maxIterations: 1, tolerance: 0, maxMemoryBytes: 2000 };
function finish(session) {
  for (let i = 0; !session.status.done; ++i) {
    assert.ok(i < 100); session.step(10, { timeBudgetMs: 1000, maxChunks: 100 });
  }
  return session.snapshot().result;
}
function encode(value) {
  if (value === undefined) return { $undefined: true };
  if (typeof value === 'number' && (!Number.isFinite(value) || Object.is(value, -0))) return { $number: Object.is(value, -0) ? '-0' : String(value) };
  if (ArrayBuffer.isView(value)) return { $typed: value.constructor.name, values: Array.from(value, encode) };
  if (Array.isArray(value)) return value.map(encode);
  if (value && typeof value === 'object') {
    assert.equal(Object.getOwnPropertySymbols(value).length, 0, 'Unexpected symbolic public result');
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, encode(v)]));
  }
  return value;
}
const results = {};
function record(key, value) {
  const encoded = encode(value);
  if (Object.hasOwn(results, key)) assert.deepEqual(encoded, results[key], `Reference methods must agree: ${key}`);
  else results[key] = encoded;
}
for (const phase of ['done', 'partial']) {
  const session = createSession('fcm', X, { ...options, maxIterations: phase === 'partial' ? 2 : 1 });
  try {
    if (phase === 'partial') session.step(1, { timeBudgetMs: 1000, maxChunks: 100 });
    record(`continued-${phase}`, finish(session));
  } finally { session.dispose(); }
  const tiny = createSession('fcm', tinyX, tinyOptions);
  try {
    if (phase === 'partial') tiny.step(1, { timeBudgetMs: 1000, maxChunks: 100 });
    record('tiny-continued', finish(tiny));
  } finally { tiny.dispose(); }
}
for (const method of ['updateParameters', 'configure']) {
  for (const [oldM, newM] of [[64, 2], [64, 128], [2, 64]]) {
    const session = createSession('fcm', X, { ...options, m: oldM });
    try {
      finish(session);
      if (method === 'updateParameters') session.updateParameters({ m: newM });
      else session.configure(X, { ...options, m: newM });
      record(`warm-${oldM}-${newM}`, finish(session));
    } finally { session.dispose(); }
  }
  const tiny = createSession('fcm', tinyX, tinyOptions);
  try {
    finish(tiny);
    if (method === 'updateParameters') tiny.updateParameters({ m: 4 });
    else tiny.configure(tinyX, { ...tinyOptions, m: 4 });
    record('tiny-warm-2-4', finish(tiny));
  } finally { tiny.dispose(); }
}
assert.equal(Object.keys(results).length, 7);
fixture.results = results;
await writeFile(destination, JSON.stringify(fixture, null, 2) + '\n');
console.log(JSON.stringify({ status: 'passed', referenceCases: 7, verifiedSourceFiles: fixture.source_files.length }));
