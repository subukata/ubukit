import http from 'node:http';
import { constants } from 'node:fs';
import { lstat, open, realpath } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = await realpath(fileURLToPath(new URL('../../', import.meta.url)));
const port = Number(process.env.PORT ?? 8765);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new RangeError('Invalid PORT');

// Only browser entry points and reviewed runtime modules are public. Do not turn
// this into a repository-root server or automatically include newly added files.
const assets = new Map();
for (const name of ['index.html', 'test.html', 'app.js', 'browser-tests.js']) {
  assets.set(`/examples/browser/${name}`, `examples/browser/${name}`);
}
for (const name of [
  'clustering', 'core', 'entropy-fcm', 'external-metrics', 'fcm-stable', 'index',
  'iteration-kernels', 'kmeans-wasm', 'metric-scheduler', 'metric-wasm',
  'neighborhood', 'optimization', 'realtime-worker-client', 'realtime-worker',
  'rmcm-grid', 'rmcm-wasm', 'rmcm', 'session-hooks', 'session-validation',
  'session', 'som-elite', 'som-numerics', 'som-olp', 'som-pca-wasm',
  'som-training-wasm', 'som', 'worker-client', 'worker'
]) {
  assets.set(`/javascript/src/${name}.js`, `javascript/src/${name}.js`);
}
assets.set('/', 'examples/browser/index.html');

function headerCount(req, name) {
  let count = 0;
  for (let i = 0; i < req.rawHeaders.length; i += 2) {
    if (req.rawHeaders[i].toLowerCase() === name) count++;
  }
  return count;
}

function localAuthority(value, listeningPort) {
  if (typeof value !== 'string') return null;
  const match = /^(localhost|127\.0\.0\.1)(?::([1-9][0-9]{0,4}))?$/i.exec(value);
  if (!match || Number(match[2] ?? 80) !== listeningPort) return null;
  return match[1].toLowerCase();
}

function localRequest(req, listeningPort) {
  if (headerCount(req, 'host') !== 1) return false;
  const host = localAuthority(req.headers.host, listeningPort);
  if (!host) return false;
  const origins = headerCount(req, 'origin');
  if (origins === 0) return true; // Ordinary navigation need not send Origin.
  if (origins !== 1 || typeof req.headers.origin !== 'string') return false;
  const origin = /^http:\/\/([^/]+)$/i.exec(req.headers.origin);
  return origin !== null && localAuthority(origin[1], listeningPort) === host;
}

class ForbiddenAsset extends Error {}

async function readAsset(relative) {
  // Refuse symlinks in every component, including directory junctions. Checking
  // only a lexical prefix would let a link escape the approved asset directories.
  let filename = root;
  const components = relative.split('/');
  for (let i = 0; i < components.length; i++) {
    filename = path.join(filename, components[i]);
    const info = await lstat(filename);
    if (info.isSymbolicLink() || (i < components.length - 1 ? !info.isDirectory() : !info.isFile())) {
      throw new ForbiddenAsset();
    }
  }
  if (await realpath(filename) !== filename) throw new ForbiddenAsset();
  // NOFOLLOW also prevents a final-component symlink swap between checking and
  // opening on platforms that provide it. This is a local development server,
  // not a sandbox against an attacker with write access to the running checkout.
  const file = await open(filename, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    if (!(await file.stat()).isFile()) throw new ForbiddenAsset();
    return await file.readFile();
  } finally {
    await file.close();
  }
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  function reply(status, body) {
    res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(req.method === 'HEAD' ? undefined : body);
  }
  if (!localRequest(req, server.address().port)) return reply(403, 'Forbidden');
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('Allow', 'GET, HEAD');
    return reply(405, 'Method not allowed');
  }
  // Match the raw origin-form path without URL normalization, decoding, or path
  // resolution. Encoded separators, dot segments and absolute targets cannot
  // alias a public asset. Query strings may be used for cache-busting.
  if (!req.url.startsWith('/') || req.url.startsWith('//') || /[\\#\x00-\x20\x7f]/.test(req.url)) {
    return reply(400, 'Bad request');
  }
  const relative = assets.get(req.url.split('?')[0]);
  if (!relative) return reply(403, 'Forbidden');
  try {
    const data = await readAsset(relative);
    res.writeHead(200, {
      'Content-Type': relative.endsWith('.html') ? 'text/html; charset=utf-8' : 'text/javascript; charset=utf-8',
      'Content-Length': data.length
    });
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch (error) {
    if (error instanceof ForbiddenAsset || error.code === 'ELOOP') return reply(403, 'Forbidden');
    return reply(404, 'Not found');
  }
});
server.listen(port, '127.0.0.1', () => {
  console.log(`UbuKit demo http://localhost:${server.address().port}/ ; browser tests /examples/browser/test.html`);
});
