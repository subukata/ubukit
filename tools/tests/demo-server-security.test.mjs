// Run from the repository root: node --test tools/tests/demo-server-security.test.mjs
// Exercises the real server in a disposable fixture; never reads actual secrets.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import net from 'node:net';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { cp, mkdir, mkdtemp, readFile, readdir, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = fileURLToPath(new URL('../../', import.meta.url));
const sentinel = 'DEMO_TEST_PRIVATE_SENTINEL';

async function startServer(root) {
  const child = spawn(process.execPath, [path.join(root, 'examples/browser/serve.js')], {
    env: { ...process.env, PORT: '0' }, stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';
  let errors = '';
  child.stderr.on('data', data => { errors += data; });
  try {
    const port = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Server startup timed out: ${errors}`)), 10000);
      timer.unref();
      const finish = (error, value) => {
        clearTimeout(timer);
        child.removeListener('error', onError);
        child.removeListener('exit', onExit);
        child.stdout.removeListener('data', onData);
        error ? reject(error) : resolve(value);
      };
      const onError = error => finish(error);
      const onExit = code => finish(new Error(`Server exited (${code}): ${errors}`));
      const onData = data => {
        output += data;
        const match = /http:\/\/localhost:(\d+)\//.exec(output);
        if (match) finish(null, Number(match[1]));
      };
      child.on('error', onError);
      child.on('exit', onExit);
      child.stdout.on('data', onData);
    });
    return { child, port };
  } catch (error) {
    child.kill();
    throw error;
  }
}

function request(port, target, { method = 'GET', headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ hostname: '127.0.0.1', port, path: target, method,
      headers: { Host: `localhost:${port}`, ...headers }, agent: false }, res => {
      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('error', reject);
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
    });
    req.on('error', reject);
    req.setTimeout(5000, () => req.destroy(new Error('HTTP request timed out')));
    req.end();
  });
}

function rawRequest(port, headerLines, target = '/') {
  return new Promise((resolve, reject) => {
    const socket = net.connect(port, '127.0.0.1');
    let result = '';
    socket.setTimeout(5000, () => socket.destroy(new Error('Raw HTTP request timed out')));
    socket.on('error', reject);
    socket.on('data', data => { result += data; });
    socket.on('end', () => resolve({ status: Number(/^HTTP\/1\.[01] (\d+)/.exec(result)?.[1]), body: result }));
    socket.on('connect', () => socket.end(`GET ${target} HTTP/1.1\r\n${headerLines.join('\r\n')}\r\nConnection: close\r\n\r\n`));
  });
}

function assertDenied(response, label, statuses = [403]) {
  assert.ok(statuses.includes(response.status), `${label}: expected ${statuses}, got ${response.status}`);
  assert.ok(!response.body.toString().includes(sentinel), `${label}: private contents leaked`);
}

await test('loopback demo server security and asset compatibility', { timeout: 30000 }, async t => {
  const base = await mkdtemp(path.join(tmpdir(), 'ubukit-demo-server-'));
  const root = path.join(base, 'repo');
  let child;
  t.after(async () => {
    if (child && child.exitCode === null && child.signalCode === null) {
      const exited = once(child, 'exit');
      child.kill();
      await exited;
    }
    await rm(base, { recursive: true, force: true });
  });
  await mkdir(path.join(root, 'examples'), { recursive: true });
  await mkdir(path.join(root, 'javascript'), { recursive: true });
  await cp(path.join(repository, 'examples/browser'), path.join(root, 'examples/browser'), { recursive: true });
  await cp(path.join(repository, 'javascript/src'), path.join(root, 'javascript/src'), { recursive: true });
  await writeFile(path.join(root, 'package.json'), '{"type":"module"}');
  for (const relative of ['.env', '.git/HEAD', 'README.md', 'javascript/package.json',
    'javascript/src/private.js', 'javascript/src/.private.js', 'examples/browser/secrets.html']) {
    await mkdir(path.dirname(path.join(root, relative)), { recursive: true });
    await writeFile(path.join(root, relative), sentinel);
  }
  await writeFile(path.join(base, 'outside.js'), sentinel);
  await mkdir(path.join(base, 'repo-sibling'));
  await writeFile(path.join(base, 'repo-sibling/private.js'), sentinel);
  const running = await startServer(root);
  child = running.child;
  const { port } = running;

  await t.test('demo pages, every runtime module and workers retain exact bytes and MIME types', async () => {
    const routes = ['/examples/browser/index.html', '/examples/browser/test.html',
      '/examples/browser/app.js', '/examples/browser/browser-tests.js'];
    for (const name of await readdir(path.join(repository, 'javascript/src'))) {
      assert.ok(name.endsWith('.js'), `Review newly added runtime asset: ${name}`);
      routes.push(`/javascript/src/${name}`);
    }
    for (const route of routes) {
      const response = await request(port, route);
      assert.equal(response.status, 200, route);
      assert.deepEqual(response.body, await readFile(path.join(repository, route.slice(1))), route);
      assert.equal(response.headers['content-type'], route.endsWith('.html')
        ? 'text/html; charset=utf-8' : 'text/javascript; charset=utf-8');
      assert.equal(response.headers['content-length'], String(response.body.length));
      assert.equal(response.headers['cache-control'], 'no-store');
      assert.equal(response.headers['x-content-type-options'], 'nosniff');
      assert.equal(response.headers['access-control-allow-origin'], undefined);
    }
    const home = await request(port, '/?cache=123');
    assert.equal(home.status, 200);
    assert.deepEqual(home.body, await readFile(path.join(root, 'examples/browser/index.html')));
  });

  await t.test('the complete page, import and worker dependency closure remains public', async () => {
    const queue = ['/', '/examples/browser/test.html'];
    const visited = new Set();
    while (queue.length) {
      const route = queue.shift();
      if (visited.has(route)) continue;
      visited.add(route);
      const response = await request(port, route);
      assert.equal(response.status, 200, route);
      const source = response.body.toString();
      for (const pattern of [
        /\b(?:import|export)\s+[^;]*?\bfrom\s*['"]([^'"]+)['"]/g,
        /\bimport\s*\(\s*['"]([^'"]+)['"]/g,
        /\bnew\s+URL\s*\(\s*['"]([^'"]+)['"]\s*,\s*import\.meta\.url/g,
        /<script\b[^>]*\bsrc=['"]([^'"]+)['"]/g
      ]) {
        for (const match of source.matchAll(pattern)) {
          if (match[1] === 'node:worker_threads') continue;
          const dependency = new URL(match[1], `http://localhost:${port}${route}`);
          assert.equal(dependency.origin, `http://localhost:${port}`);
          queue.push(dependency.pathname);
        }
      }
    }
    for (const route of ['/examples/browser/app.js', '/examples/browser/browser-tests.js',
      '/javascript/src/index.js', '/javascript/src/worker.js', '/javascript/src/realtime-worker.js']) {
      assert.ok(visited.has(route), `Missing browser branch: ${route}`);
    }
    t.diagnostic(`Verified ${visited.size} page, module and worker routes`);
  });

  await t.test('HEAD has GET metadata with no body; mutation and other methods are refused', async () => {
    for (const route of ['/', '/javascript/src/worker.js']) {
      const get = await request(port, route);
      const head = await request(port, route, { method: 'HEAD' });
      assert.equal(head.status, 200);
      assert.equal(head.body.length, 0);
      assert.equal(head.headers['content-length'], get.headers['content-length']);
      assert.equal(head.headers['content-type'], get.headers['content-type']);
      assert.equal(head.headers['x-content-type-options'], 'nosniff');
    }
    for (const method of ['POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'TRACE']) {
      const response = await request(port, '/', { method });
      assert.equal(response.status, 405, method);
      assert.equal(response.headers.allow, 'GET, HEAD');
      assert.equal(response.headers['x-content-type-options'], 'nosniff');
    }
    const head = await request(port, '/.env', { method: 'HEAD' });
    assertDenied(head, 'denied HEAD');
    assert.equal(head.body.length, 0);
  });

  await t.test('unrelated, wrong-port and malformed Host authorities are refused', async () => {
    for (const host of [`localhost:${port}`, `LOCALHOST:${port}`, `127.0.0.1:${port}`]) {
      assert.equal((await request(port, '/', { headers: { Host: host } })).status, 200, host);
    }
    for (const host of ['attacker.example', `attacker.example:${port}`, 'localhost', '127.0.0.1',
      `localhost:${port === 65535 ? 65534 : port + 1}`, `localhost:${port}/`, `localhost:${port}@attacker.example`,
      `localhost.attacker.example:${port}`, `localhost.:${port}`, `127.1:${port}`, `2130706433:${port}`,
      `0x7f000001:${port}`, `[::1]:${port}`, `localhost:0${port}`, `localhost:+${port}`,
      `localhost :${port}`, `localhost:${port},attacker.example`, `http://localhost:${port}`]) {
      assertDenied(await request(port, '/', { headers: { Host: host } }), host, [400, 403]);
    }
    for (const lines of [[], ['Host:'], [`Host: localhost:${port}`, 'Host: attacker.example'],
      ['Host: attacker.example', `Host: localhost:${port}`], [`Host: localhost:${port}`, `Host: localhost:${port}`]]) {
      assertDenied(await rawRequest(port, lines), JSON.stringify(lines), [400, 403]);
    }
  });

  await t.test('Origin when present must be the same local HTTP origin', async () => {
    for (const host of ['localhost', '127.0.0.1']) {
      const headers = { Host: `${host}:${port}`, Origin: `http://${host}:${port}` };
      assert.equal((await request(port, '/', { headers })).status, 200);
    }
    for (const origin of ['null', 'http://attacker.example', `http://attacker.example:${port}`,
      `http://127.0.0.1:${port}`, `https://localhost:${port}`, 'http://localhost',
      `http://localhost:${port}/`, `http://localhost:${port}?x=1`, `http://localhost:${port}#x`,
      `http://localhost:${port}@attacker.example`, `http://localhost:${port} http://attacker.example`, '']) {
      assertDenied(await request(port, '/', { headers: { Origin: origin } }), origin);
    }
    assertDenied(await rawRequest(port, [`Host: localhost:${port}`,
      `Origin: http://localhost:${port}`, `Origin: http://localhost:${port}`]), 'duplicate Origin');
  });

  await t.test('private files, traversal, encoded separators and normalization aliases are refused', async () => {
    for (const target of ['/.env', '/.git/HEAD', '/README.md', '/package.json', '/examples/browser/serve.js',
      '/javascript/package.json', '/javascript/src/private.js', '/javascript/src/.private.js',
      '/examples/browser/secrets.html', '/javascript/src/', '/examples/browser/',
      '/../outside.js', '/%2e%2e%2foutside.js', '/%2e%2e%2frepo-sibling/private.js',
      '/javascript/src/../../.env', '/javascript/src/%2e%2e/%2e%2e/.env',
      '/javascript%2fsrc/index.js', '/javascript%5csrc/index.js', '/javascript%252fsrc/index.js',
      '/examples/browser/../browser/index.html', '/examples/browser/%2e/index.html',
      '/examples//browser/index.html', '/examples/browser/index.html/', '/examples/browser/index.html%00',
      '/%2Eenv', '/%', '/%GG', '/%C0%AF.env']) {
      const response = await request(port, target);
      assertDenied(response, target);
      assert.equal(response.headers['x-content-type-options'], 'nosniff');
    }
    for (const target of [`http://localhost:${port}/`, '//localhost/', '/javascript\\src/index.js', '/#fragment']) {
      assertDenied(await request(port, target), target, [400]);
    }
  });

  await t.test('allowlisted file and directory symlinks cannot disclose other files', async t => {
    const page = path.join(root, 'examples/browser/index.html');
    const savedPage = `${page}.original`;
    await rename(page, savedPage);
    try {
      try {
        await symlink(path.join(base, 'outside.js'), page, 'file');
      } catch (error) {
        if (process.platform === 'win32' && error.code === 'EPERM') {
          t.skip('Windows does not permit creating symlinks with this account');
          return;
        }
        throw error;
      }
      assertDenied(await request(port, '/'), 'out-of-root file symlink');
      await rm(page);
      await symlink(path.join(root, '.env'), page, 'file');
      assertDenied(await request(port, '/'), 'in-repository private symlink');
      await rm(page);
      await symlink(savedPage, page, 'file');
      assertDenied(await request(port, '/'), 'even an in-root public file symlink');
    } finally {
      await rm(page, { force: true });
      await rename(savedPage, page);
    }
    const source = path.join(root, 'javascript/src');
    const savedSource = `${source}.original`;
    await rename(source, savedSource);
    await mkdir(path.join(base, 'external-modules'));
    await writeFile(path.join(base, 'external-modules/index.js'), sentinel);
    try {
      await symlink(path.join(base, 'external-modules'), source, process.platform === 'win32' ? 'junction' : 'dir');
      assertDenied(await request(port, '/javascript/src/index.js'), 'out-of-root directory symlink');
    } finally {
      await rm(source, { force: true });
      await rename(savedSource, source);
    }
    assert.equal((await request(port, '/')).status, 200, 'server remains usable after rejection');
    assert.equal((await request(port, '/javascript/src/index.js')).status, 200);
  });

  await t.test('missing approved assets fail without exposing paths or crashing', async () => {
    const page = path.join(root, 'examples/browser/test.html');
    await rename(page, `${page}.original`);
    const response = await request(port, '/examples/browser/test.html');
    assert.equal(response.status, 404);
    assert.equal(response.body.toString(), 'Not found');
    assert.equal(response.headers['x-content-type-options'], 'nosniff');
    assert.equal((await request(port, '/')).status, 200);
  });
});
