// Exercise the actual repository demo server, including its Host/Origin policy.
// Usage: node tools/ci/browser/demo-smoke.mjs /path/to/repository /path/to/browser-results
// Uses the same locked browser tools/engines as run.mjs; installs nothing itself.
import { chromium, firefox, webkit } from '@playwright/test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

assert.ok(process.argv[2] && process.argv[3], 'Supply the repository root and browser-results directory');
const root = resolve(process.argv[2]);
const output = resolve(process.argv[3]);
const algorithms = ['kmeans', 'fcm', 'rcm', 'exrcm', 'som-olp', 'neighborhood'];
const expectedCases = [...algorithms, 'cancel-heartbeat-reuse', 'transfer-detachment'];
const reports = [];
await mkdir(output, { recursive: true });

let server;
let stderr = '';
async function startServer() {
  server = spawn(process.execPath, [resolve(root, 'examples/browser/serve.js')], {
    cwd: root, env: { ...process.env, PORT: '0' }, stdio: ['ignore', 'pipe', 'pipe']
  });
  server.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4096); });
  return new Promise((done, reject) => {
    let stdout = '';
    const timer = setTimeout(() => finish(new Error(`Demo server startup timed out: ${stderr}`)), 10000);
    const finish = (error, port) => {
      clearTimeout(timer);
      server.removeListener('error', onError);
      server.removeListener('exit', onExit);
      server.stdout.removeListener('data', onData);
      error ? reject(error) : done(port);
    };
    const onError = error => finish(error);
    const onExit = code => finish(new Error(`Demo server exited (${code}): ${stderr}`));
    const onData = chunk => {
      stdout = (stdout + chunk).slice(-4096);
      const match = /http:\/\/localhost:(\d+)\//.exec(stdout);
      if (match) finish(null, Number(match[1]));
    };
    server.on('error', onError);
    server.on('exit', onExit);
    server.stdout.on('data', onData);
  });
}

async function browserTests(page) {
  await page.locator('#start').click();
  await page.waitForFunction(() => /^(PASS|FAIL)$/.test(document.getElementById('status').textContent),
    undefined, { timeout: 30000 });
  const report = JSON.parse(await page.locator('#output').innerText());
  assert.equal(report.status, 'PASS', JSON.stringify(report));
  assert.deepEqual(report.cases.map(item => item.name), expectedCases);
  assert.ok(report.cases.every(item => item.status === 'pass'));
  return report;
}

try {
  const port = await startServer();
  for (const [engineName, engine] of Object.entries({ chromium, firefox, webkit })) {
    let browser;
    try {
      browser = await engine.launch({ headless: true });
      for (const hostname of ['localhost', '127.0.0.1']) {
        const name = `demo-${engineName}-${hostname.replaceAll('.', '-')}`;
        const errors = [];
        let page;
        const report = { engine: engineName, hostname, version: browser.version(),
          target: 'repository-demo-server', status: 'failed', actualBrowserExecution: false };
        try {
          page = await browser.newPage();
          page.on('pageerror', error => errors.push(String(error)));
          page.on('requestfailed', request => errors.push(`requestfailed ${request.url()} ${request.failure()?.errorText}`));
          page.on('response', response => {
            // The demo has no favicon. All requested application assets must load.
            if (response.status() >= 400 && new URL(response.url()).pathname !== '/favicon.ico') {
              errors.push(`HTTP ${response.status()} ${response.url()}`);
            }
          });
          const base = `http://${hostname}:${port}`;
          const response = await page.goto(`${base}/examples/browser/test.html`);
          assert.equal(response.status(), 200);
          report.actualBrowserExecution = true;
          assert.equal(response.headers()['x-content-type-options'], 'nosniff');
          report.browserTests = await browserTests(page);
          // Repeat after Worker cancellation/disposal to catch lifecycle regressions.
          report.repeatedBrowserTests = await browserTests(page);
          await page.screenshot({ path: resolve(output, `${name}-worker-tests.png`), fullPage: true });

          assert.equal((await page.goto(`${base}/`)).status(), 200);
          await page.locator('#count').fill('80');
          await page.locator('#new').click();
          report.demoAlgorithms = [];
          for (const algorithm of algorithms) {
            await page.locator('#algorithm').selectOption(algorithm);
            await page.locator('#run').click();
            await page.waitForFunction(() => !document.getElementById('run').disabled,
              undefined, { timeout: 30000 });
            const status = await page.locator('#status').innerText();
            assert.match(status, /^Done/, `${algorithm}: ${status}`);
            const details = JSON.parse(await page.locator('#details').innerText());
            assert.equal(details.algorithm, algorithm);
            report.demoAlgorithms.push({ algorithm, status });
          }
          await page.screenshot({ path: resolve(output, `${name}-demo.png`), fullPage: true });
          assert.deepEqual(errors, []);
          report.status = 'passed';
        } catch (error) {
          report.error = String(error);
          if (page) {
            try { await page.screenshot({ path: resolve(output, `${name}-failure.png`), fullPage: true }); }
            catch (screenshotError) { errors.push(`screenshot: ${screenshotError}`); }
          }
        } finally {
          if (page) {
            try { await page.close(); }
            catch (error) { errors.push(`page close: ${error}`); }
          }
        }
        report.errors = errors;
        if (errors.length) report.status = 'failed';
        reports.push(report);
        await writeFile(resolve(output, `${name}.json`), JSON.stringify(report, null, 2) + '\n');
      }
    } catch (error) {
      reports.push({ engine: engineName, status: 'failed', phase: 'launch', error: String(error) });
    } finally {
      if (browser) {
        try { await browser.close(); }
        catch (error) { reports.push({ engine: engineName, status: 'failed', phase: 'close', error: String(error) }); }
      }
    }
  }
} catch (error) {
  reports.push({ status: 'failed', phase: 'server-or-runner', error: String(error), serverStderr: stderr });
} finally {
  if (server?.pid && server.exitCode === null && server.signalCode === null) {
    const exited = once(server, 'exit');
    server.kill();
    const timer = setTimeout(() => server.kill('SIGKILL'), 5000);
    try { await exited; } finally { clearTimeout(timer); }
  }
  await writeFile(resolve(output, 'demo-summary.json'), JSON.stringify(reports, null, 2) + '\n');
}
if (reports.length !== 6 || reports.some(report => report.status !== 'passed')) process.exitCode = 1;
console.log(JSON.stringify(reports, null, 2));
