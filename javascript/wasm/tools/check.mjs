/** Two bounded stages, with a read-only sibling runtime and disposable local outputs. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { root, generated, runtimeSnapshot, verifyPreservedSources } from './common.mjs';

const report = {
  schema_version: 1, observed_utc: new Date().toISOString(), node: process.version,
  platform: process.platform, arch: process.arch, stage_timeout_ms: 60_000,
  passed: false, stages: []
};
let before;
try {
  before = runtimeSnapshot();
  report.runtime_package_version = before.package_version;
  report.runtime_snapshot_sha256 = before.sha256;
  report.preserved_source_files_verified = verifyPreservedSources();
  fs.mkdirSync(path.join(root, 'verification'), { recursive: true });
  for (const script of ['tools/verify-rebuild.mjs', 'tools/test-kernels.mjs']) {
    const result = spawnSync(process.execPath, [script], {
      cwd: root, timeout: 60_000, stdio: 'inherit', killSignal: 'SIGTERM'
    });
    report.stages.push({ script, exit_status: result.status, signal: result.signal, passed: !result.error && result.status === 0 });
    if (result.error) throw result.error;
    assert.equal(result.status, 0, `${script} failed (${result.status ?? result.signal})`);
  }
  const after = runtimeSnapshot();
  assert.deepEqual(after, before, 'Sibling runtime was modified during verification');
  report.runtime_unchanged = true;
  report.passed = true;
} catch (error) {
  report.error = error.message;
  console.error(error.stack);
  process.exitCode = 1;
} finally {
  if (before) report.runtime_unchanged = runtimeSnapshot().sha256 === before.sha256;
  fs.rmSync(generated, { recursive: true, force: true });
  fs.mkdirSync(path.join(root, 'verification'), { recursive: true });
  fs.writeFileSync(path.join(root, 'verification/check.json'), JSON.stringify(report, null, 2) + '\n');
}
if (report.passed) console.log('Byte-exact rebuild and bounded correctness checks passed. Sibling runtime unchanged; temporary outputs removed.');
