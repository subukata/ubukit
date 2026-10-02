/** Local-only inputs. No network requests, package installation or runtime edits. */
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const runtime = path.resolve(root, '../package');
export const generated = path.join(root, '.generated');
export const readJSON = file => JSON.parse(fs.readFileSync(file, 'utf8'));
export const inventory = readJSON(path.join(root, 'INVENTORY.json'));
export const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

export function wrapperBytes(wrapper) {
  const text = fs.readFileSync(path.join(runtime, 'src', wrapper), 'utf8');
  const matches = [...text.matchAll(/const bytes\s*=\s*new Uint8Array\((\[[\s\S]*?\])\);/g)];
  assert.equal(matches.length, 1, `Expected exactly one embedded byte array in ${wrapper}`);
  const array = JSON.parse(matches[0][1]);
  assert.ok(Array.isArray(array) && array.every(n => Number.isInteger(n) && n >= 0 && n <= 255), `Invalid byte array in ${wrapper}`);
  return Buffer.from(array);
}

export function capturedBytes(binary) {
  let bytes;
  if (binary.wrapper) {
    bytes = wrapperBytes(binary.wrapper);
  } else {
    const fixture = readJSON(path.join(root, binary.fixture)).binaries.find(b => b.sha256 === binary.sha256);
    assert.ok(fixture, `Missing historical fixture ${binary.sha256}`);
    bytes = Buffer.from(fixture.base64, 'base64');
  }
  assert.equal(bytes.length, binary.bytes, `Captured binary length changed: ${binary.sha256}`);
  assert.equal(hash(bytes), binary.sha256, `Captured binary digest changed: ${binary.sha256}`);
  return bytes;
}

export function verifyPreservedSources() {
  const files = readJSON(path.join(root, 'SOURCE_FILES.json')).files;
  for (const file of files) {
    const bytes = fs.readFileSync(path.join(root, file.path));
    assert.equal(bytes.length, file.bytes, `Preserved source length changed: ${file.path}`);
    assert.equal(hash(bytes), file.sha256, `Preserved source digest changed: ${file.path}`);
  }
  return files.length;
}

export function runtimeSnapshot() {
  const files = [];
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) files.push({ path: path.relative(runtime, file), sha256: hash(fs.readFileSync(file)) });
      else throw new Error(`Unsupported runtime file type: ${file}`);
    }
  }
  visit(runtime);
  return { package_version: readJSON(path.join(runtime, 'package.json')).version, files, sha256: hash(JSON.stringify(files)) };
}

export async function loadWabt() {
  const require = createRequire(import.meta.url);
  let entry;
  try {
    // Use an existing installation offline. WABT_MODULE_PATH is the absolute package directory.
    entry = process.env.WABT_MODULE_PATH
      ? require.resolve(path.join(path.resolve(process.env.WABT_MODULE_PATH), 'index.js'))
      : require.resolve('wabt');
  } catch {
    throw new Error('WABT is unavailable. Run npm ci --ignore-scripts --no-audit --no-fund, or set WABT_MODULE_PATH to an existing wabt 1.0.39 package directory.');
  }
  const metadata = readJSON(path.join(path.dirname(entry), 'package.json'));
  assert.equal(metadata.name, 'wabt');
  assert.equal(metadata.version, '1.0.39', 'Byte-exact checks require pinned WABT 1.0.39');
  return await require(entry)();
}
