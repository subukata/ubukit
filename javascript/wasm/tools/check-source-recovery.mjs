/** Read-only local WAT derivation/byte-exact checks. No kernel is instantiated. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opts = {};
for (let i = 0; i < args.length; i += 2) {
  assert.ok(['--root', '--derived-source', '--wabt-module'].includes(args[i]), `Unknown option: ${args[i]}`);
  assert.ok(args[i + 1] && !args[i + 1].startsWith('--'), `Missing value for ${args[i]}`);
  assert.ok(!(args[i] in opts), `Repeated option: ${args[i]}`);
  opts[args[i]] = args[i + 1];
}
const root = path.resolve(opts['--root'] ?? path.join(here, '..'));
const derivedFile = path.resolve(opts['--derived-source'] ?? path.join(root, 'derived-wat/rmcm/pre_f32_radius_candidates.wat'));
const sha256 = b => crypto.createHash('sha256').update(b).digest('hex');
const expected = {
  metric: 'b7b68848eecdca210e2bf13288c5313ee38d47bb315046c40ff143654cb61656',
  oldRmcm: '63768dc7e7388372fbbc1a6517e4e99b570699cbfb8f9d18b9ccbd897ca9ebd8',
  currentRmcm: '0e3f36d0aab2ec913f3cc78a2fb1fd840fe84b58f0f3fd2d3e7c9264f7c030f3',
};
const inputs = [];
const tracked = new Map();
function read(file, label, hash) {
  const b = fs.readFileSync(file);
  if (hash) assert.equal(sha256(b), hash, `Input hash mismatch: ${label}`);
  if (!tracked.has(file)) {
    tracked.set(file, sha256(b));
    inputs.push({path: label, bytes: b.length, sha256: sha256(b)});
  }
  return b;
}
function input(relative, hash) { return read(path.join(root, relative), relative, hash); }
read(fileURLToPath(import.meta.url), 'tools/check-source-recovery.mjs');
function wrapper(name, expectedHash) {
  // Supports both the original sealed supplement and repository javascript/wasm layout.
  const relative = fs.existsSync(path.join(root, 'runtime/src', name)) ? `runtime/src/${name}` : `../src/${name}`;
  const b = input(relative);
  const m = b.toString().match(/const bytes\s*=\s*new Uint8Array\((\[[\s\S]*?\])\);/);
  assert.ok(m, `Missing numeric byte array in ${relative}`);
  const values = JSON.parse(m[1]);
  assert.ok(values.every(v => Number.isInteger(v) && v >= 0 && v <= 255));
  const bytes = Buffer.from(values);
  assert.equal(sha256(bytes), expectedHash, `Embedded kernel changed: ${relative}`);
  return {path: relative, bytes};
}
function capture(hash) {
  const embedded = `embedded/${hash}.wasm`;
  if (fs.existsSync(path.join(root, embedded))) return {path: embedded, bytes: input(embedded, hash)};
  const fixture = JSON.parse(input('evidence/historical-binaries.json'));
  const entry = fixture.binaries.find(b => b.sha256 === hash);
  assert.ok(entry, 'Missing captured historical binary');
  const bytes = Buffer.from(entry.base64, 'base64');
  assert.equal(sha256(bytes), hash);
  assert.equal(bytes.length, entry.bytes);
  return {path: 'evidence/historical-binaries.json', bytes};
}
const original = input('original-wat/rmcm/radius_candidates.wat', '020cdde353023a205fe89beb09ddf5fd4903e9bf0d98be9b1f4095eb2c7de60f').toString();
const fragment = input('original-wat/rmcm/f32_candidates.wat', '9c6c23b88dc21e59dc8b84b3c05f8e882e91f06d60be11291c976be479100160').toString();
input('evidence/build-scripts/rmcm/build.mjs', '684c4ea21fe8374cdac138e767b7575f2c36de5a126df139614d83420ffc250e');
input('evidence/build-scripts/rmcm/generate_f32.py', 'ddb649dee3099c477fedde98ef2f8db9de4f428e0e196e152ff874ad6e2cb266');
const metricPath = `reconstructed-wat/${expected.metric}.reconstructed.wat`;
const metricWat = input(metricPath, '8653a1cd960d2494ae47e8ff867e64aa95ad76c7b677493051a4f9613d77d5b3').toString();
const metric = wrapper('metric-wasm.js', expected.metric);
const currentRmcm = wrapper('rmcm-wasm.js', expected.currentRmcm);
const oldRmcm = capture(expected.oldRmcm);
assert.equal(oldRmcm.bytes.length, 3097);
assert.equal(metric.bytes.length, 1246);

const marker = ' (func (export "scan_f32_candidates")';
const markerAt = original.indexOf(marker);
assert.ok(markerAt > 0 && original.lastIndexOf(marker) === markerAt);
assert.equal(original.slice(markerAt), fragment + ')\n', 'Appended source must equal preserved Float32 fragment');
const prefixOnly = original.slice(0, markerAt) + ')\n';
const parameters = ' (param $counter i32) (param $max_pairs i32)';
const guard = '            ;; Preserve the scalar edge-guard order before later numeric errors.\n'
  + '            (i32.store (local.get $counter) (i32.add (i32.load (local.get $counter)) (i32.const 1)))\n'
  + '            (if (i32.gt_u (i32.load (local.get $counter)) (local.get $max_pairs)) (then (return (i32.const -3))))\n';
function removeOnce(text, piece) {
  assert.equal(text.split(piece).length, 2, 'Derivation must remove exactly one matching change');
  return text.replace(piece, '');
}
const derivedBody = removeOnce(removeOnce(prefixOnly, parameters), guard);
assert.ok(!derivedBody.includes('$counter') && !derivedBody.includes('$max_pairs') && !derivedBody.includes('scan_f32_candidates'));
const header = ';; DERIVED HISTORICAL RECONSTRUCTION, not a recovered original file.\n'
  + ';; Starting source: original-wat/rmcm/radius_candidates.wat\n'
  + ';; Starting source SHA-256: 020cdde353023a205fe89beb09ddf5fd4903e9bf0d98be9b1f4095eb2c7de60f\n'
  + ';; Removed the appended Float32 function and reversed filter counter/max_pairs edits.\n'
  + ';; Expected binary SHA-256: 63768dc7e7388372fbbc1a6517e4e99b570699cbfb8f9d18b9ccbd897ca9ebd8\n'
  + ';; Build: WABT 1.0.39; simd:true; write_debug_names:true.\n'
  + ';; Byte equality establishes technical lineage, not a legal authorship conclusion.\n';
const derived = read(derivedFile, 'derived-wat/rmcm/pre_f32_radius_candidates.wat').toString();
assert.equal(derived, header + derivedBody, 'Derived WAT must equal the documented deterministic transformation');

// Resolve existing trusted tooling only. No package-manager/install/network action.
const require = createRequire(path.join(root, 'package.json'));
const explicitWabt = opts['--wabt-module'] ?? process.env.WABT_MODULE_PATH;
const wabtModule = explicitWabt ? require.resolve(path.resolve(explicitWabt)) : require.resolve('wabt');
read(wabtModule, 'toolchain/wabt/index.js', 'cecf06ab5c65c3b7c05bafe601bbba61a45e4939889c5c51351e8817ac06f48d');
const wabtPackage = JSON.parse(read(path.join(path.dirname(wabtModule), 'package.json'), 'toolchain/wabt/package.json'));
assert.equal(wabtPackage.name, 'wabt');
assert.equal(wabtPackage.version, '1.0.39');
const wabt = await require(wabtModule)();
function compile(name, text, debug) {
  const m = wabt.parseWat(name, text, {simd: true});
  try {
    m.resolveNames();
    m.validate({simd: true});
    return Buffer.from(m.toBinary({log: false, write_debug_names: debug}).buffer);
  } finally { m.destroy(); }
}
function uleb(b, state) {
  let n = 0, shift = 0, v;
  do { assert.ok(state.p < b.length && shift <= 28); v = b[state.p++]; n += (v & 127) * 2 ** shift; shift += 7; } while (v & 128);
  return n;
}
function sections(b) {
  assert.equal(b.subarray(0, 8).toString('hex'), '0061736d01000000');
  const s = {p: 8}, out = [];
  while (s.p < b.length) {
    const start = s.p, id = b[s.p++], size = uleb(b, s), payload = s.p, end = payload + size;
    assert.ok(end <= b.length);
    let name = null;
    if (id === 0) { const len = uleb(b, s); assert.ok(s.p + len <= end); name = b.subarray(s.p, s.p + len).toString(); }
    out.push({id, name, offset: start, payload_bytes: size, encoded_bytes: end - start, sha256: sha256(b.subarray(start, end)), raw: b.subarray(start, end), payload: b.subarray(payload, end)});
    s.p = end;
  }
  return out;
}
function standard(b) { return Buffer.concat([b.subarray(0, 8), ...sections(b).filter(s => s.id !== 0).map(s => s.raw)]); }
function bodies(b) {
  const code = sections(b).find(s => s.id === 10).payload, pos = {p: 0};
  const n = uleb(code, pos), out = [];
  for (let i = 0; i < n; ++i) { const size = uleb(code, pos), end = pos.p + size; assert.ok(end <= code.length); out.push(code.subarray(pos.p, end)); pos.p = end; }
  assert.equal(pos.p, code.length); return out;
}
function trial(name, text, reference, debug) {
  const bytes = compile(name, text, debug);
  return {bytes, record: {write_debug_names: debug, bytes: bytes.length, sha256: sha256(bytes), byte_identical: bytes.equals(reference), standard_sections_identical: standard(bytes).equals(standard(reference)), custom_sections: sections(bytes).filter(s => s.id === 0).map(({raw, payload, ...s}) => s)}};
}
const currentControl = trial('radius_candidates.wat', original, currentRmcm.bytes, true);
assert.ok(currentControl.record.byte_identical, 'Recovered current WAT control mismatch');
const prefixControl = trial('suffix-removed-only.wat', prefixOnly, oldRmcm.bytes, true);
assert.ok(!prefixControl.record.byte_identical, 'Suffix-only source unexpectedly matches old bytes');
const oldTrials = [false, true].map(debug => trial('pre_f32_radius_candidates.wat', derived, oldRmcm.bytes, debug));
assert.ok(oldTrials[1].record.byte_identical, 'Derived historical source must match every old binary byte');
assert.ok(oldTrials[0].record.standard_sections_identical && !oldTrials[0].record.byte_identical);
const metricTrials = [false, true].map(debug => trial('metric.reconstructed.wat', metricWat, metric.bytes, debug));
assert.ok(metricTrials[0].record.byte_identical, 'Metric WAT must match every embedded metric byte');
const oldBodies = bodies(oldRmcm.bytes), currentBodies = bodies(currentRmcm.bytes), derivedBodies = bodies(oldTrials[1].bytes);
const bodyComparison = oldBodies.map((body, index) => ({index, name: ['scan', 'filter', 'check_tiny', 'scan_exact'][index], old_bytes: body.length, current_bytes: currentBodies[index].length, old_sha256: sha256(body), current_sha256: sha256(currentBodies[index]), old_current_identical: body.equals(currentBodies[index]), derived_old_identical: body.equals(derivedBodies[index])}));
assert.equal(oldBodies.length, 4); assert.equal(currentBodies.length, 5);
assert.deepEqual(bodyComparison.map(b => b.old_current_identical), [true, false, true, true]);
for (const [file, hash] of tracked) assert.equal(sha256(fs.readFileSync(file)), hash, 'An input changed during verification');
const report = {
  schema_version: 1, observed_utc: new Date().toISOString(),
  scope: 'Technical source/byte correspondence; derived historical source is not a recovered original file',
  toolchain: {wabt_version: wabtPackage.version, node: process.version, platform: process.platform, arch: process.arch, parse_options: {simd: true}, binary_options: {log: false}, wabt_module_sha256: sha256(fs.readFileSync(wabtModule))},
  safety: {no_network_or_install: true, no_kernel_instantiation_or_execution: true, no_source_snapshot_modification: true, input_hashes_unchanged: true},
  inputs,
  metric: {status: 'reconstructed_from_captured_binary', source: metricPath, embedded_reference: metric.path, original_source_capture_gap_remains: true, selected_write_debug_names: false, trials: metricTrials.map(t => t.record)},
  historical_rmcm: {status: 'derived_historical_reconstruction_from_recovered_current_source', source: 'derived-wat/rmcm/pre_f32_radius_candidates.wat', source_sha256: sha256(Buffer.from(derived)), reference: oldRmcm.path, original_standalone_file_recovered: false, derivation: {preserved_float32_suffix_exact_match: true, removed_appended_function: 'scan_f32_candidates', removed_parameters: ['counter', 'max_pairs'], removed_guard: 'accepted-pair counter increment and max_pairs return -3', transformation_reproduced_exact_source_text: true}, selected_write_debug_names: true, trials: oldTrials.map(t => t.record), current_original_control: currentControl.record, float32_suffix_only_removal_control: prefixControl.record, function_body_comparison: bodyComparison},
  interpretation: ['The metric rebuilt bytes equal the captured current embedded bytes.', 'The derived historical RMCM rebuilt bytes equal all 3,097 archived bytes, including the name custom section.', 'Compilation and byte identity do not establish numerical qualification, original-file recovery, copyright ownership, or legal clearance.'],
};
const json = JSON.stringify(report, null, 2) + '\n';
process.stdout.write(json);
