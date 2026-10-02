/** Verify deterministic disassembly without rewriting checked-in sources or the runtime. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { root, inventory, capturedBytes, loadWabt, verifyPreservedSources } from './common.mjs';
verifyPreservedSources();
const wabt = await loadWabt();
for (const binary of inventory.binaries) {
  const module = wabt.readWasm(new Uint8Array(capturedBytes(binary)), { readDebugNames: true, simd: true });
  try {
    module.applyNames();
    const text = module.toText({ foldExprs: false, inlineExport: false });
    const header = `;; RECONSTRUCTED FROM BINARY, not recovered original source.\n;; Input SHA-256: ${binary.sha256}\n;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.\n;; Original comments, source formatting, authorship and license are NOT recovered.\n`;
    const expected = fs.readFileSync(path.join(root, 'reconstructed-wat', binary.sha256 + '.reconstructed.wat'), 'utf8');
    assert.equal(header + text + '\n', expected, `Disassembly changed: ${binary.sha256}`);
  } finally {
    module.destroy();
  }
}
console.log(`All ${inventory.binaries.length} reconstructed WAT files match deterministic disassembly. No files changed.`);
