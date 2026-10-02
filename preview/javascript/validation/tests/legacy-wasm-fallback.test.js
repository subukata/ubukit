import test from 'node:test';
import assert from 'node:assert/strict';

for (const helper of ['createRadiusScanner', 'createCsrFilter']) {
  test(`${helper} falls back on a CSP SecurityError`, async () => {
    const api = await import(`../../consumer/node_modules/ubukit-js/src/rmcm-wasm.js?csp-${helper}`);
    const original = globalThis.WebAssembly;
    const blocked = Object.create(original);
    blocked.validate = () => true;
    blocked.Module = function () { const error = new Error('CSP blocked compilation'); error.name = 'SecurityError'; throw error; };
    globalThis.WebAssembly = blocked;
    try {
      const data = new Float64Array([0, 1]);
      const reference = { n: 2, d: 1, indptr: new Uint32Array([0, 2, 4]), indices: new Uint32Array([0, 1, 0, 1]) };
      const result = helper === 'createRadiusScanner'
        ? api[helper](data, 2, 1, 1, 2, 1_000_000)
        : api[helper](data, reference, 1, 1_000_000);
      assert.equal(result, null);
    } finally { globalThis.WebAssembly = original; }
  });
}
