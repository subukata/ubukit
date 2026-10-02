# Byte-exact WebAssembly rebuild supplement

This directory makes the JavaScript preview's embedded WebAssembly kernels
inspectable and reproducible without adding a runtime dependency. The current
runtime is the parent [JavaScript package](..), version `0.1.0-dev.6`.
Verification reads that package; it does not rewrite or publish it.

## What is recovered

- All **five current kernels** rebuild byte-for-byte
- **Four current kernels** match preserved original WAT; the metric kernel's
  missing original source is represented by explicitly **reconstructed WAT**
- All **seven distinct current/historical binaries** round-trip byte-for-byte,
  including both RMCM `name` custom sections
- All **13 historical WAT files** are preserved unchanged: ten complete modules
  compile and validate, and three files are generator fragments
- **3,967 bounded assertions** passed across seven groups on Node 24.19.0,
  Linux x64, against the dev.6 runtime

These are byte-reproducibility and bounded correctness results. They do not
establish new performance rankings, all-input equivalence, browser qualification,
original authorship, ownership or license clearance.

| Current wrapper | Bytes | Original source | Rebuild |
| --- | ---: | --- | --- |
| `kmeans-wasm.js` | 5,526 | `original-wat/som_kmeans/kmeans_r2w8.wat` | Exact |
| `metric-wasm.js` | 1,246 | Missing; reconstructed from binary | Exact |
| `rmcm-wasm.js` | 4,620 | `original-wat/rmcm/radius_candidates.wat` | Exact |
| `som-pca-wasm.js` | 474 | `original-wat/som_kmeans/pca_rotate.wat` | Exact |
| `som-training-wasm.js` | 9,429 | `original-wat/som_kmeans/som_train.wat` | Exact |

The historical modules are a 423-byte nearest-center experiment, matched by
`kmeans_nearest.wat`, and a 3,097-byte pre-Float32 RMCM module whose original
standalone WAT was not found. Neither replaces the current runtime kernel.
Experimental k-means tile variants compile but are not promoted as selected or
faster implementations merely because they compile.

## Run the checks

Requirements: Node 20+ with WebAssembly SIMD support and **WABT 1.0.39**. Install
the single pinned development dependency, then run the local checks:

```sh
cd javascript/wasm
npm ci --ignore-scripts --no-audit --no-fund
npm run verify
npm run reconstruct:check
```

The installation step may need network access. The verification and
reconstruction checks themselves are offline. `package-lock.json` retains the
npm registry URL and integrity digest for WABT 1.0.39. WABT is a development tool,
not a dependency of the UbuKit runtime.

To reuse an existing WABT installation without installing or copying anything:

```sh
WABT_MODULE_PATH=/absolute/path/to/node_modules/wabt npm run verify
WABT_MODULE_PATH=/absolute/path/to/node_modules/wabt npm run reconstruct:check
```

The override must identify the WABT package directory. The loader checks its name
and exact version. It does not independently authenticate an existing installation
against the npm tarball integrity digest.

`npm run verify` gives each of its two child stages a 60-second timeout. It checks
preserved source hashes, extracts the five current byte arrays, decodes the two
small historical fixtures, compiles WAT, compares bytes and SHA-256 digests,
checks imports/exports, and runs deterministic numerical/fallback cases. It
records results in `verification/*.json`. Temporary binaries and a disposable
runtime copy are created under `.generated/` and removed by the top-level runner,
including on a failed child stage. Directly running a child stage can leave those
temporary files behind. No runtime wrapper or public API is modified.

`npm run reconstruct:check` disassembles the seven captured binaries in memory
and requires the result to equal the checked-in reconstructed WAT. It writes no
files. For a separate integrity check of the checked-in supplement inputs:

```sh
sha256sum -c INPUT_SHA256SUMS
```

## Source and build evidence

- `INVENTORY.json` binds current wrapper hashes, exact embedded binary hashes,
  historical fixtures and recovered WAT. Current wrapper hashes refer to dev.6;
  historical wrapper hashes are retained separately because wrapper code can
  change while the embedded bytes remain identical
- `SOURCE_FILES.json` records exact hashes and lengths of all preserved WAT,
  historical scripts, historical tests/results and the encoded binary fixtures
- `original-wat/` contains unchanged historical source files
- `reconstructed-wat/` contains labeled disassemblies for all seven binaries
- `evidence/build-scripts/` preserves historical build commands, generators and
  tool metadata unchanged
- `evidence/work/` preserves representative historical tests and build results
  unchanged. Those complete historical suites are evidence, not the suites
  claimed to have been rerun here
- `evidence/historical-binaries.json` contains only the two small historical byte
  sequences, with their source-file hashes and historical relative locations
- `verification/` records the latest bounded checks, their environment, coverage
  and explicit limitations. Paths to generated bytes in those reports are
  ephemeral outputs removed by the top-level runner

Historical scripts expect their original directory layout and may write wrappers
or generated WAT. They are retained as evidence; use `npm run verify` for this
layout. The historical SOM/k-means compilation used `simd: true`, `resolveNames`,
`validate`, and `toBinary({write_debug_names: false})`. It selected
`pca_rotate.wat`, `kmeans_r2w8.wat`, and `som_train.wat`.

The historical RMCM build used `simd: true` and
`toBinary({log: false, write_debug_names: true})`; its embedding step inserted
`radius_candidates.wasm` into the wrapper. `generate_f32.py` appends a fragment to
an earlier source. The preserved full `radius_candidates.wat` already contains
that function, so the fragment must not be appended again.

Both RMCM binaries need their `name` custom section for complete byte equality.
The checker tries debug names both on and off and rejects a merely
executable-section-equivalent result. Omitting names removes 504 encoded bytes
from the current RMCM binary and 379 from the older one. The other five binaries
match with debug names disabled.

Reconstruction uses `readWasm({readDebugNames: true, simd: true})`, `applyNames`,
and `toText({foldExprs: false, inlineExport: false})`. It cannot recover missing
comments, formatting, omitted names, source language, author identity, ownership
or original compiler commands. No C source or C compiler flags were found in the
inspected evidence; none are invented here.

## Bounded correctness coverage

The test runner executes a disposable copy of the current package with verified
rebuilt bytes substituted. It covers:

1. All seven module ABIs, validation and initial memory
2. K-means scalar/SIMD parity across row, feature and center tails; ties and
   generator events; zero, underflow, overflow and fallback outcomes
3. SOM training and PCA scalar/SIMD parity, including generator events
4. Metric distances, one-based ranks, histogram counts, ties and end-to-end
   auto/histogram/radix results
5. RMCM Float64 and conservative Float32 candidates, exact radius boundaries,
   batch tails and CSR filtering
6. Workspace ownership, zero-budget behavior and unavailable-WebAssembly fallback

The original recovery supplement used an additional historical scalar runtime for
20 assertions. This repository supplement deliberately avoids bundling that
redundant runtime: its 3,967 assertions compare against the current dev.6 scalar
paths and are not a claim that the earlier 3,987-check suite was rerun unchanged.
The historical pre-Float32 RMCM and nearest-center modules receive ABI/byte checks,
not independent numerical qualification. The new check was run only in the
recorded Node environment, not in browsers. No performance benchmark was run.

## Provenance and licensing limits

Original-source provenance remains unresolved for the **metric kernel** and the
**older pre-Float32 RMCM kernel**. Reconstructed WAT makes their machine code
readable and reproducible; it does not turn it into recovered original source or
resolve authorship and license questions.

This supplement does not select a new project-wide license or grant new
redistribution rights. Existing runtime notices and scoped licenses remain in
[JavaScript package](..), including `NOTICE.txt`, `LICENSE-SOM.txt`,
`NOTICE-EXTERNAL-METRICS.txt`, and `LICENSE-SCIKIT-LEARN.txt`. Maintainers must review
those scopes and the unresolved provenance before making licensing or release
claims. WABT's package license is recorded in the tool lockfile; it does not
license the kernels themselves.
