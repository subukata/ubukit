# WASM development and source recovery

## Project origin and license scope

On 2026-10-02, Seiki Ubukata clarified firsthand that the computational programs
under review were developed within UbuKit together with the AI assistant. This
is recorded as the maintainer's account of project development, without implying
legal co-authorship by the assistant. The project's MIT selection includes the
metric kernel and historical RMCM kernel; the earlier pending-external-origin
exclusions are superseded. [License scope](../LICENSE-SCOPE.txt) continues to
preserve actual third-party terms.

The development account is supported by the preserved acceleration work,
assembly mappings, candidate changes and contemporaneous development reports.
A missing source file in an archive is a preservation gap; it is not evidence
that the corresponding project code came from an unidentified third party.
These records do not make a general legal ownership or copyright determination.

## Preserved development trail

- The first inspected main-line import is
  [80caa7a0](https://github.com/subukata/ubukit/commit/80caa7a0b90e1299d43cad18107290dc307dbcd4).
  Its [baseline manifest](https://github.com/subukata/ubukit/blob/80caa7a0b90e1299d43cad18107290dc307dbcd4/preview/javascript/validation/baseline_package/SOURCE_MANIFEST.json)
  identifies `ubukit_extension_20261001_core.zip`, SHA-256
  `7c52a3efee20732a878b35517ac752dea568bc3e21b7bef47728e2925e98420b`
- In that retained archive, `baseline/ASSEMBLY.json` maps the metric wrapper to
  `work/metrics/tiled_candidate/metric-wasm.js` and the older RMCM wrapper to
  `work/rmcm/javascript/rmcm-wasm.js`. The selected-wrapper hashes match the
  retained source. `archive/select_payload.py` does not include the earlier
  metric work directory in its selection
- The metric wrapper identifies `distance_rows8.wat`, but that original WAT and
  its exact original build invocation were not found in the captured source.
  The checked-in reconstructed WAT provides a byte-exact build source with
  explicitly recorded current settings
- Preserved current RMCM WAT, its Float32 fragment and append generator show the
  later development steps. Reversing the appended function and the later
  `filter` counter/max-pairs check reproduces the complete older binary

## Source status and byte verification

The metric source under `reconstructed-wat/` is a labeled disassembly, not a
recovered original file. Its SHA-256 is
`8653a1cd960d2494ae47e8ff867e64aa95ad76c7b677493051a4f9613d77d5b3`.
Compiling with WABT 1.0.39 and debug names disabled reproduces the 1,246-byte
runtime binary, SHA-256
`b7b68848eecdca210e2bf13288c5313ee38d47bb315046c40ff143654cb61656`.

`derived-wat/rmcm/pre_f32_radius_candidates.wat` is a **derived historical
reconstruction** from the preserved later `original-wat/rmcm/radius_candidates.wat`.
It is separate from all unchanged original WAT and the existing disassembly.
With WABT 1.0.39 and debug names enabled, it reproduces the entire 3,097-byte
historical binary, SHA-256
`63768dc7e7388372fbbc1a6517e4e99b570699cbfb8f9d18b9ccbd897ca9ebd8`,
including the `name` custom section. No runtime kernel is replaced.

Run the bounded source-recovery check offline with an existing pinned WABT:

```sh
cd javascript/wasm
WABT_MODULE_PATH=/absolute/path/to/node_modules/wabt node tools/check-source-recovery.mjs
```

This check verifies source derivation and full-byte compilation. It does not
instantiate or execute a kernel, install dependencies, write runtime files, or
repeat the full numerical/browser/performance suite. Its JSON output can be
saved outside the repository. The existing `reconstruct:check` also verifies
that all seven retained disassemblies remain unchanged.

The historical source's original formatting/comments and the metric's missing
original build invocation remain unrecovered. Neither prevents preserving a
clearly labeled reproducible source form. Byte equality is a technical check,
not a substitute for the maintainer's separate development clarification.

## Separate external materials

SOM and scikit-learn notices remain unchanged. UCI-derived fixture labels retain
their dataset attribution. WABT is a pinned development tool with its own terms;
its license does not determine the kernel license. No package upload,
repository visibility change or public release is authorized by this record.
