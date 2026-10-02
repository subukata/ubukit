# Rebuild embedded Float64 WebAssembly

Runtime JavaScript contains all three binaries; it performs no fetch and requires no npm package. Unsupported SIMD, blocked compilation or insufficient per-fit workspace uses the existing JavaScript route. Request the optional path with `kernelBackend: 'wasm'`; the default remains JavaScript.

For rebuilding only, use Node and the pinned WABT 1.0.39 development dependency in this directory. Run `node build-portable.mjs --check` to verify checked-in byte arrays or omit `--check` to replace them. `--output-root PATH` selects the JavaScript directory when packaging uses a different layout.

The selected k-means program is `kmeans_r2w8.wat`. `generate_kmeans.py` documents how the row/center tiling variants were generated. The selected SOM program is `som_train.wat`; regenerate it with `generate_som_prototype4.py` followed by `generate_som.py`. `pca_rotate.wat` implements the eigensolver row rotations. All arithmetic uses strict `f64`/`f64x2` add, subtract and multiply; no relaxed-SIMD, fused or reduced-precision instructions are used. JavaScript still performs exp/log so each engine retains its existing transcendental results and objective order.

Linear memories are created for each fit, with page rounding included in eligibility checks. They are never returned to callers. Output arrays remain ordinary transferable buffers. Generator yields/checkpoints remain in JavaScript. The working block is recopied after each yield so subsequent input updates are observed; inputs should remain stable within each synchronous chunk. Runtime compilation/code and VM overhead are outside primary-buffer byte counts.
