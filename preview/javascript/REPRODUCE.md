# Reproduce the JavaScript dev4 external-metrics candidate

This is a separate JS dev4 candidate. The verified JS dev3 package and Python
`0.0.0.dev3` are not replaced. Runtime dependencies added: zero. Package and
source patches have not been published or pushed by this task.

## Install and verify

Run from this directory with Node >=20 and npm:

```sh
cd package
npm pack --ignore-scripts --pack-destination ../artifacts
cd ../consumer
npm install --ignore-scripts --no-audit --no-fund ../artifacts/ubukit-js-0.1.0-dev.4.tgz
cd ..
node --test --test-concurrency=1 tests/*.test.mjs consumer/tests/*.test.mjs validation/tests/*.js validation/tests/*.mjs
UBUKIT_AUDIT_SOURCE=../consumer/node_modules/ubukit-js/src/external-metrics.js node audit/audit-numerics.mjs
node --experimental-vm-modules tests/runtime-compatibility.mjs
```

All candidate test imports resolve to a real installed tarball copy. The package
manifest verifies all 25 runtime hashes. The complete regression suite has 890
tests; the independent numerical audit is separate and contains thousands of
scalar, joint, invariance, domain and synthetic-count checks.

The pure-module VM and real Node module-worker smoke are executable here. The
browser harness (`browser-smoke.html`, `browser-worker.mjs`) can be served by a
local static server, but the cloud-browser attempt was blocked by client policy.
No actual browser execution is claimed in this deliverable.

## Independent numerical references

`audit/numeric-reference.json` was generated using 80-digit Decimal arithmetic,
exact integer binomial hypergeometric probabilities, and the original MI/EMI
formula. This is independent of the production conditional-entropy recurrence.
Use Python 3.12+ to regenerate:

```sh
python audit/generate_numeric_reference.py
```

The optional sklearn-conditioning part of that generator requires scikit-learn
1.8.0. `tests/generate_references.py` requires NumPy and scikit-learn 1.8.0 and
regenerates the additional sklearn/real-dataset panel. These are development-only
requirements; they are not JavaScript package dependencies.

## Timing

Reserve a quiet serial slot; do not run suites/other benchmarks concurrently.
The broad seven-case benchmark has three warmups and nine single-call samples:

```sh
node bench/run-node.mjs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_18" -I -B bench/run-sklearn.py
```

For the current matched N=10,000/K=50 panel, use the installed Python dev3
environment containing scikit-learn 1.8.0, NumPy and Numba. Its Python distribution
and RECORD ownership/hash are checked by the harness:

```sh
node bench/run-matched-node.mjs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
NUMBA_CACHE_DIR="$PWD/reports/fresh-numba-cache" \
"$PYTHON_DEV3" -I -B bench/run-matched-python.py
```

Both matched harnesses use 20 warmups and 21 samples, five calls per sample,
reporting median milliseconds per call. Labels are generated once by JS and
read identically in Python; both reports include their SHA256. Input allocation
is outside timing; validation, encoding and contingency construction are inside.
The sklearn joint baseline is two public calls. Numba measurements are warm and
its first AMI compilation/cache-loading call is separately reported. Use a fresh
Numba cache directory when reproducing that cold cost.

`bench/reference-baseline.mjs` is a straightforward zero-dependency JS comparison
using one-pass Map contingency construction and a scalar full-support
hypergeometric expectation for every cluster pair. It avoids deliberately slow
quadratic pair-label enumeration. It does not compress repeated marginal sizes
or use BigInt ARI products. Its code and exact timing inputs are included.

## Integration

`external-metrics-dev4.patch` is a production-package patch rooted at
`preview/javascript/package` against dev3 GitHub snapshot commit
`80caa7a0b90e1299d43cad18107290dc307dbcd4`. It includes the runtime, exports, version,
manifest, API documentation and retained license notice. `INTEGRATION_MANIFEST.json`
lists additional verification files and their intended relative paths.

Do not transplant only `index.js`: the new runtime file, export map, package file
list and BSD notice must travel together. Do not overwrite the Python version.
A repository-wide current-snapshot manifest and root README need an explicit
integration update after this patch is accepted; they are not silently rewritten
by this isolated development task.
