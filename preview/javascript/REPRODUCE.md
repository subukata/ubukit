# Reproduce JavaScript dev5 SOM and external metrics

This repository snapshot pairs JavaScript `0.1.0-dev.5` with Python
`0.0.0.dev4`. Both add traditional SOM; the external-metric runtime is unchanged. Runtime dependencies added: zero. The source is a private preview;
no npm/PyPI publication or deployment is implied. For both languages, start with
[the aggregate replay instructions](../REPRODUCE.md).

## Install and verify

Run from this directory with Node >=20 and npm:

```sh
mkdir -p artifacts
cd package
npm pack --ignore-scripts --pack-destination ../artifacts
cd ../consumer
npm install --ignore-scripts --no-audit --no-fund ../artifacts/ubukit-js-0.1.0-dev.5.tgz
cd ..
node --test --test-concurrency=1 tests/*.test.mjs consumer/tests/*.test.mjs validation/tests/*.js validation/tests/*.mjs
UBUKIT_AUDIT_SOURCE=../consumer/node_modules/ubukit-js/src/external-metrics.js node audit/audit-numerics.mjs
node tools/check-package-docs.mjs
node --experimental-vm-modules tests/runtime-compatibility.mjs
node --experimental-vm-modules tests/som-browser-compatibility.mjs
```

Use a clean consumer without an old node_modules or package-lock.json for a
fresh replay. All candidate test imports resolve to a real installed tarball copy. The package
manifest verifies all 26 runtime hashes. The complete regression suite has 945
tests; the independent numerical audit is separate and contains thousands of
scalar, joint, invariance, domain and synthetic-count checks.

The pure-module VM and real Node module-worker smoke are executable here. The
browser harness (`browser-smoke.html`, `browser-worker.mjs`) can be served by a
local static server, but the cloud-browser attempt was blocked by client policy.
No actual browser execution is claimed in this repository snapshot. Test and
audit commands can regenerate report JSON files. Keep the committed measurements
when reviewing a replay; generated timings are not a reason to rewrite frozen
benchmark evidence.

## Preserve historical evidence

Use a throwaway checkout for reference regeneration or timing reruns: these
programs overwrite tracked JSON. `tools-build-report.py` is retained as the
historical standalone-report generator and is not an integration replay step;
it can restore obsolete delivery wording and regenerate matched-summary files.
Do not run it to update this repository's integration report. Keep all original
measurement values and distinguish new observations from the committed record.

## Independent numerical references

`audit/numeric-reference.json` was generated using 80-digit Decimal arithmetic,
exact integer binomial hypergeometric probabilities, and the original MI/EMI
formula. This is independent of the production conditional-entropy recurrence.
Use Python 3.12+ with scikit-learn 1.8.0 to regenerate the recorded oracle and
conditioning files (the Decimal oracle itself uses only the standard library):

```sh
python audit/generate_numeric_reference.py
```

The generator also runs its sklearn-conditioning panel.
`tests/generate_references.py` requires NumPy and scikit-learn 1.8.0 and
regenerates the additional sklearn/real-dataset panel. These are development-only
requirements; they are not JavaScript package dependencies.

## Historical external-metric timing (dev4 / Python dev3)

Set `PYTHON_18` to the absolute path of a Python interpreter with scikit-learn
1.8.0. Set `PYTHON_DEV3` to the absolute path of a separate environment with the
installed UbuKit Python dev3 package, scikit-learn 1.8.0, NumPy and Numba. These
are shell variables naming existing interpreters, not commands to install them.

Reserve a quiet serial slot; do not run suites/other benchmarks concurrently.
The broad seven-case benchmark has three warmups and nine single-call samples:

```sh
node bench/run-node.mjs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_18" -I -B bench/run-sklearn.py
```

For the recorded matched N=10,000/K=50 panel, use the installed Python dev3
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

## Traditional SOM references and timing

The new tests cover online sample updates, true frozen-BMU batch epochs, sessions,
projection, ownership, schedule endpoints and exceptional arithmetic. A clean VM
and real Node workers are separate runtime checks, not actual browser evidence.

`tools/generate-som-reference.py` rebuilds the ten-case shared scalar fixture
using the Python standard library. `bench/som-bounded.mjs` reruns the traditional
SOM bounded panel and overwrites `reports/som-bounded-benchmark.json`. Run either
only in a throwaway checkout and compare without rewriting frozen evidence.
The measurement scope is documented in [../SOM_ADDITION.md](../SOM_ADDITION.md).

## Repository integrity and layout

The current production package is `package/`, with tests and numerical evidence
alongside it in this directory. Build the tarball locally as shown above;
generated archives are ignored and are not tracked. No standalone patch file or
external integration manifest is required to reconstruct this repository tree.

From the repository root, run `python3.12 preview/tools/verify_snapshot.py` on a
clean checkout to verify 61 Python and 26 JavaScript runtime hashes plus the
committed verification snapshot. Replaying tests can rewrite result JSON, so
preserve or restore committed evidence before repeating that integrity check.

Do not transplant only `index.js`: the SOM runtime, session validation, session
implementation, shared PCA export, package metadata and manifest must travel
together. Preserve the external-metric file, export map and BSD notice as well.
The current aggregate scope is documented in [../VERIFICATION.md](../VERIFICATION.md).
