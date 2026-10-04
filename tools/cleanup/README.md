> Historical qualification only. Its EFCM exact-baseline runner and tests are disabled under the float64-only policy. Use `python/tests/test_entropy_fcm.py` and `javascript/tests/entropy-fcm.test.mjs` for the current EFCM contract; previous exact-identity evidence does not qualify this numerical change.

# Focused cleanup regression

Run the reviewed EFCM/TPE/JS cleanup checks against a freshly installed candidate
Python package and this checkout's JS runtime. The controls do not install,
download, publish, benchmark, modify product code, or change security settings.
Use Python 3.12 on POSIX and Node >=20. The recorded run used Python3.12.14,
Node24.19.0, NumPy2.3.5, SciPy1.17.0, scikit-learn1.8.0,
threadpoolctl3.6.0/joblib1.5.3, with Numba absent.

After installing this checkout's Python package into a clean environment:

    python3 tools/cleanup/run.py --python /absolute/venv/bin/python --output /absolute/new/cleanup-check

The Git object for alpha3-equivalent base
`6828121d78fac3fb25be7c6d499eec73344c2520` must exist locally. No fetch is attempted.
An already materialized baseline checkout can instead be supplied with
`--baseline-root /absolute/base-checkout`; all baseline runtime hashes are checked.
`--repo` defaults to this checkout. Output must be new, outside the checkout.
The Python interpreter must import the exact candidate package from site-packages;
its 64 source hashes are verified before tests. The declared npm payload is copied from
this checkout into the generated consumer to preserve exact import-path checks. Test layouts and baseline runtimes are staged only in output.

The five added test files preserve the previously verified assertion bytes.
Existing tests and fixtures are reused from the repository, without importing
benchmarks or creating duplicate tracked runtime trees. `contract.json` binds
baseline/candidate source and preserved repository/test input hashes. This is an
exact-candidate regression gate; future intentional source/test edits require
reviewing/updating this explicit contract.

Expected results: Python29 EFCM +71 TPE +5 namespace/combined =105;
JavaScript402 regressions +8 public API checks =410. The API checks inspect exports,
arity and prototypes, not exhaustive behavior. Python60CPU/90wall seconds per
invocation, 512MiB virtual address limit for TPE and1GiB for other suites;
BLAS/OpenMP1 thread. Node90CPU/120wall seconds, sequential files, no address-space
limit. CPU is per process; wall timeout covers the process group.

This is focused local correctness. It does not replace full release,
Numba-present, browser, cross-platform, or performance qualification. Durations
printed by test frameworks are incidental audit metadata, not benchmark results.
