# Python dev6: one installed namespace

The current private Python package is `0.0.0.dev6`. Only `ubukit` is installed
at the top level. The existing 38 public exports, signatures, defaults, result
fields, stopping conditions and numerical bodies are retained. JavaScript
`0.1.0-dev.6` and all its runtime bytes are unchanged.

## Import and saved-state migration

Use `import ubukit` and its existing public functions. The former `portable_accel`,
`ubukit_fcm`, `ubukit_rmcm`, `rough_cmeans`, `_numba_kernel`, `external_metrics`
and `_external_metrics_numba` top-level names are not installed or aliased.
Internal implementations now live under `ubukit._impl`; these are private paths,
not additional supported public APIs. The localized SOM experiment remains
explicit opt-in at `ubukit._impl.portable_accel.som_olp_localized`.

The runtime source mapping is in
[the migration manifest](python/verification/namespace/migration-change-manifest.json).
Of 62 inherited Python files, 58 retain exact bytes and four change only imports,
facade resolution/version or docstrings. All 62 normalized numerical ASTs match.
One private package marker is added, giving 63 Python runtime files.

Use a clean environment. Old pickle module paths are unsupported; preserve the
old dev5 environment when an existing saved object still needs it. Rebuild Numba
caches in a new directory rather than reusing old compiled caches. The existing
FCM/RMCM missing-Numba error strings still mention `ubukit-fcm[numba]` or
`ubukit-rmcm[numba]`. For this package, install the aggregate wheel's `[numba]`
extra instead. The messages were retained to avoid silently changing error contracts.

No old release artifacts, frozen numerical oracles, historical measurements or
earlier Git checkpoints are replaced. Retired top-level Python source files are
removed only from the current installed-source tree. No GPU prototype is included.

## Completed candidate validation

The final standalone dev6 candidate completed 30 stages across fresh Linux
CPython 3.12.14 base-wheel, Numba-wheel and base-sdist environments:

- Base wheel/sdist, each: 128 main tests, 200 extreme/HPO tests, 2,314 inherited
  runtime tests, 27 external-metric facade tests; optional skips remain recorded
- Numba wheel: 135 main, 233 extreme/HPO, 2,456 runtime and 31 external facade tests;
  three optional comparator skips and 12,212 backend-parity comparisons
- Exact ownership/source/wheel/sdist/installed binding for all 63 runtime files
- 38 export names, 36 inspectable signatures and 30 fixed dev5/dev6 result groups match
- 113 high-precision cases and four Numba families under old-name collisions;
  separate cold-cache save and warm-process reload checks passed

The detailed aggregate is
[candidate-validation.json](python/verification/namespace/candidate-validation.json).
These are completed candidate checks, not a new full-suite run during GitHub
integration. Fixed fixtures do not prove all-input bitwise equivalence or speed.
The stack used NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 and threadpoolctl
3.6.0; the optional stack used Numba 0.67.0 and llvmlite 0.49.0.

## Repository integration checks

Integration starts from main `2dcb8cfed9fa52a9c7ac981b038978e559895834`.
All final candidate package-source bytes are preserved. The separately reviewed
PR11 overflow-portability and nonwinning-center test body is retained from main,
with namespace imports updated; it is not replaced by the older copy in the
standalone candidate. This preserved test adjustment is outside the candidate's
reported full-suite run and receives its own focused checks.

[NAMESPACE_VERIFICATION.json](NAMESPACE_VERIFICATION.json) records the focused
repository-source checks, source/hash audit and CI-driver plans. The 90 current
runtime files and all current verification inputs are snapshot-bound.

For a reproducible read-only comparison against preserved dev5 sources:

```sh
python3.12 preview/python/tools/check_namespace_migration.py --baseline-root /path/to/preserved-dev5/staging/src
```

## CI and remaining gates

The manual CI drivers are namespace-aware. The historical approved-SHA lock,
manual dispatch and budget safeguards remain fail-closed. A candidate source SHA
and cost preflight require separate approval before a remote run. This source
integration does not dispatch Actions or start external compute.

Prior Python dev5 Windows/macOS results do not qualify Python dev6. No new
cross-platform run, speed/memory benchmark, minimum-dependency matrix, registry
publication or project-license decision is included. JavaScript qualification
and historical CI evidence retain their original scope.
