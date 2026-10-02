# Verification scope

## Current JavaScript dev4 integration replay, 2026-10-02

The current pair is Python `0.0.0.dev3` and JavaScript `0.1.0-dev.4`.
Python's 60 runtime files are byte-identical to the merged dev3 snapshot.
The current JavaScript tree was materialized from PR source with every source
blob checked against its Git SHA, then packed and installed into a clean
consumer. No prepared node_modules or prior tarball was reused.

- Installed JavaScript package: 890 tests passed, zero failures or skips
- Independent 80-digit Decimal oracle: 3,043 ARI and 12,040 AMI comparisons;
  maximum absolute errors 0 for ARI and 3.33e-16 for AMI
- ESM isolation, clean VM module and real Node module-worker checks passed
- Current runtime snapshot: 60 Python + 25 JavaScript files, all hashes verified
- Tests, numerical references and committed evidence bound by VERIFICATION_SNAPSHOT.json
- Root README JavaScript example and all seven package entrypoints checked
- Actual browser execution is unverified; the earlier cloud-browser harness
  attempt was blocked. Node/VM results are not a browser compatibility claim
- No GitHub Actions workflow files or commit checks are configured; the checks
  above are local source replays, not a GitHub CI pass

The original [JavaScript report](javascript/REPORT.md) preserves the standalone
numerical and timing evidence. Integration edits change documentation and setup,
not production runtime code or recorded benchmark values. Rebuilt archive bytes
can differ from the standalone artifact because package documentation changed.

## Historical dev3 repository-source replay, 2026-10-02

At that replay, the 60 Python and 24 JavaScript runtime files matched the
previously validated dev3 artifacts exactly. Documentation/provenance paths were corrected for
repository use. Runtime code was not changed during synchronization.

The repository source was rebuilt and installed non-editably in fresh local
environments. Candidate imports are checked against installed distribution
ownership and runtime hashes. No source-tree substitution is used.

- Base Python: 2,314 runtime cases passed, 71 optional/reference skips; includes
  both original 8,000-point sparse cases
- Base Python new numerical/optimization suite plus standalone membership-axis
  and read-only preflight tests: 211 passed (200 new + 8 axis + 3 preflight)
- Base Python ARI/AMI: 27 facade tests passed, plus the complete numerical panel
  below and the portability script
- Numba Python: 2,456 runtime cases passed, 3 optional comparator skips;
  244 extra tests passed (233 new + 8 axis + 3 preflight), 31 external facade
  tests passed, plus the full ARI/AMI numerical and 12,212 backend-parity panel
- Root README Python/JavaScript examples and Python all-method examples passed
- Source-integrity audit: all 84 runtime hashes passed
- JavaScript: all 704 tests passed against a locally packed/installed tarball,
  including real Node Workers, sessions, cancellation/restart and optimizer tests

The final runtime artifact was rebuilt after documentation-only edits;
installation-origin, ownership and API guards were rerun. Artifact hashes differ
from the earlier local delivery because package documentation was updated.

## Earlier dev3 installed-artifact matrix

Before this repository synchronization, four clean environments passed: base
wheel, base sdist, compatibility Numba wheel, and final-stack Numba wheel.
These historical gates are distinct from the source replay above.

| Per environment | Base (without Numba) | Numba |
|---|---:|---:|
| Runtime passed | 2,314 | 2,456 |
| Runtime skipped | 71 | 3 |
| New numerical/optimizer tests passed | 200 | 233 |
| External metric facade tests passed | 27 | 31 |

Each environment also passed API/README, installed ownership, archive origin,
and uninstall/reinstall checks. The wheel rebuilt from the earlier sdist and
the earlier npm tarball rebuilt identically under the recorded tools; this is
historical evidence, not a byte-identity claim for documentation-edited archives.

ARI/AMI numerical panel per environment: 2,960 independent ARI, 11,708 independent
AMI, 3,053 sklearn ARI, 12,080 regular sklearn AMI, 132 singular, 1,116 invariance,
17 invalid-input/option and 4 synthetic overflow checks. Optional environments
also exercised 12,212 NumPy/Numba parity checks.

## Tested stack and limits

Linux x86-64, CPython 3.12.14, Node v24.19.0. NumPy 2.3.5, SciPy 1.17.0,
threadpoolctl 3.6.0, pytest 8.4.2. The final stack uses scikit-learn 1.8.0 and
optional Numba 0.67.0 / llvmlite 0.49.0. The historical compatibility stack uses
scikit-learn 1.7.2, Numba 0.63.1 / llvmlite 0.46.0. See constraints and hashed
requirements files for exact recorded dependencies.

- 13 reference-only fuzzy tests are excluded from the installed runtime harness;
  they never execute the candidate. Candidate-vs-reference checks remain
- Base runtime skips include 68 optional-Numba cases and 3 optional comparator
  cases. The Numba environments retain 3 optional scikit-fuzzy comparator skips
- Extreme numerical contract assertions were deliberately revised and then
  supplemented with independent oracle tests; unchanged-baseline success is
  not claimed. Read NUMERICAL_LIMITS.md
- Other OSes, architectures, Python versions, minimum dependency versions and
  real-browser dev3 execution remain unverified
- The separate browser demo uses an older vendor snapshot. Its browser checks
  do not establish dev3 browser compatibility
- Benchmarks are exploratory and are not universal speed or quality guarantees

Virtualenvs, dependency wheels and generated archives are not committed.
The selected JavaScript regression log, numerical evidence and benchmark reports
are committed under javascript/reports and javascript/audit; complete historical
Python machine logs are not included. REPRODUCE.md runs the included harnesses and writes local
results. SOURCE_SNAPSHOT.json and VERIFICATION_SNAPSHOT.json bind the submitted
runtime and test/reference bytes respectively.
