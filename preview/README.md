# UbuKit private preview: Python dev3 / JavaScript dev4

This is the current integrated source snapshot. Start with the repository
[README](../README.md), [REPRODUCE.md](REPRODUCE.md), and
[VERIFICATION.md](VERIFICATION.md).

- Python: `python/staging`, version `0.0.0.dev3`, 31 lazy facade exports
- JavaScript: `javascript/package`, version `0.1.0-dev.4`, seven export entrypoints
- JavaScript ARI, AMI and shared-contingency joint scores; see [external metrics](javascript/package/EXTERNAL_METRICS.md)
- Optional lightweight TPE/random optimization; no added mandatory dependency
- Reviewed FCM/SOM extreme-range repairs and numerical diagnostics
- Complete installed-package regression harnesses with frozen test-only references

Python's 60 runtime files are unchanged from the verified dev3 snapshot.
JavaScript dev4 adds external metrics and updates the root exports; all 25 JS
runtime files are bound by SOURCE_SNAPSHOT.json alongside the Python files.
VERIFICATION_SNAPSHOT.json binds the tests, references and supporting evidence.
The JavaScript source was packed and installed in a clean consumer for 890 tests,
with an independent high-precision audit and Node/VM-worker smoke checks.
Actual browser execution remains unverified.

Package documentation was adapted for repository use, so rebuilt archive bytes
can differ from the earlier standalone delivery. No registry publication, new
project-wide license or broad performance guarantee is implied.

The repository's old top-level source trees remain historical checkpoints.
Use the paths above for new work. Read NUMERICAL_LIMITS.md before extreme inputs.
