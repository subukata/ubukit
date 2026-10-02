# Repository layout

This first phase changes outer layout and development paths. Numerical implementation bytes, public APIs, package versions, and backend behavior are unchanged.

- `python/`: build metadata and the sole Python runtime root `src/ubukit/`; language tests and installed-artifact verification
- `javascript/`: npm metadata, 27 runtime JavaScript files, tests, fixed comparison references, and `wasm/` rebuild inputs
- `docs/`: user guides; `examples/`: existing small examples and browser demo
- `tests/contracts/`: cross-language reference index; `tests/integration/`: shared contract checks
- `benchmarks/`: manual reproducible benchmark drivers, separate from correctness gates
- `tools/ci/`: manual, budget-gated validation; `tools/provenance/`: current content snapshots

The active `javascript/validation/` baselines and Python verification oracles are deliberately retained because current regression tests import them. A filename containing “legacy” does not make its live contract redundant. Research-only sources, stale package artifacts, old reports, and old root API docs are preserved in the complete archive instead of competing with the current product entrypoints.

## Preservation and restoration

The complete original main snapshot is preserved privately before any file is omitted from this candidate:

- Repository: `subukata/ubukit`
- Full main commit: `5a197111b605c495f3f2a7ba69a6e09bb44e8caf`
- Git tree: `8915dff121685cae29437a1e56adcc187a4a9881`
- Original files: 1013; original bytes: 10382775
- Archive: `ubukit-main-5a197111b605c495f3f2a7ba69a6e09bb44e8caf.tar.gz`
- Archive SHA-256: `6587c138abdbeb7a54e9767e8ad47faca511fd5ad1c8b3296f360e16661fdf30`

The companion `full-main-manifest.json` records path, Git blob SHA-1, SHA-256, byte length, and file mode for every original file. A fresh extraction restored every file and verified its content and mode. The companion migration map classifies every original path as retained, moved, modified, or archive-only.

Use `python tools/archive/verify_restore.py --archive /path/to/the/archive.tar.gz --manifest /path/to/full-main-manifest.json --output /path/to/new-empty-directory` to verify and restore a private copy. Keep the archive and manifest together. The local archive does not promise permanent remote storage. Before omitting history from a remote default branch, choose and verify an approved private archival destination or persistent immutable ref. No such remote change is part of this local candidate.

Links to the immutable original commit in old investigation documents identify historical evidence. They do not certify the migrated package. No phase-1 result is a new performance claim or GPU/macOS/Windows/browser qualification.

## Later phase

Only after this outer-layout candidate and its installed artifacts pass review: consider a separate internal module cleanup inside `ubukit._impl` and JavaScript runtime modules. Keep API adapters, sample-major membership contracts, backend-selection rules, elite SOM/WASM kernels, numerical oracles, and license notices unchanged unless separately reviewed. Make one internal change per patch and rerun reference, installed-artifact, optional-backend and fallback checks.
