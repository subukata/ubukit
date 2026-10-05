# Agent instructions

Read `DESIGN.md` before changing code. In short:

- Keep the library small. A new clustering method is an `assign` function on
  the shared engine (`python/src/ubukit/_core.py`, `js/src/core.js`).
- Never add alternative backends, version- or CPU-specific branches,
  extended-range rescue paths, or new runtime dependencies. Optional Numba
  kernels follow DESIGN.md principle 2.
- Changes to math, defaults or API go into Python, JavaScript and
  `docs/algorithms.md` together; regenerate the fixtures with
  `python fixtures/generate.py` and list changed results in `CHANGELOG.md`.
  Speed-only changes may touch one language and must leave the fixtures
  unchanged.
- Justify performance work with `python bench/run.py` (before and after), and
  keep the simpler version unless the gain is large.

## Commands

From the repository root:

```sh
cd python && pip install -e . --group dev
ruff check . ../fixtures ../bench && ruff format --check . ../fixtures ../bench && pytest
cd ../js && npm ci --ignore-scripts && npm test && npm run check
cd .. && python bench/run.py --smoke
```
