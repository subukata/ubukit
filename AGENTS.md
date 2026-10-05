# Agent instructions

Read `DESIGN.md` before changing code. In short:

- Keep the library small. A new clustering method is an `assign` function on
  the shared engine (`python/src/ubukit/_core.py`, `js/src/core.js`).
- Never add alternative backends, version- or CPU-specific branches,
  extended-range rescue paths, or new runtime dependencies.
- Change Python and JavaScript together and keep names, parameters and
  equations identical (`docs/algorithms.md`). Regenerate fixtures with
  `python fixtures/generate.py` after numerical changes.
- Justify performance work with a measurement, and keep the simpler version
  unless the gain is large.

## Commands

```sh
cd python && pip install -e . --group dev
ruff check . && ruff format --check . && pytest
cd js && npm ci && npm test && npm run check
```
