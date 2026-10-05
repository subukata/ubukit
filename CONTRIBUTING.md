# Contributing

How to change UbuKit. What it is and why it is built this way is in
`DESIGN.md`; read that first.

## Checks

From the repository root, with Python 3.12+ and Node.js 22+:

```sh
cd python && pip install -e . --group dev
ruff check . ../fixtures ../bench && ruff format --check . ../fixtures ../bench && pytest
cd ../js && npm ci --ignore-scripts && npm test && npm run check
cd .. && python bench/run.py --smoke
```

CI runs the same checks on Linux, Windows and macOS. ruff is pinned to one
minor series; bump it in a pull request of its own.

## Kinds of change

| Change | Languages | Fixtures | CHANGELOG |
|---|---|---|---|
| Math, defaults or API | both, plus `docs/algorithms.md` | regenerate | "Changed results" / "Changed API" |
| Speed only | one is fine | must not change | not needed |
| Bug fix | wherever the bug is | regenerate only if results were wrong | "Fixed" |

- Regenerate the fixtures with `python fixtures/generate.py`. The file has
  one case per line, so the diff names the cases whose results changed. If a
  speed-only change alters them, the math changed.
- A pull request that changes an algorithm includes the `python bench/run.py`
  table from before and after (`--only name` runs a subset).
- A design decision (a new principle, an exception to one, a choice between
  approaches) gets a short entry in `docs/decisions.md`.

## Done

- [ ] The checks above pass.
- [ ] A bug fix has a test that fails on the old code.
- [ ] `docs/algorithms.md`, and the "Current architecture" section of
      `DESIGN.md`, match the code.
- [ ] CHANGELOG and `docs/decisions.md` have their entries, if the tables
      above call for them.

## Pull requests

One change per pull request. When pull requests are stacked, retarget each
dependent to `main` before merging its base: deleting a merged base branch
closes the pull requests that target it.

## CI and releases

- A new action is pinned to a full commit SHA with the version in a comment,
  gets only the permissions it needs, and checks out with
  `persist-credentials: false`.
- To release, bump `python/src/ubukit/__init__.py` and `js/package.json`
  together, date the version in `CHANGELOG.md`, and let the maintainer push
  the `v*` tag. `.github/workflows/release.yml` tests and builds both
  packages without credentials, then publishes them with trusted publishing.
