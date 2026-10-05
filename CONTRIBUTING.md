# Contributing

How to change UbuKit. What it is and why it is built this way is in
`DESIGN.md`; read that first.

## Checks

From the repository root, with Python 3.12+ and Node.js 22+:

```sh
cd python && pip install -e . --group dev
ruff check . ../fixtures ../bench && ruff format --check . ../fixtures ../bench && pytest
cd ../js && npm ci --ignore-scripts && npm test && npm run check
cd .. && python bench/run.py --size smoke --check
```

Run them before every pull request: CI does not run on pushes or pull
requests (see "CI and releases"). ruff is pinned to one minor series; bump it
in a pull request of its own.

## Kinds of change

| Change | Languages | Fixtures | CHANGELOG |
|---|---|---|---|
| Math, defaults or API | both, plus `docs/algorithms.md` | regenerate | "Changed results" / "Changed API" |
| Speed only | one is fine | must not change | not needed |
| Bug fix | wherever the bug is | regenerate only if results were wrong | "Fixed" |

- Regenerate the fixtures with `python fixtures/generate.py`. The file has
  one case per line, so the diff names the cases whose results changed. If a
  speed-only change alters them, the math changed.
- A pull request that changes an algorithm includes benchmark tables; see
  below. A new algorithm also gets a case in `bench/cases.py`.
- A design decision (a new principle, an exception to one, a choice between
  approaches) gets a short entry in `docs/decisions.md`.

## Benchmark

`bench/run.py` runs the cases of `bench/cases.py` in Python, JavaScript and,
where it computes the same quantity, scikit-learn, and reports time
(median, +- half the range), iterations, Python peak memory and quality.

| Question | Command |
|---|---|
| Is the change faster? Are the results unchanged? | `python bench/run.py --compare main --only name` |
| Did the behavior of the default settings change? | `python bench/run.py --quality --only name` |
| How does the time grow with N? | `python bench/run.py --size s,m,l --only name` |

`--compare` alternates the two trees in ABBA order and says "faster" or
"slower" only when the samples do not overlap and the medians differ by at
least 5%, and "changed" when the results differ. Identical trees measured
this way still differ by up to 5%. Times depend on the machine and on BLAS
threads (`--threads 1` fixes them for the Python side), so compare runs from
one machine only. Paste the tables into the pull request.

## Done

- [ ] The checks above pass.
- [ ] A bug fix has a test that fails on the old code.
- [ ] An algorithm change has the benchmark tables; a new algorithm has a case.
- [ ] `docs/algorithms.md`, and the "Current architecture" section of
      `DESIGN.md`, match the code.
- [ ] CHANGELOG and `docs/decisions.md` have their entries, if the tables
      above call for them.

## Pull requests

One change per pull request. When pull requests are stacked, retarget each
dependent to `main` before merging its base: deleting a merged base branch
closes the pull requests that target it.

## CI and releases

- CI (`.github/workflows/ci.yml`) runs only when the maintainer starts it
  (Actions > CI > Run workflow), on one operating system per run, typically
  once per system before a release. Actions minutes are limited while the
  repository is private (macOS counts 10x and Windows 2x), and the local
  checks cover each change.
- A new action is pinned to a full commit SHA with the version in a comment,
  gets only the permissions it needs, and checks out with
  `persist-credentials: false`.
- To release, bump `python/src/ubukit/__init__.py` and `js/package.json`
  together, date the version in `CHANGELOG.md`, and let the maintainer push
  the `v*` tag. `.github/workflows/release.yml` tests and builds both
  packages without credentials, then publishes them with trusted publishing.
