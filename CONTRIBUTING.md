# Contributing

How to change UbuKit. What it is and why it is built this way is in
`DESIGN.md`; read that first.

## Checks

From the repository root, with Python 3.12+ and Node.js 22+:

```sh
cd python && pip install -e . --group dev
ruff check . ../fixtures ../bench && ruff format --check . ../fixtures ../bench && mypy && pytest
cd ../js && npm ci --ignore-scripts && npm test && npm run check
cd .. && python bench/run.py --size smoke --check
```

Tests mirror the modules in both languages (`python/tests/test_<module>.py`,
`js/test/<module>.test.js`, with shared JavaScript data in `js/test/helpers.js`).
`python/tests/typing_api.py` (checked by mypy) and `js/test/typing-api.js`
(checked by `npm run check`, in strict mode against the declarations that
npm users get) are not run: they call the API as typed code does, so a
public function that loses its types fails.
Run them before every pull request; CI repeats most of them on Ubuntu (see
"CI and dependencies"). ruff and mypy are each pinned to one minor series;
bump them in pull requests of their own.

## Kinds of change

| Change | Languages | Fixtures | CHANGELOG |
|---|---|---|---|
| Math, defaults or API | both, plus `docs/algorithms.md` | regenerate | "Changed results" / "Changed API" |
| Speed only | one is fine | regenerate; only the last digits may change | not needed |
| Bug fix | wherever the bug is | regenerate only if results were wrong | "Fixed" |

- Regenerate the fixtures with `python fixtures/generate.py`. The file has
  one case per line, so the diff names the cases whose results changed. A
  speed-only change may reorder floating-point operations and so change the
  last digits (differences below about 1e-15 times the scale of the data or
  value); commit them with the change and say so in the pull request, so
  that the file stays exactly what the code produces. A larger change means
  the math changed.
- A pull request that changes an algorithm includes benchmark tables; see
  below. A new algorithm also gets a case in `bench/cases.py`.
- A change to a loop that has a Numba kernel (`python/src/ubukit/_numba.py`)
  changes the kernel too; `tests/test_numba.py` fails until they agree. A
  new kernel needs a large measured gain over NumPy on few cores as well as
  many, a test against its reference there, and a "numba" benchmark case.
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
| What does UbuKit gain over the textbook code? | `python bench/run.py --baseline` |

`--compare` alternates the two trees in ABBA order and says "faster" or
"slower" only when the samples do not overlap and the medians differ by at
least 5%, and "changed" when the results differ. Identical trees measured
this way still differ by up to 5%. Times depend on the machine and on BLAS
threads (`--threads 1` fixes them for the Python side), so compare runs from
one machine only. Paste the tables into the pull request.

`bench/baseline.py` holds the textbook NumPy form of every method (not
shipped): `--baseline` times it, shows each time's speed-up over it, and
marks a case whose results differ from it, so a change to a method's results
changes its baseline too. It holds dense (N, N) arrays, so it runs at sizes
up to m.

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

## CI and dependencies

- CI (`.github/workflows/ci.yml`) runs on every pull request on Ubuntu:
  Python 3.12 and 3.14, the oldest supported NumPy and SciPy, Node 22 and 24,
  the package builds and the benchmark's parity check, in about two minutes.
  A new push to the pull request cancels the earlier run. The `main` ruleset
  requires one check, "All checks", which fails when any other job fails, so
  changing the jobs (a new Python or Node version) needs no ruleset change. Windows and macOS
  run when the maintainer starts CI by hand (Actions > CI > Run workflow),
  once each before a release. Standard runners cost nothing in a public
  repository; while the repository is private, where minutes are paid
  (macOS counts 10x and Windows 2x), pull requests skip every job.
- A new action is pinned to a full commit SHA with the version in a comment,
  gets only the permissions it needs, and checks out with
  `persist-credentials: false`.
- Dependabot proposes the month's updates to the actions and to the npm
  development dependencies as one pull request per ecosystem, and only
  versions at least a week old. CI runs on these pull requests like any
  other; before merging, also read the release notes of an updated action,
  which runs only in CI and releases.
  The Python requirements are deliberate lower bounds and are not updated.

## Releasing

Only the maintainer releases. A published version can never be replaced
(PyPI refuses a file name it has seen and npm a version number), so a broken
release is fixed by the next patch version.

Once, before the first release:

1. Make the repository public. npm does not generate provenance in a private
   repository, and `release.yml` and `js/package.json` require it; CI on
   standard runners is then free. Turn on private vulnerability reporting in
   the security settings, which `SECURITY.md` points to.
2. Create the `pypi` and `npm` environments, each limited to tags matching
   `v*`, with the maintainer as required reviewer and without the
   administrators' bypass, so every upload waits for a last confirmation.
   Add two tag rulesets for `v*`: "release tags: create" restricts
   creations, with only the repository admin role allowed to bypass it, and
   "release tags: fixed" restricts updates and deletions and blocks force
   pushes, with no bypass at all. A ruleset's bypass applies to every rule
   in it, so only the split lets the maintainer make release tags while
   nobody, the maintainer included, can move or remove one; fixing a
   mistaken tag means disabling the second ruleset on purpose. `release.yml`
   also refuses a tag whose commit is not on `main`.
3. On PyPI, add a pending trusted publisher for `ubukit`: owner `subukata`,
   repository `ubukit`, workflow `release.yml`, environment `pypi`.
4. On npmjs.com, add the trusted publisher for `ubukit` (same fields,
   environment `npm`) and set its publishing access to disallow tokens. npm
   trusts only an existing package, so 0.1.0 was uploaded once with a
   short-lived token, revoked right after.

Every release:

1. Run the checks on an up-to-date `main`, then CI once on each operating
   system.
2. In one pull request, set the version in `python/src/ubukit/__init__.py`
   and `js/package.json`, and replace "(unreleased)" in `CHANGELOG.md` with
   the date.
3. After merging it, tag the merge and push the tag
   (`git tag v0.1.0 && git push origin v0.1.0`). `release.yml` tests both
   packages, builds them in a separate job with only the build tools after
   checking that the tag is on `main` and matches both versions, and, once
   the environments are approved, publishes them to PyPI and then to npm. If
   the npm upload fails, fix the cause and re-run that job; PyPI already has
   the version.
4. Install the published versions in a clean environment
   (`pip install ubukit==0.1.0`, `npm install ubukit@0.1.0`) and fit one
   example.
