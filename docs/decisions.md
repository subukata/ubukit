# Decisions

Why UbuKit is the way it is, newest first. Each entry gives the decision, the
reason and what was rejected, with the pull requests that carried it. Add an
entry when a pull request makes or changes a design decision.

## 2026-10-06 One release checklist; Dependabot batched and delayed

The release steps, including the one-time setup, are one checklist under
"Releasing" in `CONTRIBUTING.md`, which starts by making the repository
public. Dependabot stays, with one pull request per ecosystem a month and
only for versions at least a week old. *Why:* the one-time steps lived only
in comments of `release.yml`, and a mistake is permanent because neither
registry accepts a version again; npm does not generate provenance in a
private repository, so the first tag would have failed there. Dependabot's
update jobs do not count against Actions minutes, and it keeps the pinned
actions of the publishing workflow current; since CI no longer checks its
pull requests, fewer and older updates are easier to check by hand and less
likely to carry a compromised release. *Rejected:* removing Dependabot and
refreshing the pins by hand before each release (an easy step to skip), and
publishing npm without provenance while private.

## 2026-10-06 A typed public Python API

Every public Python function and the TPE class annotate their parameters and
results (`ArrayLike` inputs, `Literal` choices such as `average`, and the
aliases `Init`, `MapInit`, `Grid` and `Space`), and the internal helpers are
typed so that mypy finds nothing beyond SciPy's missing stubs. *Why:* the
package declared `Typing :: Typed` while 28% of the parameters were
annotated, so editors and type checkers knew little about the API; now
mypy `--strict` accepts correct calls and catches a string `k`, missing
`lam`/`gamma` and an unknown `average`. *Rejected:* dropping the `Typed`
classifier instead (gives up editor help for users). mypy, pinned to one
minor series like ruff, is part of the local checks (SciPy's missing stubs
are ignored in `pyproject.toml`), so the annotations stay true.

## 2026-10-06 One engine shape in both languages

Python's `iterate` is now a generator that takes the method's `view`, like
JavaScript's, and each fitting function is written once as a generator;
`stepwise` turns it into the plain function and lists it in `ubukit.steps`,
whose iterations offer `result()` as in JavaScript. *Why:* after the
JavaScript-only changes the engines had drifted apart (Python returned a
tuple that four methods each turned into a Result, its k-means modified its
state in place, and `DESIGN.md` described only JavaScript), and plotting the
SOM-OLP learning for research needs the real iterates in Python too. Results
and speed are unchanged (benchmark ratios 0.85-1.05). *Rejected:* a
`callback` argument on every function (duplicates the generators) and
keeping the generators JavaScript-only (two engine shapes to maintain).

## 2026-10-06 Every iteration's result, on demand

The JavaScript step generators yield `{ iteration, result }`, where
`result()` returns the Result the run would return had it stopped at that
iteration; the loop builds it with the same function that builds the final
Result. *Why:* the browser demo could show only finished runs: the
generators yielded just the iteration number, so drawing the SOM-OLP
learning meant restarting from the centers, which reset the memberships the
latent positions come from. *Rejected:* building every intermediate Result
eagerly (O(NK) per iteration even when unused), an `onIteration` callback on
every function (duplicates the generators), and an API to resume from a
saved state (exposes internal state; keeping the generator already resumes).

## 2026-10-05 CI only by hand while the repository is private (#50)

CI runs only when the maintainer starts it, on one operating system per run,
and agents may not trigger workflows without asking. *Why:* the rebuild
(#36) replaced the previous workflow, which ran only by hand on one system per
run after confirming the remaining minutes, with CI on every pull request and
every push to `main` on three systems. In one day (#36-#49) that ran about 33
times at over 100 billed minutes each (macOS counts 10x) and used the
account's entire 2,000 monthly minutes, while every problem it could have
found had already been caught by the local checks. *Rejected:* removing CI
altogether (it is the only check on macOS and Linux before a release) and
Linux-only CI on every pull request (still repeats the local checks).

## 2026-10-05 Hamerly's bounds for k-means (#49)

k-means runs Lloyd's iterations as a step with Hamerly's (2010) bounds: a
point keeps its label without computing its other distances when its exact
distance to its center is below half the distance from that center to the
nearest other one and below a lower bound on its distance to every other
center. Points within 1e-9 of a bound are recomputed, so the iterates are
Lloyd's (bitwise in JavaScript; fixtures unchanged). *Why:* `bench/run.py
--compare` showed Python 1.6x faster on separated blobs, 1.45x on uniform
noise and 3.2x with K = 100 (scikit-learn's gap there fell from 16x to about
4x), and JavaScript 1.9-4.5x faster on every case. *Rejected after
measuring:* the second-nearest distance from `np.partition` (15% slower than
Lloyd on overlapping clusters, where most points are recomputed; masking the
nearest and taking the minimum is not), keeping only the half-distance test
(slower than Lloyd), and Elkan's K lower bounds per point (K times the
memory for a gain that matters only at large K).

## 2026-10-05 Greedy k-means++ seeding (#48)

Each seed is the best, by the resulting sum of squared distances, of
2 + floor(ln K) candidates drawn in proportion to the squared distance to the
nearest seed (as in scikit-learn), instead of a single draw. *Why:*
`bench/run.py --quality` showed single draws putting two seeds into one
well-separated cluster: over 10 seeds, k-means ARI rose from a median of
0.87 to 1.0 (26.5 to 2 iterations), K = 100 from 0.84 to 0.96, and EFCM, RCM
and RMCM from about 0.87 to 1.0; recovering 10 separated blobs exactly went
from 6 to 16 of 20 seeds in Python and 4 to 18 in JavaScript. Seeding costs
2 + floor(ln K) times more distance evaluations, which fewer iterations
usually repay. *Rejected:* k-means|| (built for distributed data) and
several restarts (multiplies the whole run).

## 2026-10-05 A benchmark that can tell differences apart (#47)

The benchmark now compares two git trees in alternating ABBA order and calls
a difference only for non-overlapping samples at least 5% apart; it reports
medians with spreads, iterations, Python peak memory, BLAS threads, a
scikit-learn column, quality over seeds with the default initialization
(`--quality`), growth with N per iteration (`--size s,m,l`), and fails CI if
the languages disagree at small sizes. *Why:* the first benchmark ran each
side separately and could not resolve differences under about 20% (Python
batch SOM ranged over 0.182-0.215 s for one tree); BLAS threads moved times
by 20%; and scikit-learn's k-means, timed without a warm-up (0.063 s cold,
0.019 s warm), looked 1.4x faster than ours when it is 5.5-7.6x faster per
iteration. The 13-15% gains reported for
EFCM and SOM-OLP in #45 were within that noise and are not claimed; FCM's
1.4x is. Identical trees still differ by up to 5% under ABBA, hence the
threshold. *Rejected:* tracking times in CI, whose shared runners are too
noisy to compare.

## 2026-10-05 Documents by role (#46)

`DESIGN.md` says what and why, `CONTRIBUTING.md` how, `AGENTS.md` only what
is specific to agents, and this file the reasons; `CLAUDE.md` just includes
`AGENTS.md`. *Why:* the same rules lived in `DESIGN.md` and `AGENTS.md`, and
three of six pull requests had to edit both; principles named functions and
current state, so refactors rewrote them; and the reasons existed only in
pull request descriptions. *Rejected:* a separate file per agent tool, which
would duplicate the rules again.

## 2026-10-05 JavaScript distances stay direct differences (#45)

Plain loops replaced TypedArray callbacks on the hot paths (FCM 1.4x faster,
bitwise-identical results). *Rejected after measuring:* the Gram identity
(4.19 to 4.76 ms, no gain in V8, less accurate) and four-accumulator loops (at
most 1.14x at D = 64, slower at small D). The remaining gap to Python is
scalar loops versus BLAS.

## 2026-10-05 Windowed expected mutual information (#44)

AMI sums the hypergeometric expectation only within sqrt(35 min(a, b)) of its
mean; by Hoeffding's inequality the omitted mass is below 2 exp(-70), so the
value equals the full sum to rounding (differences at most 1e-18), 10-12x
faster at N = 1e6. *Rejected:* keeping the full sum for exactness that
rounding already erases.

## 2026-10-05 One loop, many steps (#43)

The engine owns the control (stopping, limits, history, progress) and methods
supply a step; the standard step is `lloyd`, and the online SOM's step is an
epoch. *Why:* exact accelerations such as Hamerly's k-means and optional
compiled kernels replace the step, not the loop, and the online SOM had a
loop of its own in each language. *Rejected:* letting accelerated methods own
their loops (principle 1), and a distance hook only (too narrow for bounds
and epochs).

## 2026-10-05 Benchmark, fixture format and change rules (#42)

`bench/run.py` defines each case once and times and scores it in both
languages; fixtures have one case per line; speed-only changes may touch one
language and must leave the fixtures unchanged, so an unchanged fixture file
means a pure speedup. *Why:* "justify with a measurement" had no measurement,
quality regressions had no number, and the one-line fixture file made diffs
unreadable.

## 2026-10-05 Numba as an optional, explicit kernel (#42)

A Numba kernel may replace a hot step if it is installed as an extra, chosen
per call (`engine="numba"`), reproduces the reference iterates, passes the
same tests and shows a large benchmark gain. None exists yet. *Why:*
sequential, branching steps (online SOM updates, bound checks in accelerated
k-means) do not vectorize, and the maintainer wants that speed. *Rejected:*
using Numba automatically when installed (results would depend on the
environment) and a Numba backend for everything (a second implementation of
every algorithm).

## 2026-10-05 PCA initialization (#40, #41)

Python eigendecomposes the smaller of XᵀX and XXᵀ; JavaScript uses
Rayleigh-Ritz on a Krylov basis of at most 64 vectors (exact for D <= 64);
each axis is oriented by its first clearly nonzero component. *Why:* the D x D
eigendecomposition took 5.9 s at D = 4000 in Python and 25 s at D = 784 in
JavaScript (now 14 ms and 0.18 s), and orienting by the largest
component mirrored the map on standardized 2-D data, whose axes tie in
magnitude (59 of 200 row reorderings). *Rejected:* subspace iteration
(fragile near ties) and Householder plus QL in JavaScript (much longer code).

## 2026-10-05 Input scale contract (#40, #41)

Rows must have norms below 1e150 and, unless all rows are equal, some feature
must span at least 1e-150. *Why:* data at 1e200 gave NaN memberships and
merged clusters, and data at 1e-170 gave ARI 0 marked "converged" and
trustworthiness 1.0, all without an error. *Rejected:* rescaling internally,
a rescue path that principle 3 rules out.

## 2026-10-05 Release path (#38, #39, #40)

Trusted publishing to PyPI and npm from `release.yml`: a credential-free job
tests and builds both packages, and publishing jobs only upload those files.
The first npm release uses a short-lived token because npm can only trust an
existing package. Dependabot does not manage the Python requirements, which
are deliberate lower bounds. *Why:* no third-party code may run where a
publishing token can be minted. *Rejected:* long-lived tokens and the
previous hand-made provenance and archive checks.

## 2026-10-05 Rebuild around one engine (#36)

The library was rewritten as one alternating loop on NumPy/SciPy and plain
JavaScript. *Why:* the previous code had accumulated alternative paths and
machinery (subnormal recovery in FCM, warm-memory and metric fast paths,
session storage, exact-identity release pinning and archive verification;
#15-#35) that made it slow to change and hard to verify. *Rejected:*
cleaning it up path by path; the maintainer chose to replace it.

## 2026-10-05 Agreement across languages by equations, not bits (#36)

Python and JavaScript share names, parameters and equations, each uses its
own random generator, and fixtures check that JavaScript reproduces Python to
~1e-9 when the initialization is given. *Rejected:* bit-identical results,
which would need a shared random generator and no BLAS.
