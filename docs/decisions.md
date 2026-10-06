# Decisions

Why UbuKit is the way it is, newest first. Each entry gives the decision, the
reason and what was rejected, with the pull requests that carried it. Add an
entry when a pull request makes or changes a design decision.

## 2026-10-06 CI on every pull request, on Ubuntu, once public (#73)

CI runs its six Ubuntu jobs on every pull request, and a new push cancels
the earlier run; Windows and macOS still run by hand, once each before a
release. While the repository is private, pull requests skip every job, and
agents may start no workflow but this one. *Why:* the repository is about
to become public, where standard runners cost nothing; the runs of
2026-10-05 took 1 to 2.3 minutes from start to finish, three systems
included (6.6 job-minutes for 14 jobs), and Dependabot's pull requests no
longer need checking out by hand. The decision of #50 was about paid
minutes, which the skip keeps should the repository become private again.
*Rejected:* all three systems on every pull request (Windows and macOS
rarely differ here, and their jobs are the slowest) and keeping CI manual
(no reason once runs are free).

## 2026-10-06 scikit-learn's notice for three expressions in ARI and AMI (#72)

`THIRD_PARTY_NOTICES.txt` (in the repository and both packages) keeps
scikit-learn's BSD 3-Clause notice for the ARI expression from pair counts
with its fn = fp = 0 case, the AMI of 1.0 for single clusters, and the
signed epsilon of the AMI denominator; comments mark them in both
languages, and the packages declare `MIT AND BSD-3-Clause`. *Why:* the code
before the rebuild (#36) recorded these as adapted from scikit-learn and kept
its notice; the rebuild kept the same expressions but not the notice, which
a check before making the repository public found. Publishing the packages
redistributes them, and BSD-3-Clause asks that its notice go with them.
*Rejected:* relying on the expressions being too small to be covered (it
contradicts the earlier judgement) and rewriting them in another form (the
last digits of ARI would change, and the rewrite would still need
explaining).

## 2026-10-06 Arguments before data; the standard update by default (#71)

Every fitting function checks all its arguments before it touches the data
(the maps build their unit coordinates first, so the default `sigma` needs
no data), `lloyd`'s `update` defaults to the means weighted by the
memberships, and rmcm builds its data by field name. *Why:* a wrong `tol`,
`max_iter` or `epochs` was reported only after centering, k-means++ seeding
or the PCA start, which can take seconds on large data, and the order
differed between methods; three methods per language spelled out the same
weighted-mean update; and building rmcm's data by position would shift the
fields silently if `Data` gained one, the problem #65 removed from the
views. Iterates are bit for bit unchanged in both languages, and tests now
require arguments to be checked first. *Rejected:* checking arguments
inside `start` and `_setup` (they would need every method's options).

## 2026-10-06 JavaScript's public types are tested as npm users see them (#70)

`js/test/typing-api.js`, like `python/tests/typing_api.py`, calls the API as
typed code does, with `@ts-expect-error` on calls that must be rejected;
`npm run check` builds the declarations and checks it against them in
strict mode. *Why:* the published types come from JSDoc, and a broken
comment turns a parameter into `any` without any error, since the sources
are checked without strict mode (removing one `@param` line made the check
fail three ways). Checking the sources instead would not reach what users
import, and without strict mode TypeScript cannot narrow a generator's
`{ value, done }`, as the README does. *Rejected:* strict mode for the
sources (164 errors, nearly all internal parameters without annotations,
which users never see).

## 2026-10-06 The objective comes with the memberships (#69)

`lloyd`'s `assign` returns the memberships with the objective at them, and
a shared `softmin` gives EFCM and SOM-OLP their memberships with the sum of
the rows' soft minima, which is that objective. FCM takes it from the
minimum distance and the normalizing sum it already computes. *Why:* the
history searched the N x K distances and memberships again for each row's
minimum and maximum, about 20% of FCM's and EFCM's time in Python, while
the memberships are the minimizer of the objective for the given
prototypes and their computation already holds those quantities. Asked
whether the history could be switched off when not needed, we measured
switching it off at 1.26x for FCM and 1.29x for EFCM, and taking it from the
memberships at 1.21x and 1.24x, with nothing to switch. `bench/run.py
--compare main --threads 1`: Python FCM 1.32x, EFCM 1.22x, SOM-OLP 1.11x;
JavaScript SOM-OLP 1.25x, EFCM 1.11x, FCM 1.07x. Iterates are bit for bit
unchanged in both languages, and the history moves by at most 3.5e-16
relative. *Rejected:* an option to skip the history (an argument on eight
functions in two languages, meaningless for the four without an objective,
for 5% more than this) and faster row reductions (the second search stays).
Unlike the costs that #65 stopped passing from `assign` to `objective`
through a shared variable for one method, this value is a return value,
the same for every method with an objective.

## 2026-10-06 Fixtures hold exactly what the code produces (#68)

The fixtures are regenerated, and a speed-only change that moves their last
digits commits the regenerated file and says so. *Why:* #60 reordered the
FCM membership and AMI arithmetic, which moved two FCM cases and one AMI
value by at most 2.2e-16; the tests compare to 1e-10, so the file was left
as it was, and from then on regenerating it always showed a diff. The check
that a refactor leaves the results alone, "regenerate and see no diff", no
longer worked: the review of #65 had to rebuild `main` to tell that diff
from its own. *Rejected:* comparing regenerated fixtures with a tolerance
(a second rule beside the tests, and it would hide which cases moved).

## 2026-10-06 JavaScript keeps the shared step for speed (#67)

FCM, EFCM and SOM-OLP in JavaScript stay in stages (all distances, then all
memberships, then the weighted means) rather than one fused loop per point.
*Why:* fusing them was expected to gain 1.5-2x (unmeasured) where
JavaScript is slower than the textbook NumPy code (0.5-0.6x), but it would
give three methods loops of their own in one language only, so the same
method would have two shapes to fix; FCM's column-scaled weights need every
point before the update, so two passes would remain; each method would grow
from a few lines to a few dozen; and JavaScript already meets its target
(within an order of magnitude of Python), with the browser demo's hundreds
to thousands of points fast enough. *Rejected:* the fused loops. If
JavaScript needs speed, the shared parts (`lloyd`, `softmaxRows`,
`weightedMean`) are where to find it, for every method at once.

## 2026-10-06 Typed generators in Python (#66)

The fitting generators are the functions of `cluster.py` and `som.py`;
`ubukit.steps` is a module that lists them, and `__init__.py` makes the
plain functions from them with `stepwise`, as JavaScript's `index.js` does.
A fitting generator takes new rows or `None` by `send`. mypy also checks the
bodies of untyped functions and reports unneeded ignores, and
`tests/typing_api.py` calls the API as typed code does. *Why:* `ubukit.steps`
was a `SimpleNamespace` filled by the decorator, so to type checkers every
generator was `Any` and `ub.steps.fcm(X, "3")` passed, and the generator
type declared that nothing could be sent, so typing them would have
rejected `run.send(X)` as the README writes it; the typed API of #53 had
missed both. The decorator also filled a module-level registry as a side
effect of importing. The plain functions keep `ubukit` as their module, so
pickle and multiprocessing find them as before. *Rejected:* a typed stub
listing the generators beside the namespace (every signature twice).

## 2026-10-06 Each method's step fits it; one way to build a Result (#65)

SOM-OLP has its own step, like k-means and the maps, and the standard step
`lloyd` takes `assign(data, D)` and `update(data, U, V)`. Every view builds
its Result with `fitted`, and a fixed schedule (`tol=None`) converges in the
loop when it completes. *Why:* SOM-OLP's memberships depend on the previous
ones and its objective on its costs rather than the distances, so inside
`lloyd` it needed an argument for the previous memberships that no other
method used and passed its costs from `assign` to `objective` through a
shared variable, a workaround by principle 9; `lloyd` also passed the
iteration to `assign` and `update`, which no method had used since the batch
SOM got its own step. The views spelled the common fields of a Result by
their position, and the map views overrode the loop's `converged`. A review
of the whole library found these, with a block left over from #64 in
JavaScript's k-means step and three exports nothing imported. Every iterate
of every method, with new data sent in, is bit for bit unchanged in both
languages, and `bench/run.py --compare` finds every affected case the same.
*Rejected:* `assign` returning its costs with the memberships (a second
output for one method).

## 2026-10-06 The data are an input of every iteration (#64)

A step is `step(data, V, state, t)` and reads the data from its argument;
the fitting generators accept new rows between iterations (`send` /
`next(X)`). The engine prepares them (centering; `rmcm`'s graph), keeps the
prototypes where they are, keeps the schedule and the random state, and
keeps the step's state only through the method's `keep` (SOM-OLP's
memberships, while the number of rows stays the same: points are known only
by their rows, row i being the same point); any other state is a cache of
the old data and starts again. `max_iter=None` / `maxIter: Infinity`
removes the limit. *Why:* the browser demo needs learning to follow points
that move, and an iterative method is a map from (data, state) to state, of
which fixed data are the special case; the steps had instead captured the
data in closures, which is what kept the data from changing. The change
removed those closures, adds no path, and gives mini-batch learning for free
to the methods that carry only their prototypes (send a different batch each
iteration; not to SOM-OLP, whose memberships belong to the points, as #67
corrected this entry). JavaScript's online SOM now
shuffles from the identity each epoch, as Python's does, so the number of
rows may change. *Rejected:* passing the previous Result back as `init`
(an output carrying the next iteration's state, a new run every frame that
loses the schedule and random state; principle 9), a public state object
with a pure `step` function (exposes the internal state as API), and
swapping the data inside each method (the same logic eight times, twice).

## 2026-10-06 Designs follow the model, not the shortest path (#63)

`DESIGN.md` gains principle 9, and `AGENTS.md` asks agents to choose designs
by it. *Why:* asked to let the browser demo keep learning while its points
move, the agent first proposed passing the previous Result back as `init`,
because the current code could rebuild everything from it each frame. That
would have made an output carry the state of the next iteration, mixed
starting a run with continuing one, and lost the schedule and random state
on every frame; the fitting design makes the data an input of each
iteration. The proposal was chosen for the size of the change, which is the
failure the principle names. *Rejected:* leaving it to review (the same
choice recurs with every feature).

## 2026-10-06 Seeds stay language-specific (#63)

Python keeps NumPy's generator and JavaScript its Mulberry32; equal results
across the languages come from passing the same initial prototypes.
*Why:* the same seed in both would need one generator in both, and the one
that is easy in JavaScript (Mulberry32: 32-bit state, period 2^32) is
clearly weaker than NumPy's PCG64; lowering Python's to it buys only
convenience. *Rejected:* Mulberry32 in Python, and for now a stronger
JavaScript generator such as sfc32 or xoshiro128** (Mulberry32 serves its
uses, seeding, shuffling and TPE draws, far below its period).

## 2026-10-06 The k-means++ start kept; degenerate FCM documented (#62)

The clustering methods keep starting from greedy k-means++ seeds moved to
their cells' means, and `docs/algorithms.md` now explains why, where zero
distances are handled, and when FCM and EFCM collapse to the mean. *Why:*
measured on the benchmark data (16 dimensions, 10 to 20 seeds): with m up to
5, the start (cell means, raw seeds, or seeds plus noise of 1e-6, 1% or 10%
of the data radius) changed neither the objective nor the ARI; with
m = 50, raw seeds and seeds plus small noise stalled after one iteration on
their data points, while cell means converged. Only FCM divides by a
distance; clipping it at the smallest normal number already gives Bezdek's
rule for d = 0, so noise would add a scale parameter for nothing, and EFCM,
RCM, RMCM and the maps never divide by a distance. The collapse to the mean
follows the linearized thresholds m* = 1 / (1 - 2 lambda_max(M)) and
tau* = 2 lambda_max(S) to within 3% (m* = 1.27 for the 100-blob data), and
past them a lower objective went with an ARI of 0.03 to 0.19 against
0.96 to 0.99 below; raw seeds scored high there only because they stalled
on k-means++ seeds. *Rejected:* noise on the seeds (no gain, a scale
parameter, and both languages would need the same Gaussian draws),
relocating empty clusters (none in 80 k-means runs; they need fewer distinct
points than K), and a helper returning m* (new API; the formula is
documented).

## 2026-10-06 Faster default paths without new code paths (#60)

Without Numba, in both languages: FCM memberships as powers of
d_min^2 / d^2 (the same softmax, without logarithms); the FCM, EFCM and
SOM-OLP histories by the closed forms that the objectives take at the
memberships of the same distances (N logarithms instead of N K); the
expected MI from one table of log-factorials (Python also vectorizes the
inner sum); nearest prototypes without the (N, K) matrix (Python in
cache-sized row blocks, JavaScript fused); and Python's `sqdist` doubling the
prototypes instead of the result (exact, so bit for bit the same). The
results match the old code to rounding, and tests now check the histories
against the definitions. Best times on 16 cores (Python with one BLAS
thread), main -> new: Python batch_som 2.0x, ami with 400 distinct cluster
sizes 11.6x (3.69 -> 0.32 s), ami 50 x 40 clusters 1.3x, fcm 1.2x (m = 2) to
1.5x (m = 1.5), som_olp 1.4x, efcm 1.1x; JavaScript ami 3.0x and 6.6x, fcm
1.4x (m = 1.5), efcm 1.2x, batchSom 1.1x, somOlp 1.1x. *Why:* each replaces
code rather than adding a path, and the Numba kernels follow the same table.
The cost: building the table is O(N), so `ami` with two clusters and large
N is slower (N = 1,000,000: Python 44 -> 55 ms, JavaScript 33 -> 53 ms),
where the label encoding already dominates. *Rejected:* choosing between a
table and per-term log-gamma by size (two paths for one sum), the
vectorized sum without a table (no regression, but 1.6x less gain and
slower at 50 x 40 clusters), label sums by `bincount` or sorting (slower than
the sparse product), rmcm memberships by `bincount` (no gain), and ranks by
sorting in trustworthiness (30x at k = 1000 but 2x slower at the usual k =
5-10).

## 2026-10-06 Numba kernels for the online SOM, trustworthiness and AMI (#59)

`som`, `trustworthiness`, `continuity` and `ami` take `engine="numba"`
(`pip install "ubukit[numba]"`; Numba is in the dev group so the checks
always test it). Every method and metric was prototyped as a Numba kernel
and timed on the benchmark data (16 cores; NumPy uses all of them through
BLAS). Kept, because they gain on one core as well as many (AMI on one core
only where it is slow, with many distinct cluster sizes); the final kernels:

| | NumPy s | Numba s (16 threads) | Numba s (1 thread) |
|---|---:|---:|---:|
| online SOM, N = 5000, 10x10, 2 epochs | 0.153 | 0.019 (serial) | 0.020 |
| trustworthiness, N = 3000, k = 10 | 0.288 | 0.012 | 0.093 |
| AMI, 400 distinct cluster sizes, N = 80,200 | 3.64 | 0.148 | 1.24 |
| AMI, 50 x 40 clusters, N = 1,000,000 | 0.189 | 0.074 | 0.197 |

*Why:* these are loops that NumPy cannot vectorize (one sample after
another; ranks by (distance, index); a sum per pair of cluster sizes). The
kernels transcribe the reference loops and give the same results (the SOM
and trustworthiness exactly); parallel kernels split work only by
independent rows, so results do not depend on the thread count. Nothing is
cached on disk (the library does no file access), so each kernel compiles
on its first call in a process, 0.3-1.3 s. *Rejected:* kernels for som_olp
(4.3x on 16 threads, none on one), efcm (2.8x), batch_som (2.8x, slower
on one thread), rcm (2.0x), fcm (1.7x, slower on one thread), k-means
(1.3-1.9x, slower at K = 100), rmcm (1.0x): their work is BLAS products and
elementwise functions that NumPy already runs well, so the gain depends on
the core count and does not pay for a second implementation. ARI spends its
time encoding arbitrary labels (`np.unique`), and TPE 0.4 ms per trial;
neither benefits.

## 2026-10-06 Release files built apart from the tests; PyPI before npm (#56)

`release.yml` runs the tests in one job and builds the packages in another
that installs only the build tools (`build`, and TypeScript for the type
declarations); the npm upload waits for the PyPI upload. *Why:* the build
job also installed the development group (about 20 unpinned packages such
as pytest, scikit-learn, mypy and ruff) and ran the tests in the workspace
it then packed, so one compromised test dependency could have changed the
published files, which would still carry valid provenance. The two uploads
ran in parallel, so a failure in one could leave a version on one registry
only; a version can never be uploaded again, and npm is the upload more
likely to need a fix (its first release uses a token). *Rejected:* pinning
the development group by hash (more maintenance for a job that need not
install it) and keeping the tests out of the release workflow (the tag's
exact commit would go untested on the release platform).

## 2026-10-06 One release checklist; Dependabot batched and delayed (#55)

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

## 2026-10-06 A typed public Python API (#53)

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

## 2026-10-06 One engine shape in both languages (#52)

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

## 2026-10-06 Every iteration's result, on demand (#51)

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
