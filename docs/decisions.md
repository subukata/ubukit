# Decisions

Why UbuKit is the way it is, newest first. Each entry gives the decision, the
reason and what was rejected; the pull requests it names hold the
measurements and the review history. Add an entry when a pull request makes
or changes a design decision.

## 2026-10-18 Unit coordinates centered from the first unit; norms scaled (#121)

SOM-OLP subtracts the first unit before the mean of the unit coordinates, a
start made with `pca_scale` passes the whole scale check of a given start
(one `check_scale`), and JavaScript's principal-axis norms scale the entries
by the largest. *Why:* coincident units far from the origin got large costs
from the mean's rounding, a tiny `pca_scale` passed the upper bound alone,
and JavaScript lost the axes of data at 1e80 or 1e-100, whose squares
overflow or underflow. *Rejected:* refusing coincident or far units (they
are valid), scaling the data before the PCA (a second copy of X), and
`Math.hypot` over spread arrays (the stack limit, #109).

## 2026-10-17 pip installs wheels only (#119)

Both workflows set `PIP_ONLY_BINARY=:all:`, so pip never builds a dependency
from source, as npm's `--ignore-scripts` keeps install scripts from running.
*Why:* principle 8 promised it, but only npm enforced it: building a package
that has no wheel runs its build backend, in the test jobs and in the
release build's isolated environment. It costs nothing today: CI built
nothing from source but UbuKit itself. *Rejected:* narrowing principle 8 to
npm, and hash-pinned requirement files (the maintenance #56 rejected).

## 2026-10-17 Values made inside keep the numerical contract (#116)

What the library makes from valid arguments keeps the contract of given
values: SOM-OLP's latent costs come from centered unit coordinates, a PCA
start passes the scale check, and `softmin` raises an error naming the
temperature when its objective overflows. *Why:* far units, `pca_scale =
1e200` and a `tau` or `lam` of 1e308 gave NaN or `-inf` with "tol".
*Rejected:* a `gamma` bound covering the offset's rounding (centering
removes the cause), bounds on `pca_scale`, `tau` and `lam` before the data
(they depend on it), and a non-finite check in the loop (#77).

## 2026-10-17 SOM-OLP's gamma bounded; rcm's radius as Python has it (#115)

`gamma` times the squared extent of the units must be at most 1e300, the
bound of the data's squared norms, checked before the data; JavaScript's
`rcm` takes the radius at a ratio of 0 as the larger term, the exact value
of (1 + 0)^(1/p). *Why:* both gave NaN memberships reported as "tol"
(principle 3). *Rejected:* refusing a `p` whose reciprocal overflows (Python
gives the limit; only JavaScript's `1 ** Infinity` differs from IEEE `pow`),
scaling the latent costs (a rescue path), and a non-finite check in the loop
(#77).

## 2026-10-16 TPE's settings are read-only (#111)

`n_startup`, `n_candidates` and `gamma` are read-only properties (getters in
JavaScript) of the values the constructor checked. *Why:* as plain
attributes, a value set later skipped the checks: `n_candidates = 0` broke
the next proposal. *Rejected:* names that cannot be read (code reports a
search's settings) and setters that check again (a second place for each
check, for a use nobody has).

## 2026-10-16 The public API reviewed; rmcm's delta by name (#110)

The review that #89 asked for compared every public function of both
languages; the defaults agree. `rmcm` and `apply.rmcm` take `delta` by name,
required, as `som_olp` takes `lam` and `gamma`: the data and what the run is
about (`k`, `centers`, `grid`) go by position, a method's parameters by
name. Kept: `pca_scale` on `som_olp` only (only its softmax depends on the
start), private type aliases (spaces type-check by inference, #104), `k` for
clusters and for neighbors, and Python's positional `n_trials`. *Rejected:*
positional `lam` and `gamma` (they cannot follow the optional `grid`), a
default `delta` (no radius suits every scale), and `delta` kept positional
(the one exception).

## 2026-10-16 JavaScript options must be an object (#106)

Every JavaScript function that takes options destructures `named(options)`
in its body, which passes an object or nothing and throws `TypeError` for
anything else; #86's rest element still catches unknown names. *Why:* a
number has no keys, so destructuring it in the parameter list ran the call
with every default: `minimize(f, space, 20)`, Python's positional
`n_trials`, ran 100 trials. *Rejected:* reading `arguments` (its index
differs per function) and a `typeof` check written out in each function.

## 2026-10-16 Effort by the weight of a change (#102)

`CONTRIBUTING.md` sorts changes into light, medium and heavy, each with what
it needs beyond the checks, and `AGENTS.md` lets agents merge a light pull
request once CI passes. *Why:* every rule had been applied at full strength
to every change, so a heavy change took up to two hours before it was
opened. *Rejected:* dropping the benchmark or decision rules (they caught a
regression and keep the reasons) and a separate instruction file for agents
(a second copy of the rules, #46).

## 2026-10-16 Recipes, run by the tests (#100)

`docs/recipes.md` holds worked uses that users asked for (SOM-OLP's scales
and diagnostics, watching rough c-means converge, held-out data, adding
points to a map, federated rough c-means), and `tests/test_recipes.py` runs
its blocks. *Why:* each is a few lines on the public API, and running them
keeps them true. *Rejected:* the READMEs (their registry pages would grow,
twice), helpers for one-line diagnostics (as #62 decided for m*), and
notebooks (unrun, and their diffs carry outputs).

## 2026-10-16 Python data must hold numbers too (#99)

`as_matrix` refuses values that are not integers or floats with `TypeError`
before converting them, as JavaScript's `matrix` does since 0.1.1 (#84).
*Why:* NumPy read `"1.5"` as 1.5, `True` as 1 and complex numbers as their
real part, so the languages took different data. *Rejected:* converting in
JavaScript too (a string column is almost always a mistake) and refusing
only some kinds (each kept conversion is an exception to explain).

## 2026-10-16 Internal modules with an underscore (#98)

`cluster`, `som`, `metrics` and `tpe` became `_cluster`, `_som`, `_metrics`
and `_tpe`, so the public modules are exactly those without one (`ubukit`,
`ubukit.steps`, `ubukit.apply`, `ubukit.steps.apply`), which a test counts.
*Why:* the old paths looked public, and `from ubukit.cluster import fcm`
gave the generator, not `ubukit.fcm`. *Rejected:* modules under the old
names, or a warning `__getattr__` (a compatibility layer for paths never
promised, which #90 rules out in 0.x).

## 2026-10-16 Trained prototypes applied to new data (#97)

`apply.<method>(X, centers, ...)` (and `steps.apply`) runs each method's
assignment on new data with the prototypes fixed and given back unchanged,
taking only the model's parameters; each method builds its rule once, for
training and application. One assignment ends the run ("schedule");
SOM-OLP's memberships, which depend on the previous ones, iterate until they
settle (`Steady`). *Why:* users placing new items on a trained map or
needing a test split's memberships had only `init` with `max_iter=1`, which
moved the centers. *Rejected:* an `update=False` flag (it makes `init` carry
the model), `Result.predict` or model objects (a Result holds no
parameters), separate membership functions (each rule twice), and "tol" for
one assignment.

## 2026-10-16 Maps return their unit coordinates (#96)

The maps' Results hold `unit_coordinates`, the (K, Q) coordinates the run
used, not normalized, `centers[j]` being unit j's prototype; the run owns
its copy. *Why:* an app's developer rebuilt the grid by hand to draw the
map. *Rejected:* the names `units`, `grid` and `positions` (each misleads),
a public grid builder (the caller would still rebuild what the run used),
one shared read-only array (neither language can enforce it), and shapes,
edges or normalized coordinates (drawing is out of scope).

## 2026-10-16 A stop reason instead of `converged` (#95)

`stop_reason` (`"tol"`, `"schedule"`, `"limit"`, or `None` while none holds)
replaces `converged`; each method gives `iterate` its stopping rule as a
value, a tolerance with a limit or a schedule, and the loop decides
(principle 1). It replaces #65's rule that a schedule "converges". *Why:*
`converged` meant different things per method, and a missing tolerance stood
for a schedule (principle 9). *Rejected:* both fields (one fact twice),
`"interrupted"` (the field records rules, not the caller's choices),
`"max_iter"` (wrong beside `maxIter`), and a `schedule=True` flag
(contradictory combinations).

## 2026-10-16 Distances computed once for many embeddings (#94)

`Neighborhood(X)` holds the squared distances of X, which `trustworthiness`
and `continuity` take in place of either set of points with the same values;
a list of k gives one value per k from one count of ranks. *Why:* a search
scoring many layouts of one X recomputed its distances in every call (14x in
Python and 45x in JavaScript for 848 points of 1049 features). *Rejected:*
ranks from sorted rows (plain calls 3-4x slower, #60), a function returning
both metrics, a cache inside the metrics (a hidden session), and
`metric="precomputed"` (an array that could also be data).

## 2026-10-07 A small, written compatibility promise (#90)

`DESIGN.md` states what is promised (the public names with their parameters,
defaults and Result fields; the equations), what is not (module paths,
underscored names, pickles across versions, identical bits), and how a
promise changes: at once in 0.x, after a deprecation from 1.0. *Why:*
unstated paths and pickles had become promises, each needing a compatibility
layer. *Rejected:* setting the classes' `__module__` to keep pickles across
moved files (it breaks `inspect.getsource`, and pickles break anyway once a
Result gains a field).

## 2026-10-07 JavaScript rejects unknown options (#86)

Every JavaScript function that takes options collects those it does not take
with a rest element and throws `TypeError`, naming the camelCase spelling of
a snake_case name. *Why:* Python rejects an unknown keyword, but JavaScript
ran `max_iter` (for `maxIter`) with the default and no error. *Rejected:*
relying on the TypeScript declarations (plain JavaScript is not checked), a
list of known names per function (a second copy that can drift), and
snake_case aliases (two spellings of every option).

## 2026-10-06 npm releases are staged and approved on npmjs.com (#83)

`release.yml` runs `npm stage publish`; a version is published only when the
maintainer approves it on npmjs.com with two-factor authentication, and
npm's trusted publisher may only stage. *Why:* a compromised GitHub account
or workflow could otherwise publish; staging puts the npm account's key in
front of every version. *Rejected:* a GitHub review of `npm` too (a
redundant third approval) and installing npm in the release job (registry
code where a token can be minted).

## 2026-10-06 0.1.0 released; npm through its trusted publisher only (#82)

npm, like PyPI, publishes through its trusted publisher; its first upload
needed a short-lived token (npm trusts only an existing package), revoked
right after, and `release.yml` reads no secret. *Why:* a stored token is the
credential an attacker would want. *Rejected:* keeping the token for
convenience. (#83 then adopted the stage-only publisher, which this entry
had wrongly deferred.)

## 2026-10-06 No JavaScript allocation sized by a claimed length (#80)

`ari`, `ami` and `toRows` check their input as `matrix` does (#76): labels
must be arrays or typed arrays (`isList`), and a matrix's data must hold
exactly its rows (`isMatrix`). *Why:* `{"length": 1e9}` as labels asked for
a 4 GB array, and two `{"length": 3}` scored as identical. *Rejected:* size
limits in the library (#76).

## 2026-10-06 The online SOM back to O(K) memory; schedules as a^(1-f) b^f (#79)

The online SOM computes its winner's grid differences per sample, divided by
the width before squaring, and every geometric schedule is a^(1-f) b^f.
*Why:* #77's table of all grid distances took memory in the square of the
units (385 MB at 64 x 64 in Python), and the old form a (b/a)^f over- or
underflowed for far-apart ends. *Rejected:* keeping the table for small maps
(a second path) and the schedule in logarithms (its ends would not be
exact).

## 2026-10-06 Small temperatures and widths reach their limits (#77)

`softmin` subtracts each row's minimum before dividing by the temperature,
and the neighborhoods divide grid differences by the width before squaring,
so a tiny `tau`, `lam` or `sigma` gives its limit: k-means, hard
memberships, the winner alone. *Why:* they gave NaN marked converged, or
`ZeroDivisionError`, inside the contract (principle 3). *Rejected:* lower
bounds on the parameters (arbitrary; the overflow depends on the data's
scale) and a non-finite check in the loop (the symptom, not the cause). (Its
table of grid distances was removed by #79.)

## 2026-10-06 Input checked before it is copied; no size limit (#76)

JavaScript's `matrix` checks every row before allocating the copy, and
`runAsync` refuses a `budgetMs` that is not finite and >= 0. *Why:*
`[{"length": 1e9}]` asked for an 8 GB array before any check, and an
infinite budget never let an abort take effect. *Rejected:* a size limit in
the library (any fixed limit blocks research use; services set their own, as
`SECURITY.md` says).

## 2026-10-06 Releases only from main, with fixed tags (#75, #78)

`release.yml` stops unless the tag's commit is on `main`; two tag rulesets
let only the maintainer create `v*` tags and nobody move or delete them
(two, because a ruleset's bypass covers all its rules); the environments
allow no administrator bypass. *Why:* any tagged commit could be published
after one approval and a tag could be re-pointed, while a published version
can never be replaced. *Rejected:* building from a branch (trusted
publishing is set up for tags) and an environment limited to `main` (a tag
run has no branch).

## 2026-10-06 One required check, "All checks" (#74)

CI ends with a job that fails when any other failed or was cancelled, and
the `main` ruleset requires only it. *Why:* the job names carry the Python
and Node versions, so each yearly change would have left pull requests
waiting for checks that no longer report. *Rejected:* requiring the jobs by
name (the yearly edit) and a third-party action for the summary.

## 2026-10-06 CI on every pull request, on Ubuntu, once public (#73)

CI runs its Ubuntu jobs on every pull request, Windows and macOS by hand
once each before a release, and nothing while the repository is private.
*Why:* standard runners cost nothing in a public repository; #50's reason,
paid minutes, still holds if it turns private. *Rejected:* all three systems
on every pull request (they rarely differ here and are the slowest) and CI
by hand only.

## 2026-10-06 scikit-learn's notice for three expressions in ARI and AMI (#72)

`THIRD_PARTY_NOTICES.txt`, in the repository and both packages, keeps
scikit-learn's BSD-3-Clause notice for three expressions in ARI and AMI,
marked in comments; the packages declare `MIT AND BSD-3-Clause`. *Why:* the
rebuild kept the expressions that the old code had recorded as adapted, but
not the notice. *Rejected:* calling them too small to be covered (against
the earlier judgement) and rewriting them (ARI's last digits would change).

## 2026-10-06 Arguments before data; the standard update by default (#71)

Every fitting function checks its arguments before any work on the data
(#107 completed it for `k`, `init` and `seed`), `lloyd`'s `update` defaults
to the weighted means, and rmcm builds its data by field name. *Why:* a
wrong argument was reported only after seconds of centering, seeding or PCA,
and three methods per language spelled out the same update. *Rejected:*
checking inside `start` and `_setup` (they would need every method's
options).

## 2026-10-06 JavaScript's public types are tested as npm users see them (#70)

`js/test/typing-api.js`, like `python/tests/typing_api.py`, calls the API as
typed code does, with `@ts-expect-error` on calls that must fail, and `npm
run check` checks it in strict mode against the built declarations. *Why:*
the types come from JSDoc, and a broken comment silently turns a parameter
into `any`. *Rejected:* strict mode for the sources (164 errors, nearly all
in internal code that users never see).

## 2026-10-06 The objective comes with the memberships (#69)

`lloyd`'s `assign` returns the memberships with the objective at them: EFCM
and SOM-OLP take it from `softmin`'s row minima and sums, FCM from its
normalizing sum. *Why:* the history searched the N x K distances again,
about 20% of FCM's and EFCM's time; with identical iterates Python FCM
became 1.32x and EFCM 1.22x faster. *Rejected:* an option to skip the
history (an argument on eight functions, for 5% more).

## 2026-10-06 Fixtures hold exactly what the code produces (#68)

The fixtures are regenerated with every change, and a speed-only change that
moves their last digits commits the regenerated file and says so. *Why:*
left stale within the tests' tolerance, the file no longer showed by its
diff which cases a change moved. *Rejected:* comparing regenerated fixtures
with a tolerance (a second rule that hides which cases moved).

## 2026-10-06 JavaScript keeps the shared step for speed (#67)

FCM, EFCM and SOM-OLP in JavaScript stay in stages (distances, memberships,
weighted means) rather than one fused loop per point. *Why:* fusing would
give three methods loops of their own in one language, FCM would still need
two passes, and JavaScript already meets its target; if it needs speed, the
shared parts are where to find it. *Rejected:* the fused loops.

## 2026-10-06 Typed generators in Python (#66)

The fitting generators are the functions of `_cluster` and `_som`;
`ubukit.steps` lists them, and `stepwise` makes the plain functions, as
JavaScript's `index.js` does. *Why:* a namespace filled by a decorator typed
every generator as `Any` and declared that nothing could be sent to it.
*Rejected:* a typed stub beside the namespace (every signature twice).

## 2026-10-06 Each method's step fits it; one way to build a Result (#65)

SOM-OLP has its own step, `lloyd` takes `assign(data, D)` and `update(data,
U, V)`, and every view builds its Result with `fitted`. *Why:* inside
`lloyd`, SOM-OLP needed an argument no other method used and passed its
costs through a shared variable (principle 9), and views spelled the
Result's fields by position. *Rejected:* `assign` returning SOM-OLP's costs
too (a second output for one method). (Its rule that a schedule "converges"
in the loop was replaced by #95's stop reasons.)

## 2026-10-06 The data are an input of every iteration (#64)

A step is `step(data, V, state, t)`; the generators accept new rows between
iterations, and the prototypes, the schedule and the random state carry
over, the step's state only through the method's `keep` (SOM-OLP's
memberships, for the same points). *Why:* the browser demo needs learning to
follow moving points; fixed data are the special case. *Rejected:* passing
the previous Result back as `init` (an output carrying the next iteration's
state; principle 9), a public state object (internal state as API), and
swapping the data inside each method (the same logic everywhere).

## 2026-10-06 Designs follow the model, not the shortest path (#63)

`DESIGN.md` gains principle 9, and `AGENTS.md` asks agents to choose designs
by it. *Why:* asked to let the browser demo keep learning while its points
move, an agent first proposed passing the previous Result back as `init`,
chosen for the size of the change. *Rejected:* leaving it to review (the
same choice recurs with every feature).

## 2026-10-06 Seeds stay language-specific (#63)

Python keeps NumPy's generator and JavaScript its Mulberry32; equal results
across the languages come from passing the same initial prototypes. *Why:*
one generator in both would lower Python's to the weaker one for convenience
only. *Rejected:* Mulberry32 in Python, and for now a stronger JavaScript
generator (Mulberry32 serves seeding, shuffling and TPE draws, far below its
period).

## 2026-10-06 The k-means++ start kept; degenerate FCM documented (#62)

The clusterings keep starting from greedy k-means++ seeds moved to their
cells' means, and `docs/algorithms.md` explains why, where zero distances
are handled, and when FCM and EFCM collapse to the mean. *Why:* other starts
changed neither objective nor ARI up to m = 5, while raw seeds stalled on
their points at m = 50; the collapse follows the linearized thresholds m*
and tau* to within 3%. *Rejected:* noise on the seeds (no gain, and a scale
parameter), relocating empty clusters (none in 80 runs), and a helper
returning m* (one documented formula).

## 2026-10-06 Faster default paths without new code paths (#60)

Without Numba, in both languages: FCM memberships as powers of distance
ratios, closed-form objectives at the memberships, the expected MI from one
table of log-factorials, nearest prototypes without the N x K matrix, and
the prototypes doubled instead of the product. *Why:* each replaces code
rather than adding a path (AMI with many cluster sizes 11.6x faster, the
others 1.1-2x). *Rejected:* choosing between a table and per-term log-gamma
by size (two paths for one sum), ranks by sorting in trustworthiness (2x
slower at the usual k), and variants that measured no gain.

## 2026-10-06 Numba kernels for the online SOM, trustworthiness and AMI (#59)

`som`, `trustworthiness`, `continuity` and `ami` take `engine="numba"`. Each
kernel transcribes its NumPy reference loop and gives its results; parallel
kernels split the work only by independent rows, so results do not depend on
the thread count; nothing is cached on disk. *Why:* these loops do not
vectorize, and the kernels gain on one core as well as many. *Rejected:*
kernels for the other methods, whose work is BLAS products that NumPy runs
well, so their gain depended on the core count (SOM-OLP 4.3x on 16 threads,
none on one).

## 2026-10-06 Release files built apart from the tests; PyPI before npm (#56)

`release.yml` tests in one job and builds in another that installs only the
build tools, and npm waits for PyPI. *Why:* a compromised test dependency in
the build job could have changed the published files, which would still
carry valid provenance, and parallel uploads could leave a version on one
registry only. *Rejected:* hash-pinning the development group (maintenance
for a job that need not install it) and no tests in the release workflow
(the tag's commit would go untested there).

## 2026-10-06 One release checklist; Dependabot batched and delayed (#55)

The release steps, the one-time setup included, are one checklist in
`CONTRIBUTING.md`, and Dependabot proposes one pull request per ecosystem a
month, for versions at least a week old. *Why:* the one-time steps lived in
comments, a release mistake is permanent, and fewer, older updates are
easier to check and less likely to carry a compromised release. *Rejected:*
removing Dependabot (refreshing the pins by hand is easy to skip) and npm
without provenance while private. (Its reason that CI no longer checked
these pull requests ended with #73.)

## 2026-10-06 A typed public Python API (#53)

Every public function and class annotates its parameters and results, and
mypy checks the library. *Why:* the package declared `Typing :: Typed` with
28% of the parameters annotated. *Rejected:* dropping the classifier instead
(it gives up help from editors).

## 2026-10-06 One engine shape in both languages (#52)

Python's `iterate` is a generator taking the method's `view`, like
JavaScript's, and each fitting function is written once as a generator.
*Why:* the engines had drifted apart, and plotting SOM-OLP's learning in
Python needed the real iterates. *Rejected:* a `callback` argument on every
function (it duplicates the generators) and generators in JavaScript only.

## 2026-10-06 Every iteration's result, on demand (#51)

The generators yield `{ iteration, result }`, where `result()` builds, with
the function that builds the final Result, the Result the run would return
had it stopped there. *Why:* the browser demo could show only finished runs,
and restarting from the centers lost SOM-OLP's memberships. *Rejected:*
building every intermediate Result eagerly (O(NK) per iteration), an
`onIteration` callback (it duplicates the generators), and resuming from a
saved state (internal state as API).

## 2026-10-05 CI only by hand while the repository is private (#50)

Replaced by #73 once the repository was public. While it was private, every
run cost paid minutes: a day of pull requests on three systems (#36-#49)
used the month's 2,000 minutes, though the local checks had found every
problem first. So CI ran only when the maintainer started it, on one system
per run.

## 2026-10-05 Hamerly's bounds for k-means (#49)

k-means runs Lloyd's iterations with Hamerly's (2010) bounds, recomputing
every point within 1e-9 of a bound, so the iterates are Lloyd's. *Why:*
1.45-3.2x faster in Python and 1.9-4.5x in JavaScript; scikit-learn's lead
at K = 100 fell from 16x to about 4x. *Rejected after measuring:* the
second-nearest distance by `np.partition` (slower on overlapping clusters),
the half-distance test alone (slower than Lloyd), and Elkan's K bounds per
point (K times the memory).

## 2026-10-05 Greedy k-means++ seeding (#48)

Each seed is the best of 2 + floor(ln K) candidates drawn by squared
distance, as in scikit-learn. *Why:* single draws often put two seeds into
one cluster: k-means' median ARI over seeds rose from 0.87 to 1.0, with
fewer iterations. *Rejected:* k-means|| (built for distributed data) and
restarts (they multiply the whole run).

## 2026-10-05 A benchmark that can tell differences apart (#47)

The benchmark compares two trees in ABBA order and calls a difference only
for separated samples at least 5% apart; it reports spreads, iterations,
memory, BLAS threads, a scikit-learn column, quality over seeds and growth
with N, and fails CI when the languages disagree. *Why:* the first one could
not resolve differences under about 20% and timed scikit-learn without a
warm-up. *Rejected:* timing in CI (shared runners are too noisy).

## 2026-10-05 Documents by role (#46)

`DESIGN.md` says what and why, `CONTRIBUTING.md` how, `AGENTS.md` only what
is specific to agents, and this file the reasons; `CLAUDE.md` includes
`AGENTS.md`. *Why:* the same rules lived in two files, and the reasons only
in pull requests. *Rejected:* a file per agent tool (the rules twice again).

## 2026-10-05 JavaScript distances stay direct differences (#45)

Plain loops replace TypedArray callbacks on the hot paths. *Rejected after
measuring:* the Gram identity (no gain in V8, and less accurate) and
four-accumulator loops (at most 1.14x at D = 64, slower at small D); the
remaining gap to Python is scalar loops against BLAS.

## 2026-10-05 Windowed expected mutual information (#44)

AMI sums the hypergeometric expectation only within sqrt(35 min(a, b)) of
its mean; by Hoeffding's inequality the omitted mass is below 2 exp(-70), so
the value equals the full sum to rounding, 10-12x faster at N = 1e6.
*Rejected:* the full sum, for an exactness that rounding already erases.

## 2026-10-05 One loop, many steps (#43)

The engine owns the control (stopping, limits, history, progress) and
methods supply a step: `lloyd` for most, an epoch for the online SOM. *Why:*
exact accelerations such as Hamerly's bounds and compiled kernels replace
the step, not the loop. *Rejected:* accelerated methods owning their loops
(principle 1) and only a distance hook (too narrow).

## 2026-10-05 Benchmark, fixture format and change rules (#42)

`bench/run.py` defines each case once and times and scores it in both
languages, and the fixtures have one case per line, so a diff names the
cases a change moved. *Why:* "justify with a measurement" had no
measurement, and the one-line fixture file made diffs unreadable. (Its rule
that a speed-only change leaves the fixtures untouched was revised by #68.)

## 2026-10-05 Numba as an optional, explicit kernel (#42)

A Numba kernel may replace a hot step if it is an installed extra, chosen
per call (`engine="numba"`), reproduces the reference iterates and shows a
large measured gain; #59 added the first. *Why:* sequential, branching steps
do not vectorize. *Rejected:* using Numba whenever it is installed (results
would depend on the environment) and a Numba backend for everything (a
second implementation of every algorithm).

## 2026-10-05 PCA initialization (#40, #41)

Python eigendecomposes the smaller of XᵀX and XXᵀ, JavaScript uses
Rayleigh-Ritz on a Krylov basis of at most 64 vectors (exact for D <= 64),
and each axis is oriented by its first clearly nonzero component. *Why:* the
D x D eigendecomposition took 5.9 s at D = 4000 in Python and 25 s at D =
784 in JavaScript, and orienting by the largest component mirrored the map
on standardized 2-D data. *Rejected:* subspace iteration (fragile near ties)
and Householder with QL in JavaScript (much longer).

## 2026-10-05 Input scale contract (#40, #41)

Rows must have norms below 1e150 and, unless all rows are equal, some
feature must span at least 1e-150. *Why:* data at 1e200 gave NaN memberships
and data at 1e-170 meaningless results, both marked converged. *Rejected:*
rescaling internally (a rescue path, principle 3).

## 2026-10-05 Release path (#38, #39, #40)

Trusted publishing to PyPI and npm from `release.yml`: a job without
credentials tests and builds, and the publishing jobs only upload its files.
Dependabot leaves the Python requirements alone: they are deliberate lower
bounds. *Why:* no third-party code may run where a publishing token can be
minted. *Rejected:* long-lived tokens and the previous hand-made provenance
and archive checks.

## 2026-10-05 Rebuild around one engine (#36)

The library was rewritten as one alternating loop on NumPy and SciPy, and on
plain JavaScript. *Why:* the previous code had accumulated alternative paths
and machinery (subnormal recovery, fast paths for particular versions,
session storage, release pinning) that made it slow to change and hard to
verify. *Rejected:* cleaning it up path by path.

## 2026-10-05 Agreement across languages by equations, not bits (#36)

Python and JavaScript share names, parameters and equations, each with its
own random generator, and the fixtures check that JavaScript reproduces
Python to about 1e-9 given the initialization. *Rejected:* bit-identical
results, which would need a shared generator and no BLAS.
