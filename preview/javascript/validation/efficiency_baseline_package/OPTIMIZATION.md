# Lightweight optimization: private candidate

`ubukit-js/optimization` provides a dependency-free optimizer for bounded, flat,
single-objective hyperparameter search. It creates no threads, workers, or network
connections. The same exports are available from the package root.

## Quick start

```js
import { optimize, floatRange, intRange, categorical } from 'ubukit-js/optimization';
const result = optimize(
  p => (Math.log10(p.rate) + 2) ** 2 + (p.depth - 6) ** 2,
  {
    rate: floatRange(1e-5, 1, { log: true }),
    depth: intRange(1, 12),
    method: categorical(['a', 'b'])
  },
  { nTrials: 60, seed: 7, direction: 'minimize' }
);
console.log(result.bestParams, result.bestValue, result.history);
```

JSON distribution descriptors are accepted directly:
`{type: 'float' | 'int', low, high, log: false}` and
`{type: 'categorical', choices: [...]}`. Integer bounds are inclusive safe integers;
integer cardinality must also fit a safe integer. Log bounds must be positive.
Fixed ranges are accepted. Categorical arrays must be dense with own indexed
values; sparse arrays, inherited indexes and undefined are rejected. Categories are
unique JSON scalars: null, strings,
booleans, and finite numbers. Integer-valued numeric categories must be safe
integers (absolute value at most 2^53 - 1), including external observations; finite
noninteger values are supported. This limit prevents Python/JS category aliasing.
`true` and `1` differ; `0` and `-0` are the same.
Parameter names must be nonempty strings.

## Ask/tell and external observations

```js
import { TPEOptimizer, floatRange } from 'ubukit-js/optimization';
const search = new TPEOptimizer({ x: floatRange(-3, 3) }, { seed: 7 });
search.addTrial({ x: 1 }, 1); // prior completed experiment; consumes no RNG
const trial = search.ask();
search.tell(trial.id, trial.params.x ** 2);
const failed = search.ask();
search.tell(failed.id, null, { state: 'fail', error: 'evaluation failed' });
console.log(search.result());
```

`ask` returns `{id, params, state: 'running', value: null, error: null}`. `tell`
accepts an outstanding ID from this optimizer exactly once. Completion states are
`complete`, `fail`, and `cancelled`. Only successful finite values train the model.
Failed/cancelled trials require a null value; successful trials cannot have an
error. `addTrial` validates imported observations and permits real repeated
experiments. Returned params, trials, history, and results are owned snapshots.
Underscore-prefixed fields are unsupported implementation details.

Results contain `bestParams`, `bestValue`, `bestTrial`, `bestTrialId`, `history`,
`nAttempted`, `nCompleted`, and `stopReason`. Best values are null before success.
“Best” means best observed, not a certified optimum. Ties prefer the lower ID.

## Options

- `seed`: unsigned 32-bit integer, default 0
- `direction`: `minimize` (default) or `maximize`
- `sampler`: `tpe` (default) or `random`
- `nStartupTrials`: successful startup observations, default 12, minimum 2
- `nCandidates`: proposals scored per modeled request, default 24
- `gamma`: good-observation fraction, default 0.15
- `multivariate`: shared-component mixture, default true; false is independent TPE
- `minBandwidth`: normalized numeric kernel floor, default 0.03
- `weights`: good-observation `ei` weights (default) or `uniform`
- `avoidDuplicates`: default true; inverse alias `allowDuplicates`; do not use both
- `maxDuplicateAttempts`: random fallback retry count, default 64

`optimize` and `optimizeAsync` additionally accept:

- `nTrials`: maximum evaluation attempts, default 100; alias `budget`
- `continueOnError`: false by default; true records exceptions/nonfinite results
  as failures and continues. Failed attempts consume budget
- `onTrial`: detached finished-trial callback; return false to stop
- `signal`: optional AbortSignal

Duplicate filtering includes running, failed, cancelled, and completed trials.
Fallback uses bounded random retries, then enumeration for finite spaces of at
most 100,000 settings. `ask` raises `SearchSpaceExhaustedError` only on proven finite
exhaustion and `ProposalError` on bounded proposal failure. Convenience wrappers
return partial results with `space_exhausted` or `proposal_failed`; other normal
stop reasons are `budget_exhausted` and `callback_stopped`. Low-level snapshots
use `not_started` or `running` for an externally managed search.

All-tied observations use random sampling until there is an objective signal.
Pending trials reserve settings but are not invented observations. Set
`avoidDuplicates: false` for repeated/noisy evaluations; there is no automatic
noise model.

## Async and cancellation

```js
import { optimizeAsync, floatRange } from 'ubukit-js/optimization';
const result = await optimizeAsync(async params => await evaluateModel(params),
  { rate: floatRange(1e-5, 1, { log: true }) }, { nTrials: 30, seed: 3 });
```

Async evaluation is sequential; `onTrial` may also be async. Sync `optimize`
rejects Promise objectives. Cancellation throws `AbortError`. The signal is
checked before asking and after evaluation. In-flight objective code must honor
its own signal for prompt cancellation; the optimizer cannot interrupt synchronous
code or cancel an arbitrary pending Promise. Use low-level `TPEOptimizer` to retain
partial history after exceptional exits.

The runtime module uses standard ECMAScript and can be served/bundled as a browser
ES module. Automated runtime tests here used Node 24.19.0/Linux; other browsers
and Node versions have not been certified.

## Method and numerical contract

TPE splits observations into good/bad groups, fits kernel mixtures, samples the
good model, and maximizes `log(l(params)) - log(g(params))`. Joint mode shares one
observed component across dimensions and multiplies its conditional axis kernels.
It captures some dependencies without a full covariance matrix. Independent mode
mixes each axis separately. Neither mode is a Gaussian process.

Numeric axes normalize to [0, 1], optionally in log space. Truncated Gaussian
bandwidths use neighboring observations, boundaries and the prior midpoint, with
a bandwidth floor of `max(minBandwidth, 1 / (n + 1)^2)`. A positive uniform latent
prior is always included. Categories use a center spike with uniform smoothing.
Integer scoring uses transformed bin masses, not rounded continuous densities;
narrow bins use a centered fourth-order normal-integral expansion to avoid
cancellation. The branch requires standardized width <= 0.01 and
abs(standardized midpoint) * width <= 0.1. Its relative Taylor truncation remainder
is below 4e-12 before floating roundoff. Stable bin widths are used even when
rounded endpoints coincide; truncation normalization still uses the shared CDF.
Log-integer prior bin widths are computed stably.

Mixtures use log-sum-exp. Extreme ranges avoid overflowing span arithmetic. EI
weights scale finite losses before subtraction. The shared normal CDF polynomial
has roughly 7.5e-8 absolute error; inverse-normal sampling uses a rational
approximation. This is a lightweight heuristic, not exact analytical density
integration or a reproduction of every Optuna behavior.

## Reproducibility and limits

Python/JavaScript share Mulberry32, uint32 seeds, UTF-16 parameter ordering, draw
counts, mixture formulas, candidate scoring and tie breaking. Replay requires the
same configuration and order of asks/completions. Cross-platform floating results
are tolerance-based, not promised bit-identical. Different async completion order
can change later proposals.

The search space is flat. Conditional trees, pruning, multibudget scheduling,
distributed storage, worker pools, multiple objectives, constraints and full
covariance modeling are not implemented. Memory grows with stored history;
proposal cost grows with dimensions, observations and candidates. There is no
bounded-history approximation.

This is a private candidate, not a published release. Small synthetic benchmarks
do not establish universal superiority over random search or Optuna or justify a
state-of-the-art claim. Use held-out validation and account for evaluation cost.
