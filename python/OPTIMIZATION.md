# Lightweight hyperparameter optimization

This local preview adds a dependency-free optimizer core to UbuKit. It searches
within user-provided ranges and returns the best **observed** successful setting.
It does not promise the global optimum or uniformly better performance than
Optuna. The existing UbuKit distribution still has its existing dependencies;
the optimization module adds none and also runs with Python's `-I -S` mode.

## Quick use

```python
from ubukit import optimize, float_range, int_range, categorical

space = {
    "learning_rate": float_range(1e-4, 1.0, log=True),
    "n_clusters": int_range(2, 12),
    "initializer": categorical(["sample", "pca"]),
}

def objective(params):
    # Train/evaluate your model here. This example is a stand-in.
    return (params["n_clusters"] - 5) ** 2

result = optimize(objective, space, n_trials=60, seed=42,
                  direction="minimize")
print(result.best_params, result.best_value)
print(result.trials)  # detached Trial snapshots, including failures
```

`budget=60` is an alias for `n_trials=60`; do not specify both. `n_trials=0`
returns an empty result. A failed evaluation consumes a trial. Successful values
must be finite real numbers. Both minimization and maximization are supported.
The optimizer keeps only parameters, scalar values, state, and optional error
text; model training, data, and model saving remain in the user's objective.

The result has `best_params`, `best_value`, `best_trial_id`, `history` (alias
`trials`), `n_attempted`, `n_completed`, and `stop_reason`. The best fields are
`None` if no evaluation succeeded. `result.to_dict()` creates a detached
serialization-friendly dictionary for ordinary JSON-scalar inputs. State is
in memory only; a durable resume format is not part of this preview.

## Search space

- `float_range(low, high, log=False)`: finite closed bounds, or a fixed value
  when both are equal. A log range requires a positive lower bound
- `int_range(low, high, log=False)`: inclusive integers, with unit steps. Bounds
  and the number of distinct integers must fit in the JavaScript safe-integer
  range. Log integer ranges require `low >= 1`
- `categorical([values])`: nonempty, unique finite JSON scalar values. `True`,
  `1`, and `"1"` are distinct; `0` and `-0.0` are the same numeric choice.
  Integer-valued numeric categories, whether supplied as Python integers or
  floats, must lie within `[-(2**53 - 1), 2**53 - 1]`. The same restriction
  applies to imported observations, preventing cross-language category aliases.
  Represent larger categorical identifiers as strings. Continuous numeric
  ranges are not subject to this categorical restriction
- The equivalent JSON dictionaries are accepted directly: types `float`,
  `int`, `categorical`, fields `low`, `high`, `log`, or `choices`

The search space is fixed when an optimizer is created. Conditional parameters,
constraints, multiple objectives, pruning, arbitrary float/int steps,
distributed shared storage, and unbounded domains are intentionally absent.
A single integer has a bin, not a rounded continuous-point density. The log
integer prior is proportional to each transformed bin width, using bins from
`low - 0.5` to `high + 0.5`. It is not Optuna's exact implementation contract.
The range declaration and optimizer use their own copies of scalar choices.

## Ask and tell

```python
from ubukit import TPEOptimizer

search = TPEOptimizer(space, seed=42)
trial = search.ask()
score = objective(trial.params)
search.tell(trial.id, score)
# Or: search.tell(trial.id, state="fail", error="expected evaluation failure")
print(search.result().best_params)
```

`ask()` reserves a setting and returns a detached `Trial(id, params)` in state
`running`. `tell()` finishes an issued, currently running ID exactly once.
It accepts `complete`, `fail`, or `cancelled`. Failed/cancelled records must not
have values. Invalid calls do not mutate the trial. An external successful
observation may be imported with `add_trial(params, value)`; it validates the
complete space and consumes no random draws. Imported repeated observations
are retained as real data, even when generated duplicates are disabled.

Only completed successful trials enter the density model. A failed point is
not silently treated as the worst numerical objective. `optimize()` records
failures then re-raises by default; `continue_on_error=True` continues after
ordinary exceptions or nonfinite results. Interrupts are recorded as cancelled
and propagated. A callback gets each detached finished trial; returning exactly
`False` stops after that trial.

Exact parameter duplicates, including currently running, failed, and cancelled
settings, are avoided by default. Set `avoid_duplicates=False` for intentional
repeat evaluation. `SearchSpaceExhausted` proves a finite space has no unissued
point; `ProposalError` merely means bounded retries did not find a new setting.
The high-level helper stops and reports these as `space_exhausted` or
`proposal_failed`. Close but different floating-point settings are not deduped.

One optimizer instance has one owner. Multiple `ask()` calls can feed external
workers, but the class is not a thread-safe or distributed store. Running
points are reserved but not assigned fantasy/constant-liar values, so nearby
concurrent proposals can be redundant. Completion order influences later
proposals. For reproducibility use sequential evaluation, or explicit
synchronous batches and a fixed update order.

## Algorithm and settings

Default settings are `sampler="tpe"`, `multivariate=True`,
`n_startup_trials=12`, `n_candidates=24`, `gamma=0.15`,
`weights="ei"`, and `min_bandwidth=0.03`.

- Random proposals are used until enough successful observations exist, and
  whenever all completed values are tied
- Observations are sorted by value and trial ID, then split into good and bad
  groups. The good count is `ceil(gamma * n)`, bounded from 1 to `n - 1`
- A multivariate density is a mixture of product kernels. Each mixture
  component corresponds to one observed full setting, and sampling shares
  that component across all dimensions. This can represent dependencies
  without fitting a full covariance matrix
- Numeric kernels are boundary-normalized Gaussian kernels in normalized
  linear/log coordinates. Per-observation bandwidths use neighboring values,
  endpoints and the prior center. The minimum is the larger of
  `min_bandwidth` and `1 / (group_size + 1)^2`
- A uniform prior component prevents zero support. Categorical kernels blend
  the observed category with uniform probabilities, with mixture strength
  `1 / (group_size + 1)`
- Good observations use positive improvement weights relative to the first
  bad score, normalized to mean one. The prior has weight one. Bad
  observations are equally weighted. `weights="uniform"` is an ablation
- Candidates are drawn from the good model and ranked by stable
  `log p_good - log p_bad`. Duplicate candidates are skipped, with a bounded
  random and small-finite-space fallback

`multivariate=False` uses separate mixtures per dimension. It is an explicit
independent-TPE comparison, and can miss interactions. The multivariate option
can also lose on particular problems. `sampler="random"` gives the same API
with random search for a fair local baseline. This is a research-informed
small implementation, not an Optuna algorithm clone or an implementation of
all the latest Bayesian optimization methods.

The normal CDF uses a shared polynomial approximation with about `7.5e-8`
maximum absolute error; inverse-CDF sampling uses a shared rational
approximation. Narrow normal intervals use a centered fourth-order integrated
Taylor expansion, evaluated in log space. Its branch requires standardized
full width `w <= 0.01` and `abs(midpoint) * w <= 0.1`; the relative Taylor
remainder is below `4e-12` before floating-point rounding. Integer-bin widths
are computed independently of endpoint subtraction. Broader intervals still
use the approximate CDF, so the Taylor bound does not imply that all PMFs have
that relative accuracy; very small far-tail probabilities can have substantial
relative error despite tiny absolute error. Truncation normalization also uses the approximate
CDF. No SciPy runtime is needed. Boundary normalization, probability-mass sums, extreme ranges, and
cross-language fixtures are tested. This is not arbitrary-precision numerical
integration. Proposal construction/scoring is roughly O(d n log n + c d n)
and history storage O(d n), with n observations, d dimensions, c candidates.

The Taylor mass is `phi(m) * w * (1 + (m²-1)w²/24 +
(m⁴-6m²+3)w⁴/1920)`. With `M=abs(m)+w/2`, the sixth-derivative bound gives a
relative remainder no larger than `w⁶ * (M⁶+15M⁴+45M²+15) / 322560 *
exp(abs(m)*w+w²/8)`. The branch conditions bound this by `4e-12`. This avoids
subtracting nearly identical approximate CDF values, including near zero.

## Reproducibility and JavaScript parity

Seeds are uint32 integers from 0 to `2**32 - 1`. The optimizer's Mulberry32
stream is independent of model-training RNGs. Parameter names are sorted in
UTF-16 lexicographic order, and categories have type-aware identity. Both
implementations share the formulas, draw ordering, stable ties, and fixtures.

PRNG uint32 values match exactly. Checked fixed-history log densities and
suggestions, and complete mixed-space trajectories agree within numerical
roundoff between tested Python and Node versions. This is not a promise of
bit-identical results on every libm/browser/OS. Floating-point evaluation
values, near-tied candidate scores, different model RNGs, and parallel
completion order can change the trajectory.

## Choosing the objective for clustering or SOM

Choose one common evaluation metric before searching. Internal training losses
can change meaning or scale when cluster count, FCM fuzzifier, SOM penalties,
or grid size change; comparing those losses directly can pick a misleading
winner. Use a task-appropriate shared score, and consider complexity and runtime
budgets. Ground-truth labels, if appropriate, allow external clustering scores.
Without labels there is no universally correct automatic score.

Stochastic models can average several fixed training seeds inside each
objective. The optimizer does not infer observation-noise variance. Recheck
finalists on new seeds/data before relying on a tiny apparent advantage. Count
all underlying model fits in budgets and comparisons, not only top-level
objective calls.

## Validation guidance

Small synthetic comparisons do not establish general superiority over random
search or Optuna. Use held-out validation and count objective-evaluation costs.

References informing design, not a claim to reproduce every implementation:

- [Bergstra et al. 2011, Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
- [Watanabe, updated May 2026, TPE algorithm-component analysis](https://arxiv.org/abs/2304.11127)
- [Optuna 5.0 TPESampler reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Optuna reproducibility FAQ](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)

No third-party implementation source was copied into the optimizer. Its shared
normal-function numerical coefficients are standard mathematical
approximations. Existing UbuKit notices remain unchanged.
