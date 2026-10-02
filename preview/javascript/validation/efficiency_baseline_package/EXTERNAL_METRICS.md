# JavaScript external clustering metrics

Zero additional runtime dependencies. Native ESM for Node >=20 and modern
browsers supporting BigInt, Map and typed arrays. Import from the package root
or `ubukit-js/external-metrics` (or that module's URL in a browser).

```js
import { adjustedRandScore, adjustedMutualInfoScore, adjustedScores }
  from 'ubukit-js/external-metrics';
const truth = [0, 0, 1, 1];
const predicted = ['a', 'b', 'a', 'b'];
adjustedRandScore(truth, predicted); // -0.5
adjustedMutualInfoScore(truth, predicted); // approximately -0.5
adjustedScores(truth, predicted); // { ari: -0.5, ami: approximately -0.5 }
adjustedMutualInfoScore(truth, predicted, { averageMethod: 'geometric' });
```

Snake-case aliases `adjusted_rand_score`, `adjusted_mutual_info_score`, and
`adjusted_scores` point to the same functions. Options still use JavaScript
camelCase. The existing `ubukit-js/metrics` neighborhood scheduler is unchanged.

## Input and output contract

Both inputs must be equal-length Arrays or numeric TypedArrays. Each individual
input must contain only strings or only safe integer Numbers. The two inputs
may use different label types. Arbitrary integer labels, negatives, Unicode
strings and noncontiguous labels are supported; `-0` and `0` denote one label.
Fractional/nonfinite/unsafe Numbers, BigInts, booleans, objects, holes, mixed
string/integer labels in one input, multidimensional arrays and DataViews are
rejected. Label meaning/order is irrelevant; inputs are never modified.

Each scalar function returns a Number. `adjustedScores` returns `{ari, ami}`
and shares encoding and one contingency table within that call. There is no
cross-call label/data cache. Every input is validated even in degenerate cases.

AMI defaults to arithmetic mean of the two entropies, matching scikit-learn's
default definition. `averageMethod` also accepts `geometric`, `min`, and `max`.
Empty/empty, singleton/singleton and equivalent partitions return 1. A constant
partition versus a nonconstant partition returns 0. All-singletons versus a
nonidentical nonconstant partition returns exactly 0 for arithmetic, geometric
and max. For optional min normalization that case is mathematically 0/0 and
throws `ExternalMetricDomainError` with code `AMI_SINGULAR_NORMALIZATION`.
The joint API also throws if its AMI component is unsupported; call ARI alone
when only ARI is needed.

## Numerical and performance policy

ARI integer sums and final cross-products are exact, using Number arithmetic
only under an exact-integer bound and BigInt otherwise. Its final quotient is
rounded to binary64. It does not use overflowing 32-bit pair products.

AMI evaluates the permutation-model hypergeometric expectation with a full
support, mode-centered probability recurrence and compensated summation.
Repeated marginal sizes are grouped per call. It uses the equivalent direct
conditional-entropy form `(C_expected - C_observed)/(C_expected + gap)`, avoiding
cancellation between nearly equal MI, EMI and entropy terms. There is no
sampling, tail threshold or log-gamma approximation. Extremely small probability
weights can underflow naturally in binary64; this is not arbitrary precision.

This is semantic/definition compatibility, not bit-for-bit reproduction of
scikit-learn rounding. In singleton-heavy cases scikit-learn 1.8 can return
rounding artifacts far from the exact definition. We intentionally return the
stable value and reject the undefined optional-min case rather than emulate an
arbitrary epsilon-clamped 0/0. No universal absolute-error guarantee is claimed;
see the accompanying measured reference and independent-oracle reports.

AMI supports at most `2**26` samples so all integer products used for support
modes and margin keys remain exactly representable. ARI permits input lengths
through `2**32 - 1`, subject to actual runtime memory. The AMI sample limit has
code `AMI_SAMPLE_LIMIT`.

`maxExpectedTerms` defaults to 10,000,000. It is a positive safe integer counting
the support terms in grouped, nonzero conditional-entropy expectations. Work
beyond this limit throws with code `AMI_WORK_LIMIT`; explicitly raise it when
appropriate. This limit is not a time or total memory guarantee. Input encoding
is O(N), dense/sparse contingency storage O(N+K_true+K_pred), and expectation
cost depends on the unique marginal-size support lengths. Unlike balanced
labels, many distinct large margins can be expensive. Large synchronous calls
belong in a Web Worker or Node worker. No worker/scheduler integration is added.
