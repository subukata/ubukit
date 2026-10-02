# JavaScript ARI / AMI dev4 verification and measurements

## Result

Implemented `adjustedRandScore`, `adjustedMutualInfoScore`, and `adjustedScores`
at the package root and `ubukit-js/external-metrics`. Python-style name aliases
are included; options use camelCase. Runtime dependencies added: zero. The joint
API shares one encoding and contingency. The new package is private
`ubukit-js@0.1.0-dev.4`; the earlier JS dev3 is retained in repository history
and Python remains at dev3.

## Matched current-environment N=10,000 / K=50 panel

Same labels, seed 20261002, SHA256:
`279feca5928fc800740b903785912ce2195975f2164b7281e379258eb5377042`

| Implementation | ARI ms | AMI ms | Joint ms |
|---|---:|---:|---:|
| javascript_dev4 | 0.293194 | 0.376083 | 0.386349 |
| javascript_ungrouped_baseline | 0.334082 | 8.343403 | 8.193976 |
| python_dev3_numpy | 0.080167 | 0.337750 | 0.326821 |
| python_dev3_numba_warm | same ARI as NumPy | 0.198429 | 0.209706 |
| sklearn_1_8 | 1.453738 | 24.669977 | 26.420195 |

Times are median milliseconds per invocation, from 20 warmups and 21 batches of
five calls. Input construction is excluded; public validation, label encoding
and contingency construction are included. Arithmetic AMI is used everywhere.
Sklearn's joint number is two separate public calls; the UbuKit joint APIs share
one contingency. No timing is reused from the earlier October 1 panel.

- JS joint is 68.38× faster than the sklearn two-call baseline on this case
- JS joint is 21.21× faster than the ungrouped JS comparison on this case
- JS remains 1.18× slower than Python NumPy and 1.84× slower than warm Numba here
- JS joint vs its separate ARI+AMI calls: 0.636061 → 0.386349 ms (1.65×)
- Warm Numba excludes the separately measured first AMI call: 443.437 ms, including compilation/cache loading
- ARI result: 0.62982259365711; AMI: 0.6953675530147322
- JS versus sklearn differences: ARI zero; AMI 1.99e-13

Environment: AMD EPYC 9V74 80-Core Processor, Linux x64; Node v24.19.0; Python
3.12.14, UbuKit Python 0.0.0.dev3, NumPy 2.3.5,
scikit-learn 1.8.0, Numba 0.67.0. OMP, OpenBLAS, MKL and Numba
thread limits were all 1; Numba reported one thread. Python installed-distribution
RECORD ownership and its source hash were verified. Node used one synchronous
JS thread. The measurements ran serially after coordinating with other active
benchmarks. They are machine/workload-specific, not universal speed guarantees.

Raw per-sample times and score values: `reports/matched-node.json` and
`reports/matched-python.json`. Compact dashboard-ready rows:
`reports/matched-summary.json` / `.csv`. Exact labels:
`fixtures/matched-10000-k50.json`.

## Broader performance panel

| Case | JS joint ms | Ungrouped JS joint ms | sklearn two-call ms |
|---|---:|---:|---:|
| balanced-correlated-10000-k10 | 0.366373 | 1.202695 | 7.688411 |
| balanced-correlated-100000-k20 | 2.961810 | 13.052165 | 115.452518 |
| balanced-correlated-10000-k1000 | 0.876343 | 147.887429 | 381.010863 |
| unequal-margins-30000-k20 | 5.018262 | 4.836598 | 22.325348 |
| load_iris | 0.013480 | 0.016425 | 1.089053 |
| load_wine | 0.025569 | 0.019440 | 1.136845 |
| load_breast_cancer | 0.039790 | 0.036825 | 1.246581 |

This separate exploratory panel uses three warmups and nine single-call samples,
so its tiny-input timings are noisier. All JS/Python rows use exactly the same
stored inputs and current environment. The straight JS baseline is a one-pass
Map contingency plus a scalar, full-support hypergeometric expectation per
cluster pair, without marginal-size compression. It is a meaningful numerical
baseline, not quadratic pair-label enumeration. It is faster than the candidate
on some skewed/small cases: grouping and BigInt/compensated bookkeeping have costs.
Repeated marginal sizes give the largest speedup. No blanket JS speedup is claimed.

## Verification

- Installed dev4 package: 890/890 regression tests pass, including inherited dev3 coverage
- Independent audit: 3,043 ARI and 12,040 AMI checks against exact-integer / 80-digit Decimal references
- 12,040 joint checks and 36,120 swap/order/relabel invariance checks
- Independent AMI maximum absolute error: 3.33e-16; ARI errors zero on those references
- 132 expected optional-min singular cases, 74 malformed/work-budget checks, early sample-limit and symmetric work-budget checks pass
- Large synthetic count tests check exact ARI products through N=4,294,967,292 and AMI internals through the 2**26 sample bound; these do not claim enormous public label arrays were allocated
- Extra sklearn 1.8 panel: 174 cases including Iris/Wine/WDBC labels; largest AMI difference 9.74e-11, in a high-K case
- Pure ESM imports without dependency/host globals, isolated VM module execution and a real Node module worker pass
- Actual browser execution is not verified: cloud Chrome blocked the local test URL with `net::ERR_BLOCKED_BY_CLIENT`. The browser harness is included for independent execution

The independently reviewed runtime SHA256 is:
`8a0235400c287216af0e133be8dda72bb2a5eb653ef9a8f657ff9047834d1a37`

## Numerical policy and edge cases

ARI keeps integer combinatorics exact with safe Number sums / BigInt products;
its final quotient is rounded to binary64. AMI uses a mode-centered, normalized,
full-support hypergeometric recurrence, grouped margin sizes and compensated
sums. Its equivalent conditional-entropy normalization avoids cancellation in
high-K partitions. There is no sampling or chosen tail cutoff, but binary64
underflow/rounding still exists and no universal absolute-error bound is claimed.

The default is arithmetic AMI. Empty, singleton and equivalent partitions return
1; constant-vs-nonconstant returns 0. Singleton-vs-other returns exact 0 for
arithmetic/geometric/max. Only optional min normalization in that case is truly
0/0 and throws a documented `AMI_SINGULAR_NORMALIZATION` error. AMI above 2**26
samples is rejected before allocation; the adjustable expectation-work budget
also rejects excessive work explicitly. String labels or safe integer Number
labels are supported, homogeneous within each input; other domains are rejected.

Stable definition compatibility is deliberately distinct from copying sklearn
rounding artifacts. For two different doublet partitions at N=3000, the exact
AMI is −2.2229637041155283e−7; JS differs by approximately 1.6e−16, while sklearn
1.8 returns −2.113112724214236e−6 (error 1.89e−6). For singleton-vs-doublet,
arithmetic AMI is exactly zero but sklearn can return about −3.78e−6. Undefined
optional-min results must not be compared as a normal finite-score parity case.
Full examples are in `audit/sklearn-conditioning.json`.

## Delivery and integration

Production package: `package/`; build the ignored local artifact
`artifacts/ubukit-js-0.1.0-dev.4.tgz` using `REPRODUCE.md`.
Source/API contract: `package/src/external-metrics.js` and
`package/EXTERNAL_METRICS.md`.
The committed source, tests and numerical evidence form the repository snapshot;
no standalone patch or external integration manifest is needed. The aggregate
`../SOURCE_SNAPSHOT.json` and `../VERIFICATION_SNAPSHOT.json` bind the current
runtime and verification bytes. The measured values above retain the original
standalone experiment results. See `../VERIFICATION.md` for the additional clean
repository-source replay and its limitations. Registry publication and deployment
are not part of this private source integration. See `REPRODUCE.md` for commands.
