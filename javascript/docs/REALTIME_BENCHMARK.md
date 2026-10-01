# Realtime-oriented JavaScript: measured summary

These are single-thread Node CPU observations for this feature slice, not a
browser FPS claim or a latency guarantee. The reusable driver is
`bench/realtime.js`; raw samples, process-memory observations and source hashes
are archived separately from the distributable source.

## Session latency

Each fixture uses an explicit fixed initialization (except the explicitly marked
SOM PCA case), maximum 8 iterations, blockRows=64 and a **2 ms soft** step budget.
Warm measurements have 5 warmups and 21 samples per case. Cold measurements have
9 fresh processes per case; process startup/module imports are excluded.
N/D/K are samples/features/clusters; SOM has Q=2.

| Method (N/D/K) | Full session median ms | Step p50 ms | Step p95 ms | Largest warm step ms | Largest cold step ms | Data-update p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| k-means 4096/4/8 | 3.97 | 0.39 | 0.70 | 4.25 | 7.21 | 0.18 |
| FCM 4096/4/8, m=2 | 9.86 | 0.64 | 1.56 | 10.67 | 17.01 | 0.36 |
| ExRCM 2048/2/6, p=1 | 3.37 | 0.32 | 0.59 | 6.04 | 7.40 | 0.20 |
| RMCM 384/2/4, delta=.3 | 1.98 | 0.33 | 0.46 | 3.31 | 4.68 | 0.10 |
| SOM 2048/4/8 | 10.69 | 1.01 | 2.02 | 7.00 | 7.21 | 0.14 |
| SOM PCA 512/16/8 | 3.78 | 0.34 | 0.97 | 4.27 | 9.85 | 0.10 |

Full session timing includes owned input validation/copy, every committed
iteration checkpoint copy and an explicit snapshot after each completed
iteration. Step timings include the internal checkpoint copy; explicit snapshot
cost is measured separately. Data updates include copy, equality comparison and
validation, but exclude the subsequent numerical fit/graph reconstruction.
Initialization, JIT and GC can overshoot the soft deadline substantially, as the
maxima demonstrate. p95 is an empirical quantile, not a service-level bound.

## Full fit cost and kernel improvement

The same driver compares the new synchronous run with the preceding production
source and with a session using the same initialization/iteration contract.
FCM was 12.74 ms in preceding production, 8.55 ms in the optimized synchronous
run, and 9.86 ms including session/snapshot overhead. SOM was 10.85, 10.23 and
10.69 ms respectively. These ratios apply only to these stated fixtures.

A separate audited 21-sample kernel experiment retained O(NK) FCM normalization
in both implementations. On its 2000×8, K=8, 12-iteration m=2 case, preceding
production/new kernel/simple linear/guarded linear medians were
9.12/6.41/6.35/6.74 ms. General-m and larger FCM, and the larger SOM reference,
were still faster in straightforward reference implementations. Thus the new
kernels improve the existing library on the measured cases; **they are not
uniformly faster than reasonable references**. There is no quadratic straw-man
FCM baseline and no silent fast-math/approximation change. These separate kernel
fixtures are not mixed with session timings or cold-start results.

The kernel retains direct-distance squared terms, exact zero-distance handling,
underflow fallback, generic m/p, original accumulation order, update order,
cancellation/progress events, and output contracts. New block-kernel tests plus
session/Worker/lifecycle tests supplement the historical Python-fixture suite.

## Scope and reproduction

```sh
npm test
node --expose-gc bench/realtime.js > realtime-measurement.json
node --expose-gc bench/realtime.js --baseline=/absolute/path/to/old/src/index.js
```

The optional baseline is another source checkout, not a mandatory dependency.
Run on an otherwise idle machine; set CPU affinity externally if available.
The driver verifies segmented/full results before timing and reports every raw
sample to stdout. It measures warm-state updates separately because they can
require different iterations and do not form a fair cold-fit speedup baseline.
RMCM graph reuse is only timed when X/delta/backend are unchanged; changed X
always pays the exact graph preparation cost. Memory observations after 50
parameter updates are diagnostic, not a hard process-memory ceiling or proof of
zero allocation. Python timing and browser end-to-end timing remain separate.
