# Final evidence index

All values are new restart-v2 measurements on the recorded Linux/9-CPU host.
No earlier lost-run timings are presented as current results. Source version
0.2.0a2 adds explicit PreparedSOM reuse to the measured0.2.0a1 kernels.

| Workload | Reference | Accelerated | Scope |
|---|---:|---:|---|
| k-means20k×8,K16 | scratch NumPy134.323ms; sklearn13.642ms |7.303ms |7warm, samewindow, final labels/inertia included |
| k-means10k×128,K64 | scratch NumPy3219.156ms; sklearn35.461ms |29.648ms |7warm, samewindow, explicit vector backend/finalizer |
| jointT/C10k×64,k15 | sklearn twice9.51834s |1.14259s |3warm, actual public sqrt_numba; exact scores/penalties |
| SOM70k×784,M256 | actual original175.72445s |13.62669s |3warm, complete initialization+fit; both36iterations |
| SOM3parameter fits | fresh public fits40.00818s |33.08895s |3warm workloads; new snapshot+fullSVD charged each workload |

The NumPy k-means baseline uses direct feature-ordered distances, B256 and
bincount updates. It is a reproducible scratch baseline, not an assertion that
all NumPy implementations take this time. Actual public finalization is inside
every k-means fit. Empty-cluster/stop-rule/numerical contracts still differ from
sklearn in general; these fixtures were nonempty and matched all checks.

Primary data:
- `evidence/kmeans_scratch_matched/summary.json` and `comparison.json`
- `evidence/public_metrics/summary.json`
- `som_candidate/results/fullfit70k/` and `FULL_FIT_REPORT.md`
- `som_candidate/results/prepared_sweep70k/` and `PREPARED_SVD_REPORT.md`

Additional controls:
- `evidence/public_metrics_digits/summary.json`: all1797 bundled Digits rows,
  k=[5,15], exact scores/penalties; strictNumba193.406ms beats sqrtNumba201.493ms
  in this case. sklearn runs twice per k, fourcalls total,547.467ms.
- `evidence/kmeans_thread_sensitivity/comparison.json`:3warm per1/2/4/9thread
  budget. At1thread, low-D accelerated17.43ms is slower than sklearn14.35ms;
  high-D111.09ms and110.41ms are close. Treat this as a diagnostic control,
  not a replacement for the7warm primary pair. Some MADs are substantial.
- The sensitivity control inherited a hardcoded `7` in the older driver's
  protocol prose. Its actual `repeats=3` and all three-element sample arrays
  are authoritative; raw records are retained and the correction is explicit.

First calls/runtime setup are reported separately from warm trials. Timed
outputs were validated afterward, and inputs/source hashes were checked.
The SOM comparison uses atol=rtol=1e-8, not bitwise equality to the original;
PreparedSOM's three-fit outputs were bitwise equal to fresh public fits.

Distribution evidence:
- `evidence/distribution_a2/`: final fresh build/install,33 matching source
  hashes,14 discovered no-JIT tests with12passes/2optional skips, API smokes
- `evidence/final_a2_integrated_tests.txt`: all14methods pass with optional
  runtime available
- `evidence/final_a2_source_manifest.json`:31 previous numeric modules unchanged;
  one new module and version/lazy-export changes
- `evidence/measured_public_a1/` and `evidence/distribution/`: exact earlier
  measured source and its independently tested wheel retained for reproduction

No universal fastest backend, OS portability, dependency-version grid or
production reliability is claimed. Prepared reuse retains419.787MiB of snapshot
and factors in the70k case; the scratch policy is not a totalRSS cap. The reuse
benchmark used the isolated candidate plus frozen public kernel; subsequent a2
integration changed imports/docs/description metadata and passed exact tiny
reuse tests and fresh-wheel checks.
