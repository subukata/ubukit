# Traditional SOM integration: Python dev4 / JavaScript dev5

## Added APIs and semantics

Both languages now offer online `som` and true `som_batch`, separate from
SOM-OLP. The default map is planar rectangular 16×16 with x-fast unit order,
and input features can have arbitrary dimension within the documented limits.
Online commits one sample in input order; batch freezes the epoch's BMUs and
commits weighted means simultaneously. Batch has no learning rate. Both support
explicit/sample/PCA initialization, finite nonincreasing schedules, cancellation,
resume and owned snapshots. No mandatory runtime dependency was added.

Python adds seven lazy facade exports, for 38 total, and one implementation
module. Its 58 pre-existing numerical files are unchanged; two lazy facades
change. JavaScript adds one runtime module and six root exports while retaining
seven package entrypoints; its registry now has nine algorithm/metric names.
The session, validation, root-export and shared PCA-helper plumbing changes are
additive. Existing ARI/AMI implementations and frozen metric evidence remain
unchanged.

- [Python API and numerical contract](python/staging/SOM.md)
- [JavaScript API, projection and scheduling contract](javascript/package/SOM.md)
- [Current integration checks](VERIFICATION.md)
- [Runnable reproduction](REPRODUCE.md)

## Verification boundaries

The focused Python gate contains 77 traditional SOM cases plus eight existing
facade/ownership tests. It is run from each fresh wheel/sdist installation with
isolated Python; all 61 runtime files must match the source manifest and be
owned by the installed distribution. The JS full installed-tarball suite includes
43 SOM-focused cases, prior algorithms/metrics/optimizer regressions, lifecycle
coverage, exports and all 26 runtime hashes. The final integration replay is
recorded separately in VERIFICATION.md.

The shared ten-case independent scalar fixture checks every step with explicit
initial prototypes: Python tolerance 3e-13 and JavaScript tolerance 2e-12. This
bounded fixture does not imply bit-identical initialization or all trajectories.
Lowest-index exact computed ties are stable; near ties, degenerate PCA bases
and different summation order can diverge over repeated steps. Python's extreme
BMU-distance repair is broader than JavaScript's representable-distance contract.

Earlier standalone final-candidate review additionally exercised 96 cases / 610
single-update comparisons, 18 initialization probes and 16 JS lifecycle cases.
Those are prior review evidence, not new checks claimed for this integration.
Actual browser execution, other OS/architecture combinations, declared minimum
dependency versions and a final-dev4 Numba matrix remain unverified here.

## Preserved bounded measurements

No benchmark was rerun or rewritten for repository integration. These are final
standalone-candidate observations on the same Linux machine, not universal speed
bounds or matched online-versus-batch quality comparisons.

Python final dev4, one numerical thread:

- N=200, D=8, 8×8 units, one online epoch: about 9.1 ms median; independent scalar
  correctness reference about 26.2 ms; maximum center difference about 1.1e-15
- Same input, one batch epoch: about 0.41 ms median versus 28.2 ms for the scalar
  oracle; maximum center difference about 2.2e-16
- N=2,000, D=32, 16×16 units: online 4,000 sample updates about 0.413 s; batch
  20 complete epochs about 0.139 s. These budgets perform different work
- [Python measurement JSON](python/benchmarks/results.json) contains repetitions
  and environment; [script](python/benchmarks/measure_som.py) includes the oracle

JavaScript final dev5, Node 24, one warmup and three measured sessions, including
initialization and final projection, with explicit initial prototypes:

| Input | Online one pass median | Batch two epochs median |
|---|---:|---:|
| 1,000×3 | 48.4 ms | 11.3 ms |
| 1,000×20 | 90.0 ms | 18.9 ms |
| 256×784 | 691.3 ms | 261.6 ms |

[JS measurement JSON](javascript/reports/som-bounded-benchmark.json) retains
ranges and primary-array estimates. Online and batch totals are not equal-work
or equal-quality comparisons. Cold exact-arithmetic repairs, row/chunk size,
worker overhead and uncontrolled machine load can affect latency. Soft budgets
are not hard realtime guarantees, and memory estimates exclude total runtime RSS.

The earlier [ARI/AMI report](javascript/REPORT.md) deliberately retains its
dev4 / Python dev3 revision labels and original numerical/timing values. Public
package name, project license, registry publication, public visibility and
hosting/CI deployment are separate owner decisions.
