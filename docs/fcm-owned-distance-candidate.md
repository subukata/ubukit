# FCM owned-distance NumPy candidate

Status: draft candidate, not a release qualification.

## Change and scope

The NumPy distance branch squares its newly allocated broadcast difference in
place, then performs the same axis-2 sum. This removes the second full
N-block x K x D multiplication temporary without changing multiplication or
reduction order. Caller arrays are not the destination of the in-place write.

The runtime delta is exactly +2/-1 in
`python/src/ubukit/_impl/fcm/core.py`. The explicit `numpy` backend and
`fit_fcm_numpy` use this branch. Default `auto` still selects SciPy.
The 1D, SciPy, BLAS, Numba, robust-range and final-objective paths are unchanged.
FCM's finalizer still requires SciPy; this is a Numba-free improvement, not a
SciPy-free claim.

Source identities (SHA-256):
- Qualified alpha3 core: `7c5a7a8c55ae1b28b179d65b1d15d1ef5a7ee70be667148dd4a85eb9eda1c817`
- Previously validated candidate core: `dab38acab30281fbb5487e94fca248bdd939f6cc558c5ebc00f85f6dba4a32df`
- Qualified alpha3 source commit: `c13eaf628218fd0db70d1850f2a861671f27e73c`
- This draft starts from main `6828121d78fac3fb25be7c6d499eec73344c2520`,
  which retains that core and includes the subsequent exact-alpha CI selection.

## Existing local evidence

The saved isolated-source integration completed 81 required checks:
one fresh public smoke, 28 unittest methods and 52 direct FCM parameter cases.
Seven optional Numba checks were unavailable. This was not an installed-wheel
or official pytest-collection qualification.

The preceding bounded benchmark passed all 14 cells in two fresh processes.
It used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, one thread, explicit shared
initial membership, m=2, max_iter=6 and tol=0 for the large target cells.
Each route had three warmups, then seven alternating paired rounds with two
timed samples per route per round. Timing covered the full public fit,
including normalization, thread policy, iterations, final returned-pair
objective and result construction. Baseline and candidate outputs passed the
frozen exactness, ownership and regression checks.

| Fixture (N / D / K) | Process | Baseline seconds | Candidate seconds | Time reduction |
| --- | --- | --- | --- | --- |
| 4096 / 48 / 24 | 1 | 0.073406184 | 0.047557968 | 35.21% |
| 4096 / 48 / 24 | 2 | 0.078536797 | 0.046034608 | 41.38% |
| 2048 / 96 / 16 | 1 | 0.050506392 | 0.024143985 | 52.20% |
| 2048 / 96 / 16 | 2 | 0.050050264 | 0.024984704 | 50.08% |

Controls included unchanged SciPy, 1D, tiny NumPy, one-iteration and robust m=33
cases. The 35–52% range is specific to these two measured fixtures and this
environment. It is not a claim that all backends or machines improve. No
measured peak-RSS claim is made.

The full raw inputs, timings, source and validation records remain in the
private campaign evidence; large archives are deliberately absent from this PR.

## Pending before merge or release

- Run the newly added focused regression file. It covers C/F/strided/read-only
  inputs, caller ownership, exact distance bits, repeated calls, the unchanged
  1D output buffer, and full NumPy fits against the previous expression.
  The new file itself has not yet been executed.
- Review the changed runtime hash and update the candidate source manifest
  through the repository's normal verification procedure. Existing alpha3
  manifests and verification snapshots are intentionally not re-blessed here.
- Build wheel and sdist and qualify clean consumers, including Numba-absent
  and Numba-present environments.
- Run required installed-artifact and cross-platform checks for the exact
  candidate after the manual free-budget gate is satisfied.

This draft does not change the version, CI candidate selection or workflow.
The sole workflow is manual `workflow_dispatch`; no Actions run was requested
by this change.
