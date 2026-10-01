# Fresh true full-fit SOM comparison

## Result

On all 70,000 official MNIST images, the frozen public portable API completed a
full converged fit in **13.626693 s median**, versus **175.724454 s** for the pinned
original class: **12.8956× end-to-end speedup** in this run.

Both initializers and kernels were allowed nine CPUs. Every timer included fresh
initialization, allocation/materialization, all iterations and returned-state
construction. Both methods stopped at 36 iterations under tol=1e-4, max_iters=1000.
Each had one cold call followed by three complete warmed calls. No repetition
was dropped, truncated or replaced with an earlier experiment's number.

| Method | Warm samples (s) | Median (s) | MAD (s) |
|---|---|---:|---:|
| Pinned original SOMOLP.fit | 175.724454, 176.970961, 173.772006 | 175.724454 | 1.246507 |
| Public ThreadPool + same-SVD low-rank initial scoring | 14.078700, 13.626693, 12.762274 | 13.626693 | 0.452007 |

The portable method still performs the original full SVD. It does not use a
randomized or truncated replacement SVD. The low-rank identity changes only
initial probability scoring, and the kernel retains the original exact-real
updates and lagged W/V return contract. Floating arithmetic is reordered, so
this is not a bitwise-equivalence claim against the original.

## Actual measured calls

The original class came directly from the pinned upstream source file:
`somolp.py`, commit `4361175b776987d65c348d0132b31d43505e1069`, SHA256
`7761f76770fa846f106ea031141bb953cda9d5967dc6b88d169649ee6cc2057d`.
A subclass wrapped only `_init_pca` to set the recorded thread count and record
its time. All original fit equations, stopping behavior and state timing were
unchanged. Its warmed initialization median was 7.873806 s.

The optimized timer called the actual frozen public interface:

```python
portable_accel.fit_som_olp(
    X, R, gamma=2.767, lam=2.439, max_iters=1000, tol=1e-4,
    backend='threadpool', initializer='svd_lowrank',
    policy=ExecutionPolicy(threads=9, block_rows=256,
                           max_scratch_bytes=64 * 2**20))
```

Its initializer was not separately instrumented; its full-fit total is the
measured result. Public input checks, policy activation, initialization and
kernel overhead all remained inside the timer.

## Validation

The first timed original fit supplied the saved convergence reference. No
extra untimed long oracle was run. Cold and all warmed candidate outputs passed
`np.allclose(rtol=1e-8, atol=1e-8)` for W, P, V and full history, with matching
iteration counts. Maximum absolute differences over all portable fits:

- W: 1.22056e-11
- P: 1.07070e-10
- V: 6.04527e-11
- Objective history: 1.30385e-8

Both methods' repeated outputs were bitwise stable within their own fixed
configuration. Inputs were unchanged. The 32-file public source freeze and the
candidate/driver source hashes were checked before and after measurement and
remained unchanged.

## Memory and thread scope

Full-process peak RSS was 2608.9 MiB original and 2484.8 MiB portable. These peaks
include SVD, loaded input/reference state and validation allocations. They are
not isolated algorithm-workspace measurements or a promise for another process.

The public 64 MiB policy is a **named scratch allowance, not an RSS cap**. It
covers private/merged numerators and denominators plus concurrent worker B×M
buffers. At N=70k,D=784,M=256,T=9,B=256, those named buffers total about 20.80 MB:
private numerators 14.45 MB, private denominators 0.018 MB, reduction buffers 1.608 MB,
and concurrent score buffers 4.719 MB. Outputs, full centered X, SVD storage,
guard/fallback temporaries and BLAS internals are excluded from that named cap.
The unchanged threaded implementation scopes BLAS=1 around its executor; nine
row workers provide outer parallelism with phase barriers and fixed reductions.

A separate explicit research wrapper, `threaded_policy.py`, conservatively also
budgets guard/gather block temporaries. Its 32 MiB setting chooses B131 on this
shape; its 64 MiB setting retains B256. That stricter wrapper was tiny-tested but
was **not** substituted for the public API in this measured comparison.

## Reproduction and evidence

Data source, exact archive hash, train-then-test ordering, /255 scaling, grid
construction/hash and parameters are recorded in `mnist_configuration.json`
and `evidence/mnist70k_fixture_metadata.json`. Initial states are computed fresh
inside every full-fit call; the kernel fixture's stored W0/P0 are not reused.

Use `full_fit_worker.py` first in original mode, then portable mode, with
threads=9, init-threads=9, tol=1e-4, max-iters=1000, repeats=3, and the same
reference path. The original first call writes that reference. The driver saves
partial scalar records after each completed fit and enforces its given budget.
Large local NPZ fixtures/reference arrays are reproducible and excluded from
source checkpoints.

Complete records:
- `results/fullfit70k/original_public_comparison.json`
- `results/fullfit70k/portable_public_comparison.json`
- `evidence/public_fullfit_freeze.json`
- `evidence/mnist70k_convergence_reference_metadata.json`

These are new measurements from the restarted challenge. The optional rebuilt
native control has kernel-only measurements, not an additional true public
full-fit result in this pair.
