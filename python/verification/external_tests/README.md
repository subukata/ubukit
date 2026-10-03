> The harness imports implementation modules only under `ubukit._impl`; ownership guards require the installed `ubukit` tree. Frozen numerical fixtures, tolerances and oracle bytes remain unchanged. Current runtime hashes are in `python/SOURCE_MANIFEST.json` from the repository root. Historical results below do not qualify a newly built alpha artifact.

# Installed external-metrics regression gate

This gate exercises the two frozen external-metrics modules and the three lazy `ubukit` aliases in the installed `ubukit` distribution, version `0.1.0a2`. It never imports an implementation from the checkout. The canonical modules must be owned by the selected virtualenv's installed distribution RECORD and their SHA-256 hashes must match that RECORD before and after execution.

## Run

From the repository root after installing the selected fresh environment:

```sh
python3.12 python/verification/external_tests/run_installed.py --python "$PWD/.venv/bin/python" --label repository
```

For a separately qualified Numba-enabled environment also pass `--with-numba`.
See `docs/getting-started.md` for the base installation and
`docs/contributing.md` for repository verification guidance. These paths are
relative to the repository root.

The runner always uses isolated `python -I -B`, an empty working directory, no `PYTHONPATH`, one BLAS/OpenMP/Numba thread, disabled third-party pytest plugins, and a per-environment Numba cache. It installs nothing. Output goes under `results/<label>/` with environment versions, numerical counts/errors, portability results, new facade/route JUnit results, full logs and import provenance.

The copied standalone `test_external_metrics.py` and `test_portability.py` scripts are intentionally excluded from direct pytest collection. Each runs in its own clean process through the runner. This preserves the portability test's dependency blockers without contaminating other tests.

## Numerical and API coverage

- Complete frozen validation panel: every pair of partitions for `N=0..5`; exact ARI versus independent combinatorial calculations and sklearn; independent AMI combinatorial expectation; all four averages; retained singular sklearn behavior
- Original deterministic random, Zipf-imbalanced, correlated and high-cluster-count fixtures for `N=2,3,7,17,100,1000,10000`
- Negative/integer-extreme/uint64/Python-bigint/string/bool/finite-float labels, noncontiguous arrays, near-singletons, permutations, symmetry and relabeling invariance
- All original invalid-label/option checks and four synthetic large-count overflow fixtures, without allocating billions of labels
- Optional NumPy/Numba parity over the complete original panel whenever Numba is installed
- Original isolated portability regression blocks sklearn, SciPy and Numba: ordinary NumPy scoring remains available; needed compatibility fallback and explicit nontrivial Numba execution report ImportError; input mutation is re-evaluated
- Three direct lazy facade aliases retain function identity and signatures; plain `import ubukit` imports no numerical runtime; resolving metric aliases does not import sklearn/SciPy/Numba
- All four averages through facade entrypoints, non-mutating strided inputs, joint-score contingency sharing within a call, and no cross-call input cache
- Observed calls to sklearn for singleton, near-singleton and high-cluster-density cases return the exact sklearn value; a controlled expectation mock exercises the near-zero-denominator fallback route; ordinary fixtures do not use fallback

## Exact adaptations and limits

The two numerical implementation files are not copied, edited or patched by this harness. It only imports their wheel-installed versions. Frozen test fixture sizes, seeds, numerical tolerances, and non-optional comparisons remain unchanged. The copied numerical script adds a dependency availability condition around optional Numba parity calls, preserving the entire NumPy/sklearn/independent panel in base mode, and directs report output into the selected results directory. The copied portability script changes only report destination and support imports. `tools/provenance/VERIFICATION_SNAPSHOT.json` at the repository root records the current verification bytes.

No external dataset, benchmark, speed comparison, million-row timing case, or performance certification is run. Runtime elapsed seconds in logs are test bookkeeping only. The `N=10,000` cases are original correctness fixtures and remain included. Missing Numba omits only backend-parity work; the optional environments execute it. No scikit-fuzzy dependency is needed for this external-metrics gate.

The large-count overflow tests check synthetic count arithmetic; they are not billion-element allocation tests. Floating AMI is exact-definition compatibility under exercised inputs, not exact real arithmetic or a universal sklearn roundoff guarantee. The explicit fallback delegates to the installed sklearn version, so singular rounding-dependent outputs are validated against that version.

## Historical performance provenance

The earlier private verification record (not bundled here) records the prior `final-quiet` evidence without rerunning it. Both `FINAL_REPORT.md` and `benchmark_final.json` identify Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, sklearn 1.8.0 and Numba 0.67.0. The frozen `requirements-tested.txt` instead lists Numba 0.68.0. That file is not evidence that the actual final-quiet run used 0.68.0.

The prior final-quiet runtime SHA-256 values match the supplied modules exactly:

- `external_metrics.py`: `66aa7d48cdba96a87afbb8596d51c41d2eeafd585384432171bb09036ff64c5c`
- `_external_metrics_numba.py`: `48b5616fef1d771bdde65b3a8926f8083476dca5b3792f654a631d0ae30e88ec`

Historical timing observations remain distinct from these new installed-package correctness gates.

## Historical pre-sync verification

All four installed environments passed. Each completed the original 2,960 independent ARI, 11,708 independent AMI, 3,053 sklearn ARI, 12,080 regular sklearn AMI, 132 singular, 1,116 invariance, 17 invalid-input/option and 4 synthetic-overflow checks. Each optional environment also completed 12,212 NumPy/Numba parity checks. The added facade/route suite passed 27 cases per base environment and 31 per optional environment with no skips.

Across all four runs, maximum independent AMI error was 6.7244634323661605e-15 and maximum regular sklearn AMI difference was 8.995501826064567e-13. Optional backend parity maximum was 4.85722573273506e-16. ARI comparisons were exact.

The sklearn 1.7.2 facade panels each emitted 16 expected near-singleton warnings because the number of unique labels exceeded half the sample count. The sklearn 1.8.0 facade panels emitted none. The original numerical script retains its existing UserWarning filter. These are historical pre-sync results. New local results are written under results/<label>/; aggregate machine logs are not committed.
