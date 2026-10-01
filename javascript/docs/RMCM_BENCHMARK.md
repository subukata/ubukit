# Matched Node RMCM observations

These are bounded single-thread JavaScript CPU measurements, not browser or
Python comparisons. Five rotating-order repetitions followed warm-up. Both
backends used the same input, explicit centers, fixed graph definition, and
stopping rule; all compared trajectories had identical labels, iterations and
stop reasons, with center error below 1e-10. Ratios are reference/adjoint medians.

| Shape | Final R | N / D / K | Directed E | Iterations | Full fit ref / adj (ms) | Ratio | Reused fit ref / adj (ms) | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse | True | 1000 / 4 / 8 | 1090 | 12 | 18.185 / 17.706 | 1.03× | 2.596 / 1.575 | 1.65× |
| sparse | False | 1000 / 4 / 8 | 1090 | 12 | 18.423 / 17.819 | 1.03× | 2.549 / 1.476 | 1.73× |
| moderate | True | 1000 / 4 / 8 | 69854 | 12 | 18.761 / 16.916 | 1.11× | 4.210 / 1.587 | 2.65× |
| moderate | False | 1000 / 4 / 8 | 69854 | 12 | 19.322 / 16.864 | 1.15× | 4.283 / 1.441 | 2.97× |
| complete | True | 400 / 4 / 8 | 160000 | 3 | 4.159 / 3.964 | 1.05× | 0.724 / 0.741 | 0.98× |
| complete | False | 400 / 4 / 8 | 160000 | 3 | 3.471 / 3.663 | 0.95× | 0.174 / 0.179 | 0.97× |

Full fit includes input copying, two-pass graph construction, backend preparation,
iterations, and final R when requested. Reused fit excludes preparation but
includes requested final R. Reference constructs R each iteration; adjoint only
constructs it at the end when requested. Both use the common-mean shortcut on a
complete graph. A value below 1 means the adjoint was slower in that comparison.

The moderate graph shows a meaningful reused-fit gain; graph construction makes
end-to-end gains much smaller. The complete case shows no reused-fit speedup.
These small local timings are evidence for these shapes only. Do not extrapolate
them into a universal acceleration factor.

Reproduce with `node bench/rmcm.js`. The full validation archive retains individual
timing samples, runtime metadata, and source hashes; this concise document does
not embed raw machine logs or environment dumps.
