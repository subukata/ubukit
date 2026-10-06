# Security

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/subukata/ubukit/security/advisories/new),
not in public issues. Only the latest release receives fixes.

UbuKit computes on in-memory numeric arrays and performs no file, network or
process access. Its main risk with untrusted input is resource use: fitting
costs O(N K D) time per iteration and O(N K) memory, `rmcm` builds a
neighborhood graph (memory bounded by `max_edges`; O(N^2 D) time in
JavaScript), `trustworthiness`/`continuity` cost O(N^2 (D + k)) time, and
`ami` grows with the number of distinct cluster sizes in the labels (0.3 s
for N = 80,000 with 400 distinct sizes on each side) and holds N + 1
log-factorials. UbuKit checks the shape of its input before copying it, so
memory follows the data actually given, but it sets no size limit of its
own: bound N, K, D, k, the number of clusters and the iteration counts before
passing untrusted data.
With `engine="numba"`, `trustworthiness`, `continuity` and `ami` use every
core unless `NUMBA_NUM_THREADS` limits them.
