# Security

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/subukata/ubukit/security/advisories/new),
not in public issues. Only the latest release receives fixes.

UbuKit computes on in-memory numeric arrays and performs no file, network or
process access. Its main risk with untrusted input is resource use: fitting
costs O(N K D) time per iteration and O(N K) memory, `rmcm` builds a
neighborhood graph (memory bounded by `max_edges`; O(N^2 D) time in
JavaScript), `trustworthiness`/`continuity` cost O(N^2 (D + k)) time (a
`Neighborhood` holds 8 N^2 bytes), and
`ami` grows with the number of distinct cluster sizes in the labels (0.3 s
for N = 80,000 with 400 distinct sizes on each side) and holds N + 1
log-factorials. UbuKit sizes no allocation by a length that the input only
claims: it checks the rows and labels given as lists before copying them,
and copies a `{ data, rows, cols }` matrix only as large as its `data`, so
memory follows what is actually given. It sets no size limit of its own:
bound N, K (for maps, the grid's rows x cols), D, k, the
number of clusters and the iteration counts before passing untrusted data.
With `engine="numba"`, `trustworthiness`, `continuity` and `ami` use every
core unless `NUMBA_NUM_THREADS` limits them.
