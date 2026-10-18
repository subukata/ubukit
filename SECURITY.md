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
log-factorials. The maps add their units' work: the online SOM costs
O(K (D + Q)) per sample, the batch SOM's smoothing holds rows^2 + cols^2
kernel entries, and Python's PCA start a min(N, D)^2 matrix. A list of k
adds O(N k_max) work per entry, and each TPE proposal costs O(trials x
candidates x dimensions), times the options of a categorical dimension.
UbuKit sizes no allocation by a length that the input only claims: it
checks every entry of the rows, labels and lists of k given as arrays,
holes of sparse arrays included, before allocating for them, and copies a
`{ data, rows, cols }` matrix only as large as its `data`, so memory
follows what is actually given. It sets no size limit of its own: bound N,
K (for maps, the grid's rows x cols), D, k and the length of a list of k,
the number of clusters, the iteration counts and, for TPE, the trials,
candidates and space before passing untrusted data.
With `engine="numba"`, `trustworthiness`, `continuity` and `ami` use every
core unless `NUMBA_NUM_THREADS` limits them.

These limits hold for data in ordinary containers. Getters, proxies, NumPy's
array hooks such as `__array__`, and the objective given to `minimize` are
the caller's own code and run as such; and a pickled `Result` or `TPE`, like
any pickle, runs code when loaded, so load only one from a trusted source.
