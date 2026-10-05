# Security

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/subukata/ubukit/security/advisories/new),
not in public issues. Only the latest release receives fixes.

UbuKit computes on in-memory numeric arrays and performs no file, network or
process access. Its main risk with untrusted input is resource use: fitting
costs O(N K D) per iteration, `rmcm` builds a neighborhood graph (memory
bounded by `max_edges`; O(N^2 D) time in JavaScript), and
`trustworthiness`/`continuity` cost O(N^2 D) time. Bound N, K, D and
iteration counts before passing untrusted data.
