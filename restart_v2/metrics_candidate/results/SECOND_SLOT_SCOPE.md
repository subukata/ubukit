# Second metrics v2 slot

Instrumented final timed-worker completion: 2026-10-01 00:01:07.711153 UTC.
Final comparison-summary completion: 00:01:08.006131 UTC. No further heavy timing ran in this slot. The release notice was sent afterward; the bundled-digits control was skipped to protect the four-minute cap.

## NumPy method/block comparison

Same n=2,000, d=64→2, float64, k=15, seed=20261001 as the first slot. Three warmed repetitions in fresh processes; actual two-sklearn-call baseline in every configuration. All exact scores, available integer penalties, and input byte hashes match across methods, blocks, and thread budgets.

| Threads | Requested rank block | Broadcast median s | Sortsearch median s |
|---:|---:|---:|---:|
| 1 | 32 | 0.280181 | 0.267246 |
| 1 | 256 | 0.242839 | 0.264599 |
| 9 | 32 | 0.267711 | 0.277333 |
| 9 | 256 | 0.266339 | 0.283273 |

Broadcast with requested block256 was the fastest observed NumPy choice for the nine-thread control, although its difference from block32 is small. This is a measured configuration choice, not a universal method/default claim. The public default has not been changed.

## Representative 10,000-row comparison

n=10,000, d=64→2, float64, k=15, nine CPUs/threads, requested rank block256, 32MiB dominant NumPy scratch cap. The broadcast implementation reduces its effective row block as needed to respect that cap. All methods retain full original sklearn Euclidean distance calls and independent original neighbor selection. Three warmed repetitions; no approximation or score tolerance.

| Method | Median s | MAD s | Speedup | Process peak MiB |
|---|---:|---:|---:|---:|
| Two actual sklearn calls | 9.537960 | 0.011368 | 1.00x | 2432.0 |
| NumPy broadcast | 3.337684 | 0.038238 | 2.86x | 938.7 |
| Numba scan | 1.231928 | 0.008106 | 7.74x | 1040.0 |
| Numba rounded-sqrt interval scan | 1.128320 | 0.004657 | 8.45x | 1040.2 |

All candidate scores and integer penalties match exactly. Penalties are `(660247366, 521341460)` and input hashes are identical across all four fresh processes. The sqrt route was about 9.18% faster than the plain scan in this one screen. Runtime/JIT startup is excluded from the warmed medians and retained separately in the raw records. Process RSS includes fixed runtime/JIT overhead and is not an algorithm-only workspace figure.

These are new v2 results on this execution environment, not historical recovery data or cross-OS speed guarantees. No tie-heavy digits timing was completed in this slot. Raw trials, source/input hashes, resource settings, UTC completion metadata and exact checks are retained in JSONL, `.summary.json`, `.run.json`, logs and `second_slot_summary.csv`.
