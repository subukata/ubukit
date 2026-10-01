#!/usr/bin/env bash
# Prepared only; requires a coordinator timing slot.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=restart_v2/metrics_candidate/results/digits_tie_control_t9.jsonl
if [[ -e "$out" ]]; then echo "Refusing to overwrite an existing digits control" >&2; exit 2; fi
restart_v2/.venv/bin/python -m restart_v2.metrics_candidate.benchmark \
  --dataset digits --n 1797 --d 64 --k 15 --dtype float64 --threads 9 --repeats 3 \
  --block-rows 256 --methods sklearn numpy_broadcast numba sqrt_numba \
  --output "$out" > "${out%.jsonl}.log" 2>&1
cat "${out%.jsonl}.log"
