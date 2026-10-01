#!/usr/bin/env bash
# Prepared only: run after the coordinator grants an exclusive timing slot.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=restart_v2/.venv/bin/python
TAG=${RUN_TAG:-numpy_comparison}
for t in 1 9; do
  for b in 32 256; do
    out="restart_v2/metrics_candidate/results/${TAG}_2k_t${t}_b${b}.jsonl"
    if [[ -e "$out" ]]; then echo "Refusing to append to existing comparison: $out; set a fresh RUN_TAG" >&2; exit 2; fi
    "$PY" -m restart_v2.metrics_candidate.benchmark --n 2000 --d 64 --k 15 --dtype float64 \
      --threads "$t" --repeats 3 --block-rows "$b" --max-scratch-bytes 33554432 \
      --methods sklearn numpy_broadcast numpy_sortsearch --output "$out" > "${out%.jsonl}.log" 2>&1
    cat "${out%.jsonl}.log"
  done
done
