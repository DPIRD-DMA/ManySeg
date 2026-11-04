#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:?set RECIPE}
OUT=${OUT_ROOT:-.}/experiments/E005_fusion_sen1floods11/runs
SET='{"epochs": 60, "batch_size": 16}'
run() { echo "=== $*"; uv run manyseg-bench train --recipe $RECIPE --set "$SET" --out "$OUT" \
          --evaluate-test "$@" || echo "FAILED $*"; }
for seed in 0 1 2 3 4; do
  run --provider manyseg --dataset sen1floods11_s1 --seeds $seed --tag s1
  run --provider manyseg --dataset sen1floods11_s2 --seeds $seed --tag s2
  run --provider smp --dataset sen1floods11 --seeds $seed --tag smp-early
  run --provider manyseg --dataset sen1floods11 --seeds $seed --tag ms-early
  run --provider manyseg --dataset sen1floods11 --input-mode dict --seeds $seed --tag ms-mid-concat \
      --options '{"fusion": {"type": "intermediate", "op": "concat"}}'
  run --provider manyseg --dataset sen1floods11 --input-mode dict --seeds $seed --tag ms-mid-gated \
      --options '{"fusion": {"type": "intermediate", "op": "gated"}}'
  run --provider manyseg --dataset sen1floods11 --input-mode dict --seeds $seed --tag ms-late \
      --options '{"fusion": {"type": "late", "op": "concat"}}'
done
echo "ALL DONE"
