#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:?set RECIPE}
OUT=${OUT_ROOT:-.}/experiments/E004_channel_adapters/runs
SET='{"epochs": 60, "batch_size": 16, "eval_every": 10}'
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments (shell-quoted, one run per line)
# instead of running it; used to submit the same arms as Slurm arrays (scripts/setonix/).
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
run() { echo "=== $*"; mb train --provider manyseg --dataset sen1floods11_s2 \
          --recipe "$RECIPE" --set "$SET" --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
RGB='"backbone_kwargs": {"rgb_indices": [3, 2, 1]}'
for seed in 0 1 2; do
  run --seeds $seed --tag repeat --options '{"channel_adapter": "repeat"}'
  run --seeds $seed --tag mean --options '{"channel_adapter": "mean"}'
  run --seeds $seed --tag rgb_zero --options "{\"channel_adapter\": \"rgb_plus_zero\", $RGB}"
  run --seeds $seed --tag rgb_random --options "{\"channel_adapter\": \"rgb_plus_random\", $RGB}"
  run --seeds $seed --tag projection --options "{\"channel_adapter\": \"projection\", $RGB}"
  run --seeds $seed --tag scratch --weights none
done
echo "ALL DONE"
