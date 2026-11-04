#!/usr/bin/env bash
# E017: CloudSEN12+ (13 S2 bands) — stack vs resolution band groups through one shared encoder.
# 2 arms x 3 seeds; with EMIT_TASKS, run as one 6-GCD pack (scripts/setonix/manyseg_pack.sbatch).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E017_cloudsen12_band_groups/runs
SET='{"epochs": 20, "batch_size": 16}'
# S2 L1C band order B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12 (checked on the cache: index 10
# has 0.011 mean reflectance = B10 cirrus, index 9 = B9 water vapour).
G='"band_groups": {"s2_10m": {"bands": [1, 2, 3, 7], "factor": 1}, "s2_20m": {"bands": [4, 5, 6, 8, 11, 12], "factor": 2}, "s2_60m": {"bands": [0, 9, 10], "factor": 6}}, "fusion": {"type": "intermediate", "op": "concat"}'
SHARED='"band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}, "s2_60m": {"backbone": "shared:s2_10m"}}'
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
for seed in 0 1 2; do
  mb train --provider manyseg --dataset cloudsen12 --encoder resnet34 --weights tv_in1k \
    --arch unet --recipe "$RECIPE" --set "$SET" --seeds "$seed" --out "$OUT" --evaluate-test \
    --tag e017-stack
  mb train --provider manyseg --dataset cloudsen12 --encoder resnet34 --weights tv_in1k \
    --arch unet --recipe "$RECIPE" --set "$SET" --seeds "$seed" --out "$OUT" --evaluate-test \
    --options "{$G, $SHARED}" --tag e017-groups-shared
done
echo "ALL DONE"
