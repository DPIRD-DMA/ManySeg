#!/usr/bin/env bash
# E015 (local RTX 4090): 13-band 10 m stack vs native-resolution S2 band groups (Sen1Floods11).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E015_s2_native_resolution/runs
G='"band_groups": {"s2_10m": {"bands": [1, 2, 3, 7], "factor": 1}, "s2_20m": {"bands": [4, 5, 6, 8, 11, 12], "factor": 2}, "s2_60m": {"bands": [0, 9, 10], "factor": 6}}, "fusion": {"type": "intermediate", "op": "concat"}'
SHARED='"band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}, "s2_60m": {"backbone": "shared:s2_10m"}}'
run() { uv run manyseg-bench train --provider manyseg --dataset sen1floods11_s2 --encoder resnet34 \
          --weights tv_in1k --arch unet --recipe "$RECIPE" --set '{"epochs": 60, "batch_size": 16}' \
          --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
for seed in 0 1 2; do
  run --seeds "$seed" --tag e015-stack
  run --seeds "$seed" --tag e015-groups --options "{$G}"
  run --seeds "$seed" --tag e015-groups-shared --options "{$G, $SHARED}"
done
echo "ALL DONE"
