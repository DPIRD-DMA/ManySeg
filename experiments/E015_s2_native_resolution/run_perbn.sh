#!/usr/bin/env bash
# E015 amendment 3: shared encoder with per-input BN (library default since 0a4025d).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E015_s2_native_resolution/runs
G='"band_groups": {"s2_10m": {"bands": [1, 2, 3, 7], "factor": 1}, "s2_20m": {"bands": [4, 5, 6, 8, 11, 12], "factor": 2}, "s2_60m": {"bands": [0, 9, 10], "factor": 6}}, "fusion": {"type": "intermediate", "op": "concat"}'
SHARED='"band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}, "s2_60m": {"backbone": "shared:s2_10m"}}'
for seed in 0 1 2; do
  uv run manyseg-bench train --provider manyseg --dataset sen1floods11_s2 --encoder resnet34 \
    --weights tv_in1k --arch unet --recipe fastai_like --seeds "$seed" --out "$OUT" --evaluate-test \
    --set '{"epochs": 60, "batch_size": 16, }' \
    --options "{$G, $SHARED}" --tag e015-groups-shared-perbn || echo "FAILED $seed"
done
echo "ALL DONE"
