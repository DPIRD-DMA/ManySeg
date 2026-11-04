#!/usr/bin/env bash
# E015 amendment 2: capacity control (3 encoders, same band split, all groups at 10 m).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
OUT=${OUT_ROOT:-.}/experiments/E015_s2_native_resolution/runs
G='"band_groups": {"s2_10m": {"bands": [1, 2, 3, 7], "factor": 1}, "s2_20m": {"bands": [4, 5, 6, 8, 11, 12], "factor": 1}, "s2_60m": {"bands": [0, 9, 10], "factor": 1}}, "fusion": {"type": "intermediate", "op": "concat"}'
for seed in 0 1 2; do
  uv run manyseg-bench train --provider manyseg --dataset sen1floods11_s2 --encoder resnet34 \
    --weights tv_in1k --arch unet --recipe fastai_like --seeds "$seed" --out "$OUT" --evaluate-test \
    --set '{"epochs": 60, "batch_size": 16}' --options "{$G}" --tag e015-groups-10m \
    || echo "FAILED $seed"
done
echo "ALL DONE"
