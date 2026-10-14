#!/usr/bin/env bash
# E002: ManySeg vs SMP vs fastai U-Net, resnet34 (identical tv_in1k weights), Oxford Pet 256².
# Stage A: identical "standard" recipe. Stage B: identical "fastai_like" recipe.
set -u
export MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
OUT=experiments/E002_provider_comparison_pet/runs
for recipe in standard fastai_like; do
  for seed in 0 1 2; do
    for p in manyseg smp fastai; do
      echo "=== recipe=$recipe seed=$seed provider=$p"
      uv run manyseg-bench train --provider $p --recipe $recipe --seeds $seed \
        --set '{"epochs": 20}' --out $OUT --evaluate-test || echo "FAILED $p $recipe $seed"
    done
  done
done
echo "ALL DONE"
