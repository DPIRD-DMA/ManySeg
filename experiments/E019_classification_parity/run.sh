#!/usr/bin/env bash
# E019: ManySeg image-level classifiers vs plain timm (see README.md).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E019_classification_parity/runs
ENCODERS="resnet50:tv_in1k convnext_tiny:fb_in1k pvt_v2_b0:in1k"
set_for() { case $1 in eurosat) echo '{"epochs": 30, "batch_size": 64}' ;;
  oxford_pet_breeds) echo '{"epochs": 20, "batch_size": 32}' ;; esac; }
for ds in eurosat oxford_pet_breeds; do for e in $ENCODERS; do for p in manyseg timm; do
  enc=${e%%:*} w=${e##*:}
  uv run manyseg-bench train --provider "$p" --dataset "$ds" --encoder "$enc" --weights "$w" \
    --arch none --recipe fastai_like --set "$(set_for "$ds")" --seeds 0,1,2 --out "$OUT" \
    --evaluate-test --tag "e019-$p-$enc" || echo "FAILED $ds $p $enc"
done; done; done
echo "ALL DONE"
