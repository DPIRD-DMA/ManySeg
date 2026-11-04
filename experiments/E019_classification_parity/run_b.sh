#!/usr/bin/env bash
# E019b: ManySeg image heads with timm's classifier init (normal 0.02, zero bias). README.md.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
OUT=${OUT_ROOT:-.}/experiments/E019_classification_parity/runs
ENCODERS="resnet50:tv_in1k convnext_tiny:fb_in1k pvt_v2_b0:in1k"
set_for() { case $1 in eurosat) echo '{"epochs": 30, "batch_size": 64}' ;;
  oxford_pet_breeds) echo '{"epochs": 20, "batch_size": 32}' ;; esac; }
fails=0
for ds in eurosat oxford_pet_breeds; do for e in $ENCODERS; do
  enc=${e%%:*} w=${e##*:}
  uv run manyseg-bench train --provider manyseg --dataset "$ds" --encoder "$enc" --weights "$w" \
    --arch none --options '{"head": {"init_std": 0.02}}' --recipe fastai_like \
    --set "$(set_for "$ds")" --seeds 0,1,2 --out "$OUT" --evaluate-test \
    --tag "e019-manyseg-init02-$enc" || { echo "FAILED $ds $enc"; fails=$((fails + 1)); }
done; done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
