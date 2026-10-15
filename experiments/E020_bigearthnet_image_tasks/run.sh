#!/usr/bin/env bash
# E020: BigEarthNet v2 image-level tasks: what ManySeg adds over a timm classifier (README.md).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E020_bigearthnet_image_tasks/runs
EPOCHS=${EPOCHS:?set from the smoke timing in README}
SEEDS=${SEEDS:-0,1,2}
SET="{\"epochs\": $EPOCHS, \"batch_size\": 64}"
G='"band_groups": {"s2_10m": {"bands": [1, 2, 3, 7]}, "s2_20m": {"bands": [4, 5, 6, 8, 10, 11], "factor": 2}, "s2_60m": {"bands": [0, 9], "factor": 6}}, "band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}, "s2_60m": {"backbone": "shared:s2_10m"}}, "fusion": {"type": "intermediate", "op": "concat"}'
FUSE='"fusion": {"type": "intermediate", "op": "concat"}'
run() {  # arm provider dataset arch input_mode options
  uv run manyseg-bench train --provider "$2" --dataset "$3" --encoder resnet50 \
    --weights tv_in1k --arch "$4" --input-mode "$5" --options "$6" --recipe fastai_like \
    --set "$SET" --seeds "$SEEDS" --out "$OUT" --evaluate-test --tag "e020-$1" \
    || { echo "FAILED $1"; fails=$((fails + 1)); }
}
fails=0
for arm in ${ARMS:-timm-s2 s2 groups s1s2-stack s1s2-fusion meta meta-shuffled seg timm-frac frac}; do
  case $arm in
    timm-s2)       run "$arm" timm bigearthnet none stack '{}' ;;
    s2)            run "$arm" manyseg bigearthnet none stack '{}' ;;
    groups)        run "$arm" manyseg bigearthnet none stack "{$G}" ;;
    s1s2-stack)    run "$arm" manyseg bigearthnet_s1s2 none stack '{}' ;;
    s1s2-fusion)   run "$arm" manyseg bigearthnet_s1s2 none dict "{$FUSE}" ;;
    meta)          run "$arm" manyseg bigearthnet_meta none dict '{}' ;;
    meta-shuffled) run "$arm" manyseg bigearthnet_meta_shuffled none dict '{}' ;;
    seg)           run "$arm" manyseg bigearthnet_seg unet stack '{}' ;;
    timm-frac)     run "$arm" timm bigearthnet_frac none stack '{}' ;;
    frac)          run "$arm" manyseg bigearthnet_frac none stack '{}' ;;
  esac
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
