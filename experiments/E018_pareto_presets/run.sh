#!/usr/bin/env bash
# E018: encoder x decoder Pareto sweep (local RTX 4090). STAGE=screen (1 seed, all configs) |
# confirm (seeds 1-2 for configs listed in CONFIRM) | profile (compiled bf16 latency, idle GPU).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
STAGE=${STAGE:-screen}
OUT=${OUT_ROOT:-.}/experiments/E018_pareto_presets/runs
ENCODERS="resnet18:tv_in1k resnet34:tv_in1k efficientnet_b0:ra_in1k pvt_v2_b0:in1k"
dec() { case $1 in
  unet) echo 'unet|{}' ;;
  unet_w50) echo 'unet|{"decoder_kwargs": {"channels": [128, 64, 32, 16, 8]}}' ;;
  unet_w25) echo 'unet|{"decoder_kwargs": {"channels": [64, 32, 16, 8, 8]}}' ;;
  segformer) echo 'segformer|{}' ;;
esac; }
set_for() { case $1 in oxford_pet) echo '{"epochs": 20}' ;;
  sen1floods11) echo '{"epochs": 60, "batch_size": 16}' ;; esac; }
train() {  # dataset encoder:weights decoder seed
  local enc=${2%%:*} w=${2##*:}; IFS='|' read -r arch opts <<< "$(dec "$3")"
  uv run manyseg-bench train --provider manyseg --dataset "$1" --encoder "$enc" --weights "$w" \
    --arch "$arch" --options "$opts" --recipe fastai_like --set "$(set_for "$1")" --seeds "$4" \
    --out "$OUT" --evaluate-test --tag "e018-$enc-$3" || echo "FAILED $1 $enc $3 $4"
}
case $STAGE in
  screen)
    for ds in sen1floods11 oxford_pet; do for e in $ENCODERS; do
      for d in unet unet_w50 unet_w25 segformer; do train "$ds" "$e" "$d" 0; done; done; done ;;
  confirm)  # CONFIRM="ds:enc:weights:decoder ..." chosen by the screen rule in README
    for c in ${CONFIRM:?}; do IFS=: read -r ds enc w d <<< "$c"
      for s in 1 2; do train "$ds" "$enc:$w" "$d" "$s"; done; done ;;
  profile)
    for e in $ENCODERS; do for d in unet unet_w50 unet_w25 segformer; do
      enc=${e%%:*}; IFS='|' read -r arch opts <<< "$(dec "$d")"
      uv run manyseg-bench profile --provider manyseg --encoder "$enc" --weights none \
        --arch "$arch" --options "$opts" --size 512,512 --in-channels 15 --num-classes 2 \
        --batch-sizes 1,8 --train-batch-size 8 --precision bf16 --compile --device cuda \
        --out "$OUT/profile" --tag "e018-$enc-$d" || echo "FAILED profile $enc $d"
    done; done ;;
esac
echo "ALL DONE"
