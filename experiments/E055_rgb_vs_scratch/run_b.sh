#!/usr/bin/env bash
# E055 part B: fewer labels on Sen1Floods11 S2, 900 optimiser steps at every size (README.md).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
OUT=${OUT_ROOT:?set OUT_ROOT}/experiments/E055_rgb_vs_scratch/runs_b
SEEDS=${SEEDS:-0,1,2,3,4}
fails=0
for n in ${SIZES:-12 25 63 252}; do
  case $n in  # batch, epochs, frozen epochs (15 steps) and evaluation spacing per size
    12)  BS=8;  EP=900; FR=15; EV=100 ;;
    25)  BS=16; EP=900; FR=15; EV=100 ;;
    63)  BS=16; EP=300; FR=5;  EV=50 ;;
    252) BS=16; EP=60;  FR=1;  EV=10 ;;
  esac
  EP=${EPOCHS_OVERRIDE:-$EP}
  for arm in ${ARMS:-imagenet scratch scratch_fair}; do
    case $arm in
      imagenet)     W=tv_in1k; X="\"freeze_encoder_epochs\": $FR" ;;
      scratch)      W=none;    X="\"freeze_encoder_epochs\": $FR" ;;
      scratch_fair) W=none;    X="\"freeze_encoder_epochs\": 0, \"encoder_lr_mult\": 1.0" ;;
    esac
    SET="{\"epochs\": $EP, \"batch_size\": $BS, \"eval_every\": $EV, $X}"
    uv run manyseg-bench train --provider manyseg --encoder resnet50 --weights "$W" --arch unet \
      --dataset sen1floods11_s2 --train-subset "$n" --recipe fastai_like --set "$SET" \
      --seeds "$SEEDS" --out "$OUT" --evaluate-test --tag "${TAG_PREFIX:-e055b}-n$n-$arm" \
      || { echo "FAILED n$n $arm"; fails=$((fails + 1)); }
  done
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
