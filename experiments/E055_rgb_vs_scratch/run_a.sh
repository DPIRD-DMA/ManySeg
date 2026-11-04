#!/usr/bin/env bash
# E055 part A: scratch arms on E024's protocol. Run from a worktree at 9a2d519 (README.md).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
OUT=${OUT_ROOT:?set OUT_ROOT}/experiments/E055_rgb_vs_scratch/runs_a
SEEDS=${SEEDS:-0,1,2}
fails=0
for ds in ${DATASETS:-sen1floods11_s2 bigearthnet pastis}; do
  case $ds in
    sen1floods11_s2) ARCH=unet; EP=${S1F_EPOCHS:-60}; BS=16 ;;
    bigearthnet)     ARCH=none; EP=${BEN_EPOCHS:-12}; BS=64 ;;
    pastis)          ARCH=unet; EP=${PASTIS_EPOCHS:-30}; BS=16 ;;
  esac
  for arm in ${ARMS:-scratch scratch_fair}; do
    case $arm in
      scratch)      SET="{\"epochs\": $EP, \"batch_size\": $BS}" ;;
      scratch_fair) SET="{\"epochs\": $EP, \"batch_size\": $BS, \"encoder_lr_mult\": 1.0, \"freeze_encoder_epochs\": 0}" ;;
    esac
    uv run manyseg-bench train --provider manyseg --encoder resnet50 --weights none --arch "$ARCH" \
      --dataset "$ds" --recipe fastai_like --set "$SET" --seeds "$SEEDS" --out "$OUT" \
      --evaluate-test --tag "${TAG_PREFIX:-e055a}-$ds-$arm" || { echo "FAILED $ds $arm"; fails=$((fails + 1)); }
  done
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
