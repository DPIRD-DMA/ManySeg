#!/usr/bin/env bash
# E063 part C: all arms on CloudSEN12+, one seed of all arms at a time (README.md).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
OUT=${OUT_ROOT:?set OUT_ROOT}/experiments/E063_rgb_vs_scratch/runs_c
EP=${EPOCHS:-20}
EXTRA=${EXTRA:-}  # preflight only: e.g. --subset 64
fails=0
for seed in ${SEEDS:-0 1 2}; do
  for arm in ${ARMS:-imagenet scratch scratch_fair}; do
    case $arm in
      imagenet)     W=tv_in1k; X="" ;;
      scratch)      W=none;    X="" ;;
      scratch_fair) W=none;    X=", \"encoder_lr_mult\": 1.0, \"freeze_encoder_epochs\": 0" ;;
    esac
    # shellcheck disable=SC2086
    uv run manyseg-bench train --provider manyseg --encoder resnet50 --weights "$W" --arch unet \
      --dataset cloudsen12 --recipe fastai_like --set "{\"epochs\": $EP, \"batch_size\": 16$X}" \
      --seeds "$seed" --out "$OUT" --evaluate-test --tag "${TAG_PREFIX:-e055c}-cloudsen12-$arm" $EXTRA \
      || { echo "FAILED seed $seed $arm"; fails=$((fails + 1)); }
  done
done
[ "$fails" -eq 0 ] || { echo "$fails run(s) FAILED"; exit 1; }
echo "ALL DONE"
