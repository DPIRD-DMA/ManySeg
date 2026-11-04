#!/usr/bin/env bash
# E012: unet vs unet+input_cross vs fastai_unet (all out_stride 1). STAGE=train | profile.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
STAGE=${STAGE:-train}
OUT=${OUT_ROOT:-.}/experiments/E012_unet_input_cross/runs
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments instead of running (Setonix).
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
weights() { case $1 in resnet34) echo tv_in1k ;; pvt_v2_b2) echo in1k ;; esac; }
set_for() { case $1 in
  oxford_pet) echo '{"epochs": 20}' ;;
  sen1floods11) echo '{"epochs": 60, "batch_size": 16}' ;;
esac; }
arm() {  # -> "arch|options|tag"
  case $1 in
    unet) echo 'unet|{"decoder_kwargs": {"out_stride": 1}}|e012-unet' ;;
    cross) echo 'unet|{"decoder_kwargs": {"out_stride": 1, "input_cross": true}}|e012-unet-cross' ;;
    fastai) echo 'fastai_unet|{"decoder_kwargs": {"out_stride": 1}}|e012-fastai_unet' ;;
  esac; }
if [ "$STAGE" = profile ]; then
  for enc in resnet34 pvt_v2_b2; do
    IFS='|' read -r arch opts tag <<< "$(arm cross)"
    mb profile --provider manyseg --encoder "$enc" --weights none --arch "$arch" --options "$opts" \
      --size 256,256 --in-channels 3 --num-classes 3 --batch-sizes 1,8 --train-batch-size 8 \
      --precision bf16 --device cuda --out "$OUT/profile" --tag "$tag" || echo "FAILED profile"
    mb profile --provider manyseg --encoder "$enc" --weights none --arch "$arch" --options "$opts" \
      --size 512,512 --in-channels 15 --num-classes 2 --batch-sizes 1,8 --train-batch-size 8 \
      --precision bf16 --device cuda --out "$OUT/profile" --tag "$tag" || echo "FAILED profile"
  done
  exit 0
fi
for seed in 0 1 2; do
  for setting in "sen1floods11 resnet34" "sen1floods11 pvt_v2_b2" "oxford_pet resnet34"; do
    read -r ds enc <<< "$setting"
    for a in unet cross fastai; do
      IFS='|' read -r arch opts tag <<< "$(arm $a)"
      mb train --provider manyseg --dataset "$ds" --encoder "$enc" --weights "$(weights "$enc")" \
        --arch "$arch" --options "$opts" --recipe "$RECIPE" --set "$(set_for "$ds")" \
        --seeds "$seed" --out "$OUT" --tag "$tag" --evaluate-test || echo "FAILED $tag $ds $enc $seed"
    done
  done
done
echo "ALL DONE"
