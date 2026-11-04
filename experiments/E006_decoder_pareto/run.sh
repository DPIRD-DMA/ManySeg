#!/usr/bin/env bash
# E006 decoder Pareto front. STAGE=1: the whole grid, seed 0 (36 runs). STAGE=2: seeds 1 and 2
# for the configs listed in $FRONTIER (written by analyse.py --write-frontier; <= 32 runs).
# STAGE=profile: the timing / memory sweep only (repeatable on an idle GPU; no training).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
STAGE=${STAGE:?set STAGE=1, 2 or profile}
DIR=experiments/E006_decoder_pareto
OUT=${OUT_ROOT:-.}/$DIR/runs
FRONTIER=${FRONTIER:-${OUT_ROOT:-.}/$DIR/frontier.txt}
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments (shell-quoted, one run per line)
# instead of running it; used to submit the same arms as Slurm arrays (scripts/setonix/).
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
DATASETS="oxford_pet sen1floods11"
ENCODERS="resnet34 pvt_v2_b2"
# config = <arch>:<out_stride>; "native" = the decoder's own stride (4), no out_stride option
CONFIGS="unet:1 unet:2 unet:4 fastai_unet:1 fastai_unet:2 fastai_unet:4 segformer:native
fpn:native deeplabv3plus:native"

weights() { case $1 in resnet34) echo tv_in1k ;; pvt_v2_b2) echo in1k ;; esac; }
set_for() { case $1 in
  oxford_pet) echo '{"epochs": 20}' ;;                        # as E003
  sen1floods11) echo '{"epochs": 60, "batch_size": 16}' ;;     # as E005 (stacked S1+S2)
esac; }
opts_for() {  # arch out_stride -> provider options
  if [ "$2" = native ]; then echo '{}'
  else echo "{\"decoder_kwargs\": {\"out_stride\": $2}}"; fi
}
train() {  # dataset encoder arch out_stride seed
  local tag="e006-$3-s$4"
  echo "=== $tag $1 $2 seed=$5"
  mb train --provider manyseg --dataset "$1" --encoder "$2" --weights "$(weights "$2")" \
    --arch "$3" --options "$(opts_for "$3" "$4")" --recipe "$RECIPE" --set "$(set_for "$1")" \
    --seeds "$5" --out "$OUT" --tag "$tag" --evaluate-test || echo "FAILED $tag $1 $2 $5"
}
profile() {  # encoder arch out_stride size channels classes
  local tag="e006-$2-s$3"
  echo "=== profile $tag $1 $4 c=$5"
  mb profile --provider manyseg --encoder "$1" --weights none --arch "$2" \
    --options "$(opts_for "$2" "$3")" --size "$4" --in-channels "$5" --num-classes "$6" \
    --batch-sizes 1,8 --train-batch-size 8 --precision bf16 --device cuda \
    --out "$OUT/profile" --tag "$tag" || echo "FAILED profile $tag $1 $4"
}

case $STAGE in
  1)
    for ds in $DATASETS; do for enc in $ENCODERS; do for c in $CONFIGS; do
      train "$ds" "$enc" "${c%%:*}" "${c##*:}" 0
    done; done; done ;;
  2)
    [ -f "$FRONTIER" ] || { echo "no frontier file $FRONTIER (run analyse.py first)"; exit 2; }
    # fd 3: the runs must not consume the frontier list through stdin
    while read -r ds enc arch os <&3; do
      [ -z "$ds" ] || [ "${ds#\#}" != "$ds" ] && continue
      for seed in 1 2; do train "$ds" "$enc" "$arch" "$os" "$seed"; done
    done 3< "$FRONTIER" ;;
  profile)
    # Pet evaluates 256^2 RGB; Sen1Floods11 evaluates 512^2 tiles of 15 stacked channels.
    for enc in $ENCODERS; do for c in $CONFIGS; do
      profile "$enc" "${c%%:*}" "${c##*:}" 256,256 3 3
      profile "$enc" "${c%%:*}" "${c##*:}" 512,512 15 2
    done; done ;;
  *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
echo "ALL DONE"
