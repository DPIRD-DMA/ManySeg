#!/usr/bin/env bash
# E013: CloudSEN12+ high. Shared S2-only U-Net baseline + aux fusion (Q1), metadata (Q2) and
# input_cross (Q3) arms. 8 arms per seed: with EMIT_TASKS, each seed is one 8-GCD pack
# (scripts/setonix/manyseg_pack.sbatch ... 8), so a pack stages the cache once.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E013_cloudsen12/runs
SET='{"epochs": 20, "batch_size": 16}'
MID='"fusion": {"type": "intermediate", "op": "concat"}'
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
run() { mb train --provider manyseg --encoder resnet34 --weights tv_in1k --recipe "$RECIPE" \
          --set "$SET" --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
for seed in 0 1 2; do
  S="--seeds $seed"
  # Shared baseline (Q1 S2-only, Q2 no metadata, Q3 plain unet)
  run $S --dataset cloudsen12 --arch unet --tag e013-s2
  # Q1: static S1GBM backscatter + DEM
  run $S --dataset cloudsen12_s2_aux --arch unet --tag e013-aux-early
  run $S --dataset cloudsen12_s2_aux --arch unet --input-mode dict --options "{$MID}" \
      --tag e013-aux-mid
  # Q2: metadata (lat/lon, day of year, solar zenith/azimuth)
  run $S --dataset cloudsen12_meta --arch unet --input-mode dict --tag e013-meta-film
  run $S --dataset cloudsen12_meta --arch unet --input-mode dict \
      --options '{"input_options": {"meta": {"conditioning": "channels"}}}' --tag e013-meta-channels
  run $S --dataset cloudsen12_meta_nogeo --arch unet --input-mode dict --tag e013-meta-film-nogeo
  # Q3: fastai's full-resolution input stage on a second remote-sensing dataset
  run $S --dataset cloudsen12 --arch unet --options '{"decoder_kwargs": {"input_cross": true}}' \
      --tag e013-unet-cross
  run $S --dataset cloudsen12 --arch fastai_unet --tag e013-fastai_unet
done
echo "ALL DONE"
