#!/usr/bin/env bash
# E007: decoder norm {batch, group, none} x encoder BN {train, frozen} x batch {2, 16}.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E007_small_batch_norm/runs
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
for seed in 0 1 2; do
  for ds in oxford_pet sen1floods11; do
    ep=$([ "$ds" = oxford_pet ] && echo 20 || echo 60)
    for bs in 2 16; do
      for norm in batch group none; do
        for encbn in false true; do
          dn=$([ "$norm" = none ] && echo null || echo "\"$norm\"")
          mb train --provider manyseg --dataset "$ds" --encoder resnet34 --weights tv_in1k \
            --arch unet --options "{\"decoder_kwargs\": {\"norm\": $dn}}" --recipe "$RECIPE" \
            --set "{\"epochs\": $ep, \"batch_size\": $bs, \"freeze_encoder_bn\": $encbn}" \
            --seeds "$seed" --out "$OUT" --evaluate-test \
            --tag "e007-$ds-b$bs-$norm-encbn_$([ "$encbn" = true ] && echo frozen || echo train)" \
            || echo "FAILED $ds $bs $norm $encbn $seed"
        done
      done
    done
  done
done
echo "ALL DONE"
