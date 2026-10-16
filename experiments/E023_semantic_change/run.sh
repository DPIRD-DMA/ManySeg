#!/usr/bin/env bash
# E023: semantic change detection on SECOND, fused vs per-input land-cover heads (README.md).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E023_semantic_change/runs
SEEDS=${SEEDS:-0,1,2}
SET="{\"epochs\": ${EPOCHS:-30}, \"batch_size\": 8}"
SIAM='"fusion": {"type": "intermediate", "op": "concat_diff"}, "input_options": {"t2": {"backbone": "shared:t1", "share_norm": true}}'
fails=0
run() {  # tag input_mode options task_kwargs
  uv run manyseg-bench train --provider manyseg --arch unet --encoder resnet34 \
    --weights tv_in1k --recipe fastai_like --dataset second --input-mode "$2" \
    --options "$3" --task-kwargs "$4" --set "$SET" --seeds "$SEEDS" --out "$OUT" \
    --evaluate-test --tag "e023-$1" || { echo "FAILED $1"; fails=$((fails + 1)); }
}
for arm in ${ARMS:-stack siam-fused siam-perinput-shared siam-perinput-separate}; do
  case $arm in
    stack)                  run "$arm" stack '{}' '{"per_input": false}' ;;
    siam-fused)             run "$arm" dict "{$SIAM}" '{"per_input": false}' ;;
    siam-perinput-shared)   run "$arm" dict "{$SIAM}" '{"per_input": true, "share": true}' ;;
    siam-perinput-separate) run "$arm" dict "{$SIAM}" '{"per_input": true, "share": false}' ;;
  esac
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
