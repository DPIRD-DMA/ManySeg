#!/usr/bin/env bash
# E024: ImageNet vs remote-sensing pretrained ResNet-50 weights (README.md). Local RTX 4090.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E024_rs_pretrained_weights/runs
SEEDS=${SEEDS:-0,1,2}
fails=0
run() {  # arm dataset weights arch set
  uv run manyseg-bench train --provider manyseg --encoder resnet50 --weights "$3" --arch "$4" \
    --dataset "$2" --recipe fastai_like --set "$5" --seeds "$SEEDS" --out "$OUT" \
    --evaluate-test --tag "e024-$2-$1" || { echo "FAILED $2 $1"; fails=$((fails + 1)); }
}
for ds in ${DATASETS:-sen1floods11_s2 bigearthnet pastis}; do
  case $ds in
    sen1floods11_s2) ARCH=unet; SET="{\"epochs\": ${S1F_EPOCHS:-60}, \"batch_size\": 16}" ;;
    bigearthnet)     ARCH=none; SET="{\"epochs\": ${BEN_EPOCHS:-12}, \"batch_size\": 64}" ;;
    pastis)          ARCH=unet; SET="{\"epochs\": ${PASTIS_EPOCHS:-30}, \"batch_size\": 16}" ;;
  esac
  for arm in ${ARMS:-imagenet ssl4eo_moco ssl4eo_dino satlas}; do
    case $arm in
      imagenet)    run "$arm" "$ds" tv_in1k "$ARCH" "$SET" ;;
      ssl4eo_moco) run "$arm" "$ds" ssl4eo_s2_moco "$ARCH" "$SET" ;;
      ssl4eo_dino) run "$arm" "$ds" ssl4eo_s2_dino "$ARCH" "$SET" ;;
      satlas)      run "$arm" "$ds" satlas_s2_si_ms "$ARCH" "$SET" ;;
    esac
  done
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
