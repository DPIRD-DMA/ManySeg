#!/usr/bin/env bash
# E030 chain: preflight (both arms, tiny), then 6 runs (train -> evaluate), interleaved by seed.
# Run from a frozen copy of the code (see OVERNIGHT.md); logs in runs/.
set -euo pipefail
cd "$(dirname "$0")"
export E030_CACHE="$MANYSEG_BENCH_DATA/cache/whu_aerial"
mkdir -p runs
git rev-parse HEAD > runs/commit.txt 2>/dev/null || true
for arm in B0 B2; do
  uv run python train.py --arm $arm --seed 0 --cache "$E030_CACHE" --epochs 1 --limit 64 --tag _preflight
  uv run python evaluate.py --run ${arm}_s0_preflight --cache "$E030_CACHE" --limit 32
done
for seed in 0 1 2; do
  for arm in B0 B2; do
    if [ ! -f "runs/${arm}_s${seed}_eval.json" ]; then
      [ -f "runs/${arm}_s${seed}.pt" ] || uv run python train.py --arm $arm --seed $seed --cache "$E030_CACHE"
      uv run python evaluate.py --run ${arm}_s${seed} --cache "$E030_CACHE"
    fi
  done
done
echo "E030 chain done"
