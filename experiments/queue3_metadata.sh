#!/usr/bin/env bash
# Queue 3: E009 metadata conditioning. Waits for queue 2 to finish and for the GPU to be free of
# other bench training (queue 4 also starts after queue 2), then runs E009 from a frozen
# worktree of the then-HEAD.
# Launch (not done automatically):
#   nohup bash experiments/queue3_metadata.sh > experiments/queue3.log 2>&1; \
#     echo $? > experiments/queue3.log.exit
set -u
MAIN="$WORKSPACE/ManySeg"
DATA="$MANYSEG_BENCH_DATA"
cd "$MAIN"
until [ -f experiments/queue2.log.exit ]; do sleep 60; done
sleep 120  # let queue 4 start (or exit) first
while pgrep -f "queue4_instances.sh|manyseg-bench train" > /dev/null; do sleep 60; done
SHA=$(git rev-parse --short HEAD)
WT="$WORKSPACE/ManySeg-runs/$SHA"
[ -d "$WT" ] || git worktree add --detach "$WT" "$SHA"
cd "$WT" && uv sync --all-packages -q
export OUT_ROOT="$MAIN" RECIPE=fastai_like MANYSEG_BENCH_DATA="$DATA"
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --out "$MAIN/runs/preflight" --tag preflight || { echo "PREFLIGHT FAILED"; exit 1; }
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --dataset sen1floods11_meta --input-mode dict --split events:5:0 \
  --out "$MAIN/runs/preflight" --tag preflight-meta || { echo "PREFLIGHT (META) FAILED"; exit 1; }
echo "##### experiments/E009_metadata_conditioning/run.sh (code $SHA)"
bash experiments/E009_metadata_conditioning/run.sh
echo "QUEUE3 DONE"
