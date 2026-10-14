#!/usr/bin/env bash
# Queue 4: E010 instance heads. Waits for queue2 to finish AND for the dataset cache to exist
# (dataset download needs user approval), then runs from a frozen worktree.
set -u
MAIN="$WORKSPACE/ManySeg"
DATA="$MANYSEG_BENCH_DATA"
cd "$MAIN"
until [ -f experiments/queue2.log.exit ]; do sleep 60; done
if [ ! -f "$DATA/cache/ai4boundaries/train_masks.npy" ]; then
  echo "DATASET NOT PREPARED: ai4boundaries (awaiting user approval)"; exit 3
fi
SHA=$(git rev-parse --short HEAD)
WT="$WORKSPACE/ManySeg-runs/$SHA"
[ -d "$WT" ] || git worktree add --detach "$WT" "$SHA"
cd "$WT" && uv sync --all-packages -q
export OUT_ROOT="$MAIN" MANYSEG_BENCH_DATA="$DATA"
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --out "$MAIN/runs/preflight" --tag preflight || { echo "PREFLIGHT FAILED"; exit 1; }
echo "##### experiments/E010_instance_heads/run.sh (code $SHA)"
bash experiments/E010_instance_heads/run.sh
echo "QUEUE4 DONE"
