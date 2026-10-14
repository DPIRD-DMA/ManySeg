#!/usr/bin/env bash
# Queue 5: E011 tree points. Waits for queue4 (E010) to finish and the OAM-TCD cache to exist.
set -u
MAIN="$WORKSPACE/ManySeg"
DATA="$MANYSEG_BENCH_DATA"
cd "$MAIN"
until [ -f experiments/queue4.log.exit ]; do sleep 60; done
if [ ! -f "$DATA/cache/oam_tcd/test_masks.npy" ]; then
  echo "DATASET NOT PREPARED: oam_tcd"; exit 3
fi
SHA=$(git rev-parse --short HEAD)
WT="$WORKSPACE/ManySeg-runs/$SHA"
[ -d "$WT" ] || git worktree add --detach "$WT" "$SHA"
cd "$WT" && uv sync --all-packages -q
export OUT_ROOT="$MAIN" MANYSEG_BENCH_DATA="$DATA"
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --out "$MAIN/runs/preflight" --tag preflight || { echo "PREFLIGHT FAILED"; exit 1; }
echo "##### experiments/E011_tree_points/run.sh (code $SHA)"
bash experiments/E011_tree_points/run.sh
echo "QUEUE5 DONE"
