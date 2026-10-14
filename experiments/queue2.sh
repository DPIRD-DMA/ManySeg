#!/usr/bin/env bash
# Second queue: waits for queue 1, then runs E004 from a frozen worktree of the then-HEAD.
set -u
MAIN="$WORKSPACE/ManySeg"
cd "$MAIN"
until [ -f experiments/queue.log.exit ]; do sleep 60; done
SHA=$(git rev-parse --short HEAD)
WT="$WORKSPACE/ManySeg-runs/$SHA"
[ -d "$WT" ] || git worktree add --detach "$WT" "$SHA"
cd "$WT" && uv sync --all-packages -q
export OUT_ROOT="$MAIN" RECIPE=fastai_like MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --out "$MAIN/runs/preflight" --tag preflight || { echo "PREFLIGHT FAILED"; exit 1; }
for s in experiments/E004_channel_adapters/run.sh; do echo "##### $s (code $SHA)"; bash "$s"; done
echo "QUEUE2 DONE"
