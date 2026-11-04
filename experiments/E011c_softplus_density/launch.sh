#!/usr/bin/env bash
# Run run.sh from a frozen worktree at the current commit as a detached user service capped at
# 12 GB RAM (no swap). Results go to OUT_ROOT (default: this checkout). UNIT=e011c by default;
# PREFLIGHT=1 is passed through.
set -euo pipefail
MAIN="$(cd "$(dirname "$0")/../.." && pwd)"
SHA=$(git -C "$MAIN" rev-parse --short HEAD)
WT="${MANYSEG_RUNS:?set to the frozen-worktree directory}/e011c-$SHA"
[ -d "$WT" ] || git -C "$MAIN" worktree add --detach "$WT" "$SHA"
(cd "$WT" && uv sync --all-packages -q)
OUT_ROOT=${OUT_ROOT:-$MAIN}
LOG="$OUT_ROOT/experiments/E011c_softplus_density/runs/${UNIT:-e011c}.log"
mkdir -p "$(dirname "$LOG")"
echo "code $SHA, worktree $WT, log $LOG"
systemd-run --user --unit "${UNIT:-e011c}" -p MemoryMax=12G -p MemorySwapMax=0 \
  -p "WorkingDirectory=$WT" -p "StandardOutput=append:$LOG" -p "StandardError=append:$LOG" \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=OUT_ROOT="$OUT_ROOT" \
  --setenv=PREFLIGHT="${PREFLIGHT:-}" \
  bash "$WT/experiments/E011c_softplus_density/run.sh"
