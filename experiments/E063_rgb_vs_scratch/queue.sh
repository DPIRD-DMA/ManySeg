#!/usr/bin/env bash
# E063 queue. Part B from the frozen E063 worktree, part A from E024's commit so its imagenet
# runs stay a same-code baseline. Results land in this checkout's experiment folder.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
export OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to the checkout that keeps the results}"
RUNS="${RUNS_ROOT:?set RUNS_ROOT to the folder of frozen worktrees}"
FROZEN="$RUNS/${E063_FROZEN:?set E063_FROZEN}"
rc=0
echo "=== part B $(git -C "$FROZEN" rev-parse --short HEAD) $(date)"
(cd "$FROZEN" && ./experiments/E063_rgb_vs_scratch/run_b.sh) || rc=1
echo "=== part A $(git -C "$RUNS/9a2d519" rev-parse --short HEAD) $(date)"
(cd "$RUNS/9a2d519" && "$FROZEN/experiments/E063_rgb_vs_scratch/run_a.sh") || rc=1
echo "=== queue finished rc=$rc $(date)"
exit "$rc"
