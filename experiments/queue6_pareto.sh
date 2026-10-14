#!/usr/bin/env bash
# Queue 6: E006 decoder Pareto front. Runs after the whole local chain queue2 -> queue4 (E010)
# -> queue5 (E011) -> queue3 (E009): waits for queue3's exit file, then for no other bench
# training, then runs E006 from a frozen worktree of the then-HEAD: stage 1 (grid, seed 0),
# the pre-registered frontier selection, stage 2 (seeds 1-2 on the frontier) and the profile
# sweep (latency / memory, contention recorded per run; re-run STAGE=profile on an idle GPU
# before quoting latencies).
# Launch (not done automatically):
#   nohup bash experiments/queue6_pareto.sh > experiments/queue6.log 2>&1; \
#     echo $? > experiments/queue6.log.exit
set -u
MAIN="$WORKSPACE/ManySeg"
DATA="$MANYSEG_BENCH_DATA"
cd "$MAIN"
until [ -f experiments/queue3.log.exit ]; do sleep 60; done
# Name the other queues explicitly: a generic "queue*.sh" pattern would match this script.
while pgrep -f "queue_after_e002|queue2\.sh|queue3_metadata|queue4_instances|queue5_points|manyseg-bench train" \
    > /dev/null; do sleep 60; done
for ds in oxford_pet sen1floods11; do
  [ -f "$DATA/cache/$ds/test_masks.npy" ] || { echo "DATASET NOT PREPARED: $ds"; exit 3; }
done
SHA=$(git rev-parse --short HEAD)
WT="$WORKSPACE/ManySeg-runs/$SHA"
[ -d "$WT" ] || git worktree add --detach "$WT" "$SHA"
cd "$WT" && uv sync --all-packages -q
export OUT_ROOT="$MAIN" RECIPE=fastai_like MANYSEG_BENCH_DATA="$DATA"
E="experiments/E006_decoder_pareto"
# Preflight: one smoke epoch for each new ingredient (reduced out_stride; the PVTv2 encoder
# and its in1k weights; stacked Sen1Floods11), so a broken arm fails here and not at hour 5.
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --options '{"decoder_kwargs": {"out_stride": 2}}' --arch fastai_unet \
  --out "$MAIN/runs/preflight" --tag preflight-e006 || { echo "PREFLIGHT FAILED"; exit 1; }
uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
  --dataset sen1floods11 --encoder pvt_v2_b1 --weights in1k --arch unet \
  --options '{"decoder_kwargs": {"out_stride": 4}}' \
  --out "$MAIN/runs/preflight" --tag preflight-e006-pvt || { echo "PREFLIGHT (PVT) FAILED"; exit 1; }
echo "##### $E/run.sh STAGE=1 (code $SHA)"
STAGE=1 bash "$E/run.sh"
echo "##### frontier selection (pre-registered rule, val only)"
uv run python "$E/analyse.py" "$MAIN/$E/runs" --write-frontier "$MAIN/$E/frontier.txt" \
  || { echo "FRONTIER SELECTION FAILED"; exit 1; }
cat "$MAIN/$E/frontier.txt"
echo "##### $E/run.sh STAGE=2 (code $SHA)"
STAGE=2 FRONTIER="$MAIN/$E/frontier.txt" bash "$E/run.sh"
echo "##### $E/run.sh STAGE=profile (code $SHA)"
STAGE=profile bash "$E/run.sh"
echo "QUEUE6 DONE"
