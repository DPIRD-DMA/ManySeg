#!/usr/bin/env bash
# Runs E003 -> E001b -> E005 from a frozen worktree after E002 finishes.
set -u
MAIN="$WORKSPACE/ManySeg"
cd "$MAIN"
until [ -f experiments/E002_provider_comparison_pet/run.log.exit ]; do sleep 60; done
echo "E002 finished with exit $(cat experiments/E002_provider_comparison_pet/run.log.exit)"
# Pre-registered rule: recipe with the higher mean final val mIoU across E002 runs.
RECIPE=$(uv run python - <<'PY'
import json, glob, statistics, collections
by = collections.defaultdict(list)
for f in glob.glob("experiments/E002_provider_comparison_pet/runs/*.json"):
    r = json.load(open(f)); by[r["recipe"]["name"]].append(r["final_val"]["miou"])
for k, v in by.items(): print(f"{k}: n={len(v)} mean={statistics.mean(v):.4f}", file=__import__('sys').stderr)
print(max(by, key=lambda k: statistics.mean(by[k])))
PY
)
echo "RECIPE=$RECIPE"
SHA=$(git rev-parse --short HEAD)
WT="$WORKSPACE/ManySeg-runs/$SHA"
[ -d "$WT" ] || git worktree add --detach "$WT" "$SHA"
cd "$WT" && uv sync --all-packages -q
export OUT_ROOT="$MAIN" RECIPE
# Fail fast: one tiny real run through the same CLI path before committing hours of GPU time.
if ! uv run manyseg-bench train --provider manyseg --recipe smoke --set '{"epochs": 1}' \
     --out "$MAIN/runs/preflight" --tag preflight; then
  echo "PREFLIGHT FAILED"; exit 1
fi
for s in experiments/E003_decoders_pet/run.sh experiments/E001_upsample_alignment/run_trained.sh \
         experiments/E005_fusion_sen1floods11/run.sh; do
  echo "##### $s (code $SHA)"
  bash "$s"
done
echo "QUEUE DONE"
