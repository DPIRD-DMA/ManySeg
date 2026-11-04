#!/usr/bin/env bash
# E026 part 2: training-loop variants, out-of-RAM prefetch, post-processing. Waits for the
# model grid to finish. Run from the frozen worktree; results go to the main tree's runs/.
set -u
MAIN="${MANYSEG_REPO:?set to the main checkout}"
D="experiments/E026_perf_audit"
R="$MAIN/$D/runs"
mkdir -p "$R"
until [ -f "$MAIN/experiments/e026_models.log.exit" ]; do sleep 30; done
gpu() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; }

# --- training loop: each variant alone, then all together; 2 interleaved reps ---
for cfg in "--model unet_r34 --size 512" "--model segformer_pvt --size 256" \
           "--model unet_pvt --size 256"; do
  for rep in 1 2; do
    for v in base nosync foreach fused loader nosync,foreach,fused,loader; do
      # shellcheck disable=SC2086
      uv run python "$D/train_bench.py" $cfg --variant $v --steps 100 --tag "$(gpu)" 2>&1 \
        | grep '^{' >> "$R/train.jsonl" || echo "FAILED train $cfg $v"
      echo "done train $cfg $v rep $rep"
    done
  done
done
for rep in 1 2; do for v in base loader; do  # 13 bands: more loader copies per pixel
  uv run python "$D/train_bench.py" --model unet_r34 --size 256 --in-ch 13 --variant $v \
    --steps 100 --tag "$(gpu)" 2>&1 | grep '^{' >> "$R/train.jsonl" || echo "FAILED train13 $v"
  echo "done train13 $v rep $rep"
done; done

# --- CloudSEN12+ cold reads: synchronous vs prefetch ---
for rep in 0 1; do for v in fetch_only seq prefetch; do
  uv run python "$D/fetch_bench.py" --variant $v --seed $rep 2>&1 | grep '^{' >> "$R/fetch.jsonl" \
    || echo "FAILED fetch $v"
  echo "done fetch $v rep $rep"
done; done

# --- post-processing ---
for dev in cpu cuda; do
  uv run python "$D/post_bench.py" --device $dev 2>&1 | grep '^{' >> "$R/post.jsonl" \
    || echo "FAILED post $dev"
  echo "done post $dev"
done
echo ALL DONE
