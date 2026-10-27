#!/usr/bin/env bash
# E031 chain (CPU): tune on delft, then test. Launch with run_e031_capped.sh.
set -euo pipefail
cd "$(dirname "$0")"
export E031_DATA="$MANYSEG_BENCH_DATA/raw/3dbag"
mkdir -p runs
git rev-parse HEAD > runs/commit.txt 2>/dev/null || true
[ -f runs/best.json ] || uv run python run.py tune -n 6
uv run python run.py test -n 6
echo "E031 chain done"
