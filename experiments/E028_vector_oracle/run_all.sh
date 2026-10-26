#!/usr/bin/env bash
# E028 full chain: tune on innaloo, then test on murdoch + northam.
set -euo pipefail
cd "$(dirname "$0")"
export E028_DATA="$MANYSEG_BENCH_DATA/osm_e028"
git rev-parse HEAD > runs/commit.txt
uv run python run.py tune1 -n 8
uv run python run.py tune2 -n 8
uv run python run.py test -n 8
