"""E042: E039's harness with the min-cost healing mode added to the tuning grid."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E039_unified_roads"))
import run39  # noqa: E402

base = dict(thr_k=0.2, zr=8.0, pair_deg=150.0, thr_line=0.3)
run39.GRID = [dict(base, mode="centre"), dict(base, mode="surface")] + [
    dict(base, mode="mincost", cost_cap=c, heal_cost=h, heal_r=r) for c, h, r in ((3.0, 40.0, 50), (2.0, 60.0, 60), (7.0, 25.0, 40))]
run39.RUNS = HERE / "runs"

if __name__ == "__main__":
    run39.main()
