"""E054 (README): polygons_from_corners holes False vs True at the shipped footprint settings.

    python e054.py [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path

import manyseg.vectors as V
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUNS = Path(os.environ.get("MANYSEG_RUNS", "../ManySeg-runs"))
CKPT = [RUNS / f"e036-f2d2f86/experiments/E030_whu_polygons/runs/B3_s{s}.pt" for s in (0, 1)]
THR = dict(extent_threshold=0.6, boundary_threshold=0.5, seed_threshold=0.15, min_seed=8)
PARAMS = dict(corner_threshold=0.5, big_dev=4.0, agree=0.4, merge_deg=8.0)
ORIGINAL = V.polygons_from_corners
N_HOLES = mp.Value("q", 0)  # predicted holes, across the forked decoding pool


def counted(*args, **kw):
    out = ORIGINAL(*args, **kw)
    with N_HOLES.get_lock():
        N_HOLES.value += sum(len(p.interiors) for p in out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = HERE / "runs" / f"e054{'_limit' if args.limit else ''}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    os.chdir(EXP / "E030_whu_polygons")  # evaluate38 imports common / train from the cwd
    sys.path.insert(0, str(EXP / "E030_whu_polygons"))
    sys.path.insert(0, str(EXP / "E038_fair_footprint_metrics"))
    import evaluate38 as E

    E.B0_THR = THR
    res, t0 = {"limit": args.limit, "thr": THR, "params": PARAMS}, time.time()
    for ckpt in CKPT:
        m = E.build("B3")
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m = m.to("cuda").eval()
        r = {"sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
        for split in ("val", "test"):
            polys = E.load_polygons(E.CACHE / f"{split}_poly.npz")
            r[split] = {"gt_holes": sum(len(g.interiors) for tile in polys for g in E.gt_dp(tile))}
            for holes in (False, True):
                V.polygons_from_corners = partial(counted, holes=holes)  # forked into the pool
                N_HOLES.value = 0
                (s,) = E.run(m, E.b3_probabilities, "B3", split, [PARAMS], {}, args.procs, args.limit)
                r[split][f"holes_{holes}"] = dict(s, pred_holes=N_HOLES.value)
                print(ckpt.name, split, holes, {k: round(s[k], 4) for k in ("boundary_f", "det_f1")},
                      "pred_holes", N_HOLES.value, "gt_holes", r[split]["gt_holes"],
                      f"{time.time() - t0:.0f}s", flush=True)
        res[str(ckpt)] = r
    runs = [res[str(c)]["test"] for c in CKPT]
    diff = {k: sum(t["holes_True"][k] - t["holes_False"][k] for t in runs) / len(runs)
            for k in ("boundary_f", "det_f1", "boundary_p", "boundary_r", "ciou", "n_ratio")}
    res["decision"] = {"true_minus_false": diff,
                       "adopt": diff["boundary_f"] >= -0.005 and diff["det_f1"] >= -0.005}
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
