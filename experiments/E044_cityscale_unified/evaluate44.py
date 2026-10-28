"""E044 evaluation: R0 (mask skeleton) and R4 (unified decoder) on the SAME predictions; decoders
tuned on the 9 validation regions (APLS), scored once on the 27 test regions (README).

    uv run python evaluate44.py --cache <e044 cache> --run U_s0 [--procs 6] [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import data as D
from train44 import Tiles, build

R, E = D.R, D.E
HERE = Path(__file__).resolve().parent
R0_GRID = [dict(thr=t, spur=s, gap=g, simplify=2, min_len=20)
           for t, s, g in itertools.product((0.3, 0.5), (10, 30), (0, 25, 50))]
R4_GRID = ([dict(thr_k=k, zr=8.0, pair_deg=150.0, thr_line=tl, mode=m)
            for k, tl, m in itertools.product((0.2, 0.4), (0.3, 0.5), ("centre", "surface"))]
           + [dict(thr_k=k, zr=8.0, pair_deg=150.0, thr_line=0.3, mode="mincost", cost_cap=2.0, heal_cost=60.0, heal_r=60)
              for k in (0.2, 0.4)])
_TILES: dict = {}


def gt_tiles(split):
    if split not in _TILES:
        ids = json.loads((Path(ARGS.cache) / f"{split}_ids.json").read_text())
        _TILES[split] = [t for rid in ids for t in D.region_tiles(rid)]
    return _TILES[split]


def work(item):
    torch.set_num_threads(1)
    split, i, a, r0_list, r4_list, r0_best = item
    t = gt_tiles(split)[i]
    if t.graph.number_of_edges() < 10:
        return None
    rows = []
    E.CACHE.clear()
    for k, p in enumerate(r0_list):
        m = E.road_metrics(t, E.decode_r0(a, p))
        rows.append(dict(arm="R0", pidx=k, **m))
    for k, p in enumerate(r4_list):
        m = E.road_metrics(t, R.decode_r4(a, dict(p, r0=r0_best, gap=r0_best["gap"])))
        rows.append(dict(arm="R4", pidx=k, **m))
    return rows


@torch.no_grad()
def predict(model, split, limit):
    ds = Tiles(Path(ARGS.cache), split, augment=False)
    n = min(len(ds), limit or len(ds))
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1) * 255
    for i in range(n):
        x = (ds[i][0][None].cuda().float() - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        a = {"mask": torch.sigmoid(o["mask"][0, 0].float()).cpu().numpy(),
             "line": torch.sigmoid(o["line"][0, 0].float()).cpu().numpy(),
             "direction": o["direction"][0].float().cpu().numpy()}
        kp = torch.sigmoid(o["kp"][0].float()).cpu().numpy()
        a["junction"], a["end"], a["crossing"] = kp[0], kp[1], kp[2]
        yield i, a


def run(model, split, r0_list, r4_list, r0_best, procs, limit):
    acc = {}
    items = ((split, i, a, r0_list, r4_list, r0_best) for i, a in predict(model, split, limit))
    with Pool(procs, maxtasksperchild=8) as pool:
        for rows in pool.imap_unordered(work, items):
            if rows is None:
                continue
            for r in rows:
                acc.setdefault((r["arm"], r["pidx"]), []).append(r)
    out = {}
    for (arm, k), rs in acc.items():
        cn = sum(r["cross_n"] for r in rs)
        out[(arm, k)] = {"apls": float(np.mean([r["apls"] for r in rs])), "junction_f1": float(np.mean([r["junction_f1"] for r in rs])),
                         "geo_f1": float(np.mean([r["f1"] for r in rs])) if "f1" in rs[0] else None,
                         "crossings_kept": sum(r["cross_ok"] for r in rs) / max(cn, 1), "cross_n": cn, "tiles": len(rs)}
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--run", default="U_s0")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ARGS = ap.parse_args()
    model = build()
    model.load_state_dict(torch.load(HERE / "runs" / f"{ARGS.run}.pt", map_location="cpu"))
    model = model.cuda().eval()
    t0 = time.time()
    v0 = run(model, "val", R0_GRID, [], R0_GRID[0], ARGS.procs, ARGS.limit)
    k0 = max(range(len(R0_GRID)), key=lambda k: v0[("R0", k)]["apls"])
    r0_best = R0_GRID[k0]
    v4 = run(model, "val", [], R4_GRID, r0_best, ARGS.procs, ARGS.limit)
    k4 = max(range(len(R4_GRID)), key=lambda k: v4[("R4", k)]["apls"])
    test = run(model, "test", [r0_best], [R4_GRID[k4]], r0_best, ARGS.procs, ARGS.limit)
    r0, r4 = test[("R0", 0)], test[("R4", 0)]
    dec = {"r4_minus_r0_apls": r4["apls"] - r0["apls"], "r4_minus_r0_junction_f1": r4["junction_f1"] - r0["junction_f1"],
           "r4_minus_r0_crossings": r4["crossings_kept"] - r0["crossings_kept"]}
    dec["unified_heads_viable"] = bool(dec["r4_minus_r0_apls"] >= -0.01 and dec["r4_minus_r0_junction_f1"] >= 0.05
                                       and dec["r4_minus_r0_crossings"] >= 0.20)
    res = {"run": ARGS.run, "limit": ARGS.limit, "r0_params": r0_best, "r4_params": R4_GRID[k4],
           "val_r0": v0[("R0", k0)], "val_r4": v4[("R4", k4)], "test_r0": r0, "test_r4": r4, "decision": dec,
           "val_r4_grid": [dict(R4_GRID[k], **v4[("R4", k)]) for k in range(len(R4_GRID))], "wall_s": time.time() - t0}
    print(json.dumps({k: res[k] for k in ("r0_params", "r4_params", "test_r0", "test_r4", "decision")}, indent=1, default=float))
    tag = "" if ARGS.limit is None else f"_limit{ARGS.limit}"
    (HERE / "runs" / f"eval_{ARGS.run}{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
