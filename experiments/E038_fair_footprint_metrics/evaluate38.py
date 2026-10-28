"""E038 evaluation (README): B0, B0 + regulariser and B3 on WHU val (tuning) and test, against
Douglas-Peucker (1.5 px) GT; primary boundary F with tile-edge points excluded.

    cd <e036 worktree>/experiments/E030_whu_polygons
    uv run --with buildingregulariser==0.2.5 python <repo>/experiments/E038_fair_footprint_metrics/evaluate38.py \
        --b0 <B0_s0.pt> --b3 runs/B3_s0.pt --seed 0 [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import shapely
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
from common import MEAN, STD, WHUTiles, b0_probabilities, b3_probabilities, load_polygons  # noqa: E402
from train import build  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = Path("$MANYSEG_BENCH_DATA/cache/whu_aerial")
TILE = 512
B0_THR = dict(extent_threshold=0.6, boundary_threshold=0.5, seed_threshold=0.15)
GRID = {
    "B0": [dict(simplify=s) for s in (1.0, 1.5, 2.0, 3.0)],
    "B0reg": [dict(parallel_threshold=pt, simplify_tolerance=st) for pt in (2.0, 3.0) for st in (1.0, 1.5, 2.0)],
    "B3": [dict(corner_threshold=c, big_dev=b, agree=a, merge_deg=m)
           for c, b, a, m in itertools.product((0.3, 0.5), (4.0, 8.0), (0.4, 0.6), (8.0, 15.0))],
}
SUMS = ("bp_tp", "bp_n", "bg_tp", "bg_n", "tp", "n_gt", "n_pred", "ciou", "miou", "polis", "snp", "sng",
        "i_tp", "i_n_gt", "i_n_pred", "i_ciou")


def gt_dp(polys):
    return [q for q in (g.simplify(1.5) for g in polys) if q.geom_type == "Polygon" and q.area > 1]


def touches_edge(p, m=1.0):
    x0, y0, x1, y1 = p.bounds
    return min(x0, y0) < m or max(x1, y1) > TILE - m


def boundary_counts(gt, pred, tol=1.0, step=0.5):
    def pts(polys):
        out = []
        for p in polys:
            for part in getattr(p, "geoms", [p]):
                b = part.exterior
                out.append(shapely.line_interpolate_point(b, np.arange(0, b.length, step)))
        if not out:
            return np.array([])
        a = np.concatenate(out)
        xy = shapely.get_coordinates(a)
        keep = (xy[:, 0] > 1) & (xy[:, 1] > 1) & (xy[:, 0] < TILE - 1) & (xy[:, 1] < TILE - 1)
        return a[keep]
    pp, pg = pts(pred), pts(gt)
    bg = shapely.union_all([q.exterior for p in gt for q in getattr(p, "geoms", [p])]) if gt else None
    bp = shapely.union_all([q.exterior for p in pred for q in getattr(p, "geoms", [p])]) if pred else None
    tp_p = int((shapely.distance(pp, bg) <= tol).sum()) if len(pp) and bg is not None else 0
    tp_g = int((shapely.distance(pg, bp) <= tol).sum()) if len(pg) and bp is not None else 0
    return tp_p, len(pp), tp_g, len(pg)


def metrics(gt, pred):
    from manyseg_bench.evaluation.vectors import polygon_metrics
    out = dict.fromkeys(SUMS, 0.0)
    out["bp_tp"], out["bp_n"], out["bg_tp"], out["bg_n"] = boundary_counts(gt, pred)
    m = polygon_metrics(gt, pred, gsd=0.3)
    tp = m["tp"]
    out.update(tp=tp, n_gt=m["n_gt"], n_pred=m["n_pred"], ciou=m["ciou"] * tp, miou=m["miou"] * tp,
               polis=(m["polis"] * tp) if tp else 0.0, snp=m.get("sum_n_pred", 0), sng=m.get("sum_n_gt", 0))
    gi = [g for g in gt if not touches_edge(g)]
    pi = [p for p in pred if not touches_edge(p)]
    mi = polygon_metrics(gi, pi, gsd=0.3)
    out.update(i_tp=mi["tp"], i_n_gt=mi["n_gt"], i_n_pred=mi["n_pred"], i_ciou=mi["ciou"] * mi["tp"])
    return out


def regularise(polys, pt, st):
    import geopandas as gpd
    from buildingregulariser import regularize_geodataframe
    if not polys:
        return []
    r = regularize_geodataframe(gpd.GeoDataFrame(geometry=list(polys)), parallel_threshold=pt,
                                simplify_tolerance=st, allow_circles=False, num_cores=1)
    return [q for g in r.geometry for q in getattr(g, "geoms", [g]) if q.is_valid and q.area > 1]


def labels(a):
    from manyseg.instances import instances_from_outputs
    t = {k: torch.from_numpy(a[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
    return instances_from_outputs(t["extent"], t["boundary"], t["distance"], logits=False, min_size=16, **B0_THR)[0]


def work(item):
    from manyseg.vectors import polygons_from_corners, polygons_from_labels
    torch.set_num_threads(1)
    arm, a, gt, settings, extra = item
    lab = labels(a)
    rows = []
    for p in settings:
        if arm == "B0":
            pred = polygons_from_labels(lab, simplify=p["simplify"], min_area=16)
        elif arm == "B0reg":
            pred = regularise(polygons_from_labels(lab, simplify=extra["simplify"], min_area=16),
                              p["parallel_threshold"], p["simplify_tolerance"])
        else:
            pred = polygons_from_corners(lab, a["corner"].astype(np.float32), a["direction"].astype(np.float32),
                                         corner_threshold=p["corner_threshold"], big_dev=p["big_dev"], agree=p["agree"],
                                         merge_deg=p["merge_deg"], snap_radius=4.0, simplify=1.0, min_area=16)
        rows.append(metrics(gt, pred))
    return rows


def summarise(acc):
    tp = max(acc["tp"], 1)
    prec, rec = acc["bp_tp"] / max(1, acc["bp_n"]), acc["bg_tp"] / max(1, acc["bg_n"])
    det = 2 * acc["tp"] / max(1, acc["n_gt"] + acc["n_pred"])
    idet = 2 * acc["i_tp"] / max(1, acc["i_n_gt"] + acc["i_n_pred"])
    return {"boundary_f": 2 * prec * rec / max(1e-9, prec + rec), "boundary_p": prec, "boundary_r": rec,
            "det_f1": det, "ciou_x_f1": acc["ciou"] / tp * det, "ciou": acc["ciou"] / tp, "miou": acc["miou"] / tp,
            "polis_m": acc["polis"] / tp, "n_ratio": acc["snp"] / max(1, acc["sng"]),
            "interior_det_f1": idet, "interior_ciou_x_f1": acc["i_ciou"] / max(1, acc["i_tp"]) * idet,
            "tp": acc["tp"], "n_gt": acc["n_gt"], "n_pred": acc["n_pred"]}


@torch.no_grad()
def run(model, prob, arm, split, settings, extra, procs, limit):
    ds = WHUTiles(CACHE, split, "B0", augment=False)
    gts = load_polygons(CACHE / f"{split}_poly.npz")
    n = min(len(ds), limit or len(ds))
    accs = [dict.fromkeys(SUMS, 0.0) for _ in settings]
    dev = torch.device("cuda")
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=16, num_workers=4)
    idx = 0
    with Pool(procs) as pool:
        for x, _ in dl:
            x = (x.to(dev).float() - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            items = []
            for b in range(x.shape[0]):
                a = {k: v.astype(np.float16) for k, v in prob(out, b).items()}
                items.append((arm, a, gt_dp(gts[idx]), settings, extra))
                idx += 1
            for rows in pool.map(work, items):
                for acc, r in zip(accs, rows):
                    for k in SUMS:
                        acc[k] += r[k]
    return [summarise(a) for a in accs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--b3", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    res = {"seed": args.seed, "limit": args.limit, "b0": str(args.b0), "b3": str(args.b3)}
    t0 = time.time()
    for arm, path in (("B0", args.b0), ("B3", args.b3)):
        m = build(arm)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m = m.to(dev).eval()
        prob = b0_probabilities if arm == "B0" else b3_probabilities
        arms = [("B0", {}), ("B0reg", None)] if arm == "B0" else [("B3", {})]
        for name, extra in arms:
            if name == "B0reg":
                extra = {"simplify": res["B0"]["params"]["simplify"]}
            val = run(m, prob, name, "val", GRID[name], extra, args.procs, args.limit)
            k = max(range(len(val)), key=lambda j: val[j]["boundary_f"])
            (test,) = run(m, prob, name, "test", [GRID[name][k]], extra, args.procs, args.limit)
            res[name] = {"params": GRID[name][k], "extra": extra, "val": val[k], "test": test,
                         "val_grid": [dict(GRID[name][j], **val[j]) for j in range(len(val))]}
            print(name, "val", round(val[k]["boundary_f"], 4), "test", round(test["boundary_f"], 4),
                  f"{time.time() - t0:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    t = {a: res[a]["test"] for a in ("B0", "B0reg", "B3")}
    best_base = max(t["B0"]["boundary_f"], t["B0reg"]["boundary_f"])
    res["decision_seed0"] = {
        "b3_minus_best_baseline_boundary_f": t["B3"]["boundary_f"] - best_base,
        "b3_det_f1_minus_b0": t["B3"]["det_f1"] - t["B0"]["det_f1"],
        "advance": bool(t["B3"]["boundary_f"] - best_base >= 0.01 and t["B3"]["det_f1"] >= t["B0"]["det_f1"] - 0.03),
    }
    print(json.dumps(res["decision_seed0"], indent=1))
    tag = "" if args.limit is None else f"_limit{args.limit}"
    (args.out / f"e038_s{args.seed}{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
