"""E046 evaluation (README): B0 and B3 models on FTW NL val (tuning) and test; boundary F with
chip-edge points excluded (primary), detection F1 and others secondary.

    uv run python evaluate46.py --b0 runs/B0_s0.pt --b3 runs/B3_s0.pt --seed 0 [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import shapely
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
sys.path.insert(0, str(HERE))
from common import b0_probabilities, b3_probabilities, load_polygons  # noqa: E402
from train46 import CACHE, TILE, FTWChips, S2Loader, build  # noqa: E402

THR = [dict(extent_threshold=e, boundary_threshold=b, seed_threshold=s)
       for e, b, s in itertools.product((0.4, 0.6), (0.3, 0.5), (0.15, 0.3))]
GRID = {"B0": [dict(simplify=s) for s in (0.5, 1.0, 1.5, 2.0)],
        "B3": [dict(corner_threshold=c, big_dev=b, agree=a, merge_deg=m)
               for c, b, a, m in itertools.product((0.3, 0.5), (4.0, 8.0), (0.4, 0.6), (8.0, 15.0))]}
SUMS = ("bp_tp", "bp_n", "bg_tp", "bg_n", "tp", "n_gt", "n_pred", "ciou", "polis")


def boundary_counts(gt, pred, tol=1.0, step=0.5):
    def pts(polys):
        out = [shapely.line_interpolate_point(b, np.arange(0, b.length, step))
               for p in polys for b in (p.exterior, *p.interiors)]
        if not out:
            return np.array([])
        a = np.concatenate(out)
        xy = shapely.get_coordinates(a)
        return a[(xy[:, 0] > 1) & (xy[:, 1] > 1) & (xy[:, 0] < TILE - 1) & (xy[:, 1] < TILE - 1)]

    pp, pg = pts(pred), pts(gt)
    bg = shapely.union_all([p.boundary for p in gt]) if gt else None
    bp = shapely.union_all([p.boundary for p in pred]) if pred else None
    tp_p = int((shapely.distance(pp, bg) <= tol).sum()) if len(pp) and bg is not None else 0
    tp_g = int((shapely.distance(pg, bp) <= tol).sum()) if len(pg) and bp is not None else 0
    return tp_p, len(pp), tp_g, len(pg)


def metrics(gt, pred):
    from manyseg_bench.evaluation.vectors import polygon_metrics

    out = dict(zip(("bp_tp", "bp_n", "bg_tp", "bg_n"), boundary_counts(gt, pred)))
    m = polygon_metrics(gt, pred, gsd=1.0)
    tp = m["tp"]
    out.update(tp=tp, n_gt=m["n_gt"], n_pred=m["n_pred"], ciou=m["ciou"] * tp, polis=(m["polis"] * tp) if tp else 0.0)
    return out


def work(item):
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import polygons_from_corners, polygons_from_labels

    torch.set_num_threads(1)
    arm, a, gt, thr_list, settings = item
    rows = []
    for thr in thr_list:
        t = {k: torch.from_numpy(a[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
        lab = instances_from_outputs(t["extent"], t["boundary"], t["distance"], logits=False, min_size=16, **thr)[0]
        for p in settings:
            if arm == "B0":
                pred = polygons_from_labels(lab, simplify=p["simplify"], min_area=16)
            else:
                pred = polygons_from_corners(lab, a["corner"].astype(np.float32), a["direction"].astype(np.float32),
                                             corner_threshold=p["corner_threshold"], big_dev=p["big_dev"],
                                             agree=p["agree"], merge_deg=p["merge_deg"], snap_radius=4.0,
                                             simplify=1.0, min_area=16)
            rows.append(metrics(gt, pred))
    return rows


def summarise(acc):
    tp = max(acc["tp"], 1)
    prec, rec = acc["bp_tp"] / max(1, acc["bp_n"]), acc["bg_tp"] / max(1, acc["bg_n"])
    return {"boundary_f": 2 * prec * rec / max(1e-9, prec + rec), "boundary_p": prec, "boundary_r": rec,
            "det_f1": 2 * acc["tp"] / max(1, acc["n_gt"] + acc["n_pred"]), "ciou": acc["ciou"] / tp,
            "polis_px": acc["polis"] / tp, "tp": acc["tp"], "n_gt": acc["n_gt"], "n_pred": acc["n_pred"]}


@torch.no_grad()
def run(model, prob, arm, split, thr_list, settings, procs, limit):
    ds = FTWChips(split, "B0", augment=False)
    gts = load_polygons(CACHE / f"{split}_poly.npz")
    n = min(len(ds), limit or len(ds))
    accs = [dict.fromkeys(SUMS, 0.0) for _ in range(len(thr_list) * len(settings))]
    dev = torch.device("cuda")
    dl = S2Loader(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=32, num_workers=4), dev)
    idx = 0
    with Pool(procs) as pool:
        for x, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            items = []
            for b in range(x.shape[0]):
                items.append((arm, {k: v.astype(np.float16) for k, v in prob(out, b).items()}, gts[idx], thr_list, settings))
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
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = {"seed": args.seed, "limit": args.limit}
    t0 = time.time()
    for arm, path in (("B0", args.b0), ("B3", args.b3)):
        m = build(arm)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m = m.to(dev).eval()
        prob = b0_probabilities if arm == "B0" else b3_probabilities
        # stage 1: instance thresholds (with simplify 1 / the first B3 setting); stage 2: the arm's grid
        v1 = run(m, prob, arm, "val", THR, GRID[arm][:1] if arm == "B3" else [dict(simplify=1.0)], args.procs, args.limit)
        thr = THR[max(range(len(THR)), key=lambda j: v1[j]["boundary_f"])]
        v2 = run(m, prob, arm, "val", [thr], GRID[arm], args.procs, args.limit)
        k = max(range(len(v2)), key=lambda j: v2[j]["boundary_f"])
        (test,) = run(m, prob, arm, "test", [thr], [GRID[arm][k]], args.procs, args.limit)
        res[arm] = {"thr": thr, "params": GRID[arm][k], "val": v2[k], "test": test,
                    "val_thr": [dict(THR[j], **v1[j]) for j in range(len(THR))],
                    "val_grid": [dict(GRID[arm][j], **v2[j]) for j in range(len(v2))]}
        print(arm, "val", round(v2[k]["boundary_f"], 4), "test", round(test["boundary_f"], 4),
              "det", round(test["det_f1"], 4), f"{time.time() - t0:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    t0_, t3 = res["B0"]["test"], res["B3"]["test"]
    res["decision"] = {"b3_minus_b0_boundary_f": t3["boundary_f"] - t0_["boundary_f"],
                       "b3_minus_b0_det_f1": t3["det_f1"] - t0_["det_f1"]}
    out = HERE / "runs" / f"eval_s{args.seed}{'_limit' if args.limit else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
