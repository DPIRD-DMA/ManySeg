"""E036 post hoc (registered as 'C-IoU on GT outlines cleaned of staircase vertices'): B3_s0 vs B0_s0
on 300 evenly spaced val tiles, scored against the raw WHU outlines and against outlines regularised
with buildingregulariser 0.2.5 (parallel_threshold 3 px, simplify_tolerance 1.5 px; E037), plus
rendered tiles.

    cd <e036 worktree>/experiments/E030_whu_polygons
    uv run --with buildingregulariser==0.2.5 python ../E036_corner_lines_whu/posthoc.py --b0 <B0_s0.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
from common import MEAN, STD, WHUTiles, b0_probabilities, b3_probabilities, load_polygons  # noqa: E402
from evaluate import B0_FIXED  # noqa: E402
from train import build  # noqa: E402

CACHE = Path("$MANYSEG_BENCH_DATA/cache/whu_aerial")


def regularise(polys):
    import geopandas as gpd
    from buildingregulariser import regularize_geodataframe
    if not polys:
        return []
    r = regularize_geodataframe(gpd.GeoDataFrame(geometry=list(polys)), parallel_threshold=3.0,
                                simplify_tolerance=1.5, allow_circles=False, num_cores=1)
    return [q for g in r.geometry for q in getattr(g, "geoms", [g]) if q.is_valid and q.area > 1]


def boundary_f(gt, pred, tol=1.0, step=0.5):
    """Vertex-free boundary match: points every ``step`` px along each outline; a point counts if the
    other set's outlines lie within ``tol`` px (0.3 m). Returns (tp_pred, n_pred, tp_gt, n_gt)."""
    import shapely
    def pts(polys):
        out = []
        for p in polys:
            for part in getattr(p, "geoms", [p]):
                b = part.exterior
                out.append(shapely.line_interpolate_point(b, np.arange(0, b.length, step)))
        return np.concatenate(out) if out else np.array([])
    pp, pg = pts(pred), pts(gt)
    bg = shapely.union_all([q.exterior for p in gt for q in getattr(p, "geoms", [p])]) if gt else None
    bp = shapely.union_all([q.exterior for p in pred for q in getattr(p, "geoms", [p])]) if pred else None
    tp_p = int((shapely.distance(pp, bg) <= tol).sum()) if len(pp) and bg is not None else 0
    tp_g = int((shapely.distance(pg, bp) <= tol).sum()) if len(pg) and bp is not None else 0
    return tp_p, len(pp), tp_g, len(pg)


def decode_b0(a, p):
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import polygons_from_labels
    t = {k: torch.from_numpy(a[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
    lab = instances_from_outputs(t["extent"], t["boundary"], t["distance"], logits=False,
                                 extent_threshold=p["extent_threshold"], boundary_threshold=p["boundary_threshold"],
                                 seed_threshold=p["seed_threshold"], min_size=16)[0]
    return polygons_from_labels(lab, simplify=p["simplify"], min_area=16), lab


def decode_b3(a, p):
    from manyseg.vectors import polygons_from_corners
    _, lab = decode_b0(a, dict(B0_FIXED, simplify=1.0))
    return polygons_from_corners(lab, a["corner"], a["direction"], corner_threshold=p["corner_threshold"],
                                 big_dev=p["big_dev"], agree=p["agree"], merge_deg=p["merge_deg"],
                                 snap_radius=4.0, simplify=1.0, min_area=16)


def main():
    from manyseg_bench.evaluation.vectors import polygon_metrics
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    here = Path.cwd()
    p3 = json.load(open(here / "runs" / "B3_s0_eval.json"))["params"]
    p0 = json.load(open(args.b0.with_name("B0_s0_eval.json")))["params"]
    dev = torch.device("cuda")
    models = {}
    for arm, path in (("B3", here / "runs" / "B3_s0.pt"), ("B0", args.b0)):
        m = build(arm)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[arm] = m.to(dev).eval()
    ds = WHUTiles(CACHE, "val", "B0", augment=False)
    gt_all = load_polygons(CACHE / "val_poly.npz")
    idx = list(range(0, len(ds), max(1, len(ds) // args.n)))[: args.n]
    acc = {(arm, g): dict(tp=0, n_gt=0, n_pred=0, ciou=0.0, miou=0.0, polis=0.0, snp=0, sng=0)
           for arm in ("B0", "B0+reg", "B3") for g in ("raw", "reg", "dp")}
    keep = {}
    for i in idx:
        x = (ds[i][0][None].to(dev).float() - MEAN.to(dev)) / STD.to(dev)
        gts = {"raw": list(gt_all[i]), "reg": regularise(gt_all[i]),
               # straight-line simplification only: removes raster staircases without forcing right angles
               "dp": [q for q in (g.simplify(1.5) for g in gt_all[i]) if q.geom_type == "Polygon" and q.area > 1]}
        preds = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o0, o3 = models["B0"](x), models["B3"](x)
        preds["B0"] = decode_b0(b0_probabilities(o0, 0), p0)[0]
        preds["B3"] = decode_b3(b3_probabilities(o3, 0), p3)
        preds["B0+reg"] = regularise(preds["B0"])  # the obvious baseline: masks, then buildingregulariser
        for arm, pr in preds.items():
            for g, gp in gts.items():
                m = polygon_metrics(gp, pr, gsd=0.3)
                a = acc[(arm, g)]
                a["tp"] += m["tp"]; a["n_gt"] += m["n_gt"]; a["n_pred"] += m["n_pred"]
                a["ciou"] += m["ciou"] * m["tp"]; a["miou"] += m["miou"] * m["tp"]
                a["polis"] += (m["polis"] * m["tp"]) if m["tp"] else 0.0
                a["snp"] += m.get("sum_n_pred", 0); a["sng"] += m.get("sum_n_gt", 0)
                bf = boundary_f(gp, pr)
                a["bf_tp_p"] = a.get("bf_tp_p", 0) + bf[0]; a["bf_np"] = a.get("bf_np", 0) + bf[1]
                a["bf_tp_g"] = a.get("bf_tp_g", 0) + bf[2]; a["bf_ng"] = a.get("bf_ng", 0) + bf[3]
        if len(keep) < 40:
            keep[i] = (ds[i][0].permute(1, 2, 0).numpy(), gts, preds)
    res = {}
    for (arm, g), a in acc.items():
        tp = max(a["tp"], 1)
        det = 2 * a["tp"] / max(1, a["n_gt"] + a["n_pred"])
        res[f"{arm}_{g}"] = {"ciou_x_f1": a["ciou"] / tp * det, "ciou": a["ciou"] / tp, "det_f1": det,
                             "miou": a["miou"] / tp, "polis_m": a["polis"] / tp, "n_ratio": a["snp"] / max(1, a["sng"]),
                             "boundary_f_1px": 2 * (a["bf_tp_p"] / max(1, a["bf_np"])) * (a["bf_tp_g"] / max(1, a["bf_ng"]))
                             / max(1e-9, a["bf_tp_p"] / max(1, a["bf_np"]) + a["bf_tp_g"] / max(1, a["bf_ng"]))}
    print(json.dumps(res, indent=1))
    if "--no-render" in sys.argv:
        (args.out / "posthoc_dp.json").write_text(json.dumps(dict(res, tiles=len(idx)), indent=1))
        return
    (args.out / "posthoc.json").write_text(json.dumps(dict(res, tiles=len(idx), params_b3=p3, params_b0=p0), indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    busy = sorted(keep, key=lambda k: -len(keep[k][1]["raw"]))[:3]
    fig, axs = plt.subplots(3, 3, figsize=(21, 21))
    names = {"B0": "B0 masks", "B0+reg": "B0 masks + regulariser", "B3": "B3 corners + lines"}
    for row, i in zip(axs, busy):
        img, gts, preds = keep[i]
        big = max(gts["reg"], key=lambda q: q.area) if gts["reg"] else None
        cx, cy = (big.centroid.x, big.centroid.y) if big is not None else (256, 256)
        for ax, arm in zip(row, ("B0", "B0+reg", "B3")):
            ax.imshow(img)
            for q in gts["reg"]:
                c = np.asarray(q.exterior.coords); ax.plot(c[:, 0], c[:, 1], "c-", lw=3, alpha=0.6)
            for q in preds[arm]:
                c = np.asarray(q.exterior.coords); ax.plot(c[:, 0], c[:, 1], "r.-", lw=1.2, ms=4)
            ax.set_title(f"{names[arm]} (red) vs regularised GT (cyan), tile {i}")
            ax.set_xlim(cx - 45, cx + 45); ax.set_ylim(cy + 45, cy - 45)
    plt.savefig(args.out / "posthoc_tiles.png", dpi=55, bbox_inches="tight")


if __name__ == "__main__":
    main()
