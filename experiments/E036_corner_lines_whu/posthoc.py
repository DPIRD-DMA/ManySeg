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
           for arm in ("B0", "B3") for g in ("raw", "reg")}
    keep = {}
    for i in idx:
        x = (ds[i][0][None].to(dev).float() - MEAN.to(dev)) / STD.to(dev)
        gts = {"raw": list(gt_all[i]), "reg": regularise(gt_all[i])}
        preds = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o0, o3 = models["B0"](x), models["B3"](x)
        preds["B0"] = decode_b0(b0_probabilities(o0, 0), p0)[0]
        preds["B3"] = decode_b3(b3_probabilities(o3, 0), p3)
        for arm, pr in preds.items():
            for g, gp in gts.items():
                m = polygon_metrics(gp, pr, gsd=0.3)
                a = acc[(arm, g)]
                a["tp"] += m["tp"]; a["n_gt"] += m["n_gt"]; a["n_pred"] += m["n_pred"]
                a["ciou"] += m["ciou"] * m["tp"]; a["miou"] += m["miou"] * m["tp"]
                a["polis"] += (m["polis"] * m["tp"]) if m["tp"] else 0.0
                a["snp"] += m.get("sum_n_pred", 0); a["sng"] += m.get("sum_n_gt", 0)
        if len(keep) < 40:
            keep[i] = (ds[i][0].permute(1, 2, 0).numpy(), gts, preds)
    res = {}
    for (arm, g), a in acc.items():
        tp = max(a["tp"], 1)
        det = 2 * a["tp"] / max(1, a["n_gt"] + a["n_pred"])
        res[f"{arm}_{g}"] = {"ciou_x_f1": a["ciou"] / tp * det, "ciou": a["ciou"] / tp, "det_f1": det,
                             "miou": a["miou"] / tp, "polis_m": a["polis"] / tp, "n_ratio": a["snp"] / max(1, a["sng"])}
    print(json.dumps(res, indent=1))
    (args.out / "posthoc.json").write_text(json.dumps(dict(res, tiles=len(idx), params_b3=p3, params_b0=p0), indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    busy = sorted(keep, key=lambda k: -len(keep[k][1]["raw"]))[:3]
    fig, axs = plt.subplots(3, 2, figsize=(14, 21))
    for row, i in zip(axs, busy):
        img, gts, preds = keep[i]
        for ax, arm in zip(row, ("B0", "B3")):
            ax.imshow(img)
            for q in gts["raw"]:
                c = np.asarray(q.exterior.coords); ax.plot(c[:, 0], c[:, 1], "c-", lw=3, alpha=0.6)
            for q in preds[arm]:
                c = np.asarray(q.exterior.coords); ax.plot(c[:, 0], c[:, 1], "r.-", lw=1.2, ms=4)
            ax.set_title(f"{'B0 masks' if arm == 'B0' else 'B3 corners + lines'} (red) vs WHU GT (cyan), val tile {i}")
            ax.set_xlim(128, 384); ax.set_ylim(384, 128)
    plt.savefig(args.out / "posthoc_tiles.png", dpi=55, bbox_inches="tight")


if __name__ == "__main__":
    main()
