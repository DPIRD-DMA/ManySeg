"""E036 gallery: per-building crops of B0, B0 + regulariser and B3 against the raw and regularised GT.
Pages: where B3 beats B0 + regulariser most, where it loses most, and random buildings.

    cd <e036 worktree>/experiments/E030_whu_polygons
    uv run --with buildingregulariser==0.2.5 python ../E036_corner_lines_whu/gallery.py --b0 <B0_s0.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import shapely
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))
import posthoc as H  # noqa: E402
from common import MEAN, STD, WHUTiles, b0_probabilities, b3_probabilities, load_polygons  # noqa: E402
from train import build  # noqa: E402

ARMS = ("B0", "B0+reg", "B3")
NAMES = {"B0": "B0 masks", "B0+reg": "B0 + regulariser", "B3": "B3 corners + lines"}


def best_iou(g, preds):
    best, q = 0.0, None
    for p in preds:
        if p.intersects(g):
            v = g.intersection(p).area / g.union(p).area
            if v > best:
                best, q = v, p
    return best, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "gallery")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    here = Path.cwd()
    p3 = json.load(open(here / "runs" / "B3_s0_eval.json"))["params"]
    p0 = json.load(open(args.b0.with_name("B0_s0_eval.json")))["params"]
    dev = torch.device("cuda")
    models = {}
    for arm, path in (("B3", here / "runs" / "B3_s0.pt"), ("B0", args.b0)):
        m = build(arm)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[arm] = m.to(dev).eval()
    ds = WHUTiles(H.CACHE, "val", "B0", augment=False)
    gt_all = load_polygons(H.CACHE / "val_poly.npz")
    idx = list(range(7, len(ds), max(1, len(ds) // args.n)))[: args.n]  # offset: not the posthoc tiles
    items = []  # (tile, gt_raw, gt_reg, {arm: (iou_reg, poly)}, image)
    imgs = {}
    for i in idx:
        x = (ds[i][0][None].to(dev).float() - MEAN.to(dev)) / STD.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o0, o3 = models["B0"](x), models["B3"](x)
        preds = {"B0": H.decode_b0(b0_probabilities(o0, 0), p0)[0]}
        preds["B3"] = H.decode_b3(b3_probabilities(o3, 0), p3)
        preds["B0+reg"] = H.regularise(preds["B0"])
        raw = [g for g in gt_all[i] if g.area > 150]  # >= ~13 m2: visible at crop scale
        reg = H.regularise(raw)
        imgs[i] = ds[i][0].permute(1, 2, 0).numpy()
        for g in raw:
            gr = max(reg, key=lambda q: q.intersection(g).area, default=None)
            if gr is None:
                continue
            res = {a: best_iou(gr, preds[a]) for a in ARMS}
            items.append((i, g, gr, res))
    rng = np.random.default_rng(0)
    gap = np.array([r[3]["B3"][0] - r[3]["B0+reg"][0] for r in items])
    both = np.array([min(r[3]["B3"][0], r[3]["B0+reg"][0]) > 0.5 for r in items])
    order = np.argsort(gap)
    pages = {
        "b3_better": [items[k] for k in order[::-1] if both[k]][:6],
        "b0reg_better": [items[k] for k in order if both[k]][:6],
        "random": [items[k] for k in rng.choice(len(items), 6, replace=False)],
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, rows in pages.items():
        fig, axs = plt.subplots(len(rows), 4, figsize=(16, 4.1 * len(rows)))
        for ax_row, (i, g, gr, res) in zip(axs, rows):
            x0, y0, x1, y1 = g.buffer(12).bounds
            s = max(x1 - x0, y1 - y0, 40) / 2
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            for k, ax in enumerate(ax_row):
                ax.imshow(imgs[i])
                ax.set_xlim(cx - s, cx + s); ax.set_ylim(cy + s, cy - s); ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    c = np.asarray(g.exterior.coords); ax.plot(c[:, 0], c[:, 1], "y.-", lw=1.5, ms=4)
                    c = np.asarray(gr.exterior.coords); ax.plot(c[:, 0], c[:, 1], "c-", lw=2)
                    ax.set_title(f"tile {i}: GT raw (yellow, {len(g.exterior.coords) - 1} v)\n+ regularised (cyan, {len(gr.exterior.coords) - 1} v)", fontsize=10)
                    continue
                arm = ARMS[k - 1]
                iou, q = res[arm]
                c = np.asarray(gr.exterior.coords); ax.plot(c[:, 0], c[:, 1], "c-", lw=3, alpha=0.6)
                if q is not None:
                    for part in getattr(q, "geoms", [q]):
                        c = np.asarray(part.exterior.coords); ax.plot(c[:, 0], c[:, 1], "r.-", lw=1.5, ms=5)
                nv = sum(len(p.exterior.coords) - 1 for p in getattr(q, "geoms", [q])) if q is not None else 0
                ax.set_title(f"{NAMES[arm]}\nIoU {iou:.2f} vs regularised, {nv} vertices", fontsize=10)
        titles = {"b3_better": "Where B3 beats B0 + regulariser most", "b0reg_better": "Where B0 + regulariser beats B3 most",
                  "random": "Random buildings"}
        fig.suptitle(titles[name] + " (red = prediction, cyan = regularised GT)", fontsize=14, y=1.0)
        plt.tight_layout()
        plt.savefig(args.out / f"{name}.png", dpi=70, bbox_inches="tight")
        plt.close(fig)
    print({k: len(v) for k, v in pages.items()}, "buildings considered", len(items))


if __name__ == "__main__":
    main()
