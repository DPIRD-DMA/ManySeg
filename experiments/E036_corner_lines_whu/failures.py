"""E036 B3 failure analysis: for each regularised GT building, B3's final polygon vs its own instance
mask (exact pixel outline) vs B0 + regulariser; failures (B3 IoU < 0.75) are classified:

* missed      - no B3 polygon overlaps the building by > 10%
* merged      - the matched B3 polygon also covers > 30% of another GT building
* split       - two or more B3 polygons each cover > 20% of the building
* decoder     - B3's own instance mask fits (IoU >= 0.75) but the straightened polygon does not
* mask        - B3's instance mask itself is poor (not merged/split/missed): over- or under-covers
Also reruns B3's instances with other thresholds to see whether the borrowed B0 thresholds matter.

    cd <e036 worktree>/experiments/E030_whu_polygons
    uv run --with buildingregulariser==0.2.5 python ../E036_corner_lines_whu/failures.py --b0 <B0_s0.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))
import posthoc as H  # noqa: E402
from common import MEAN, STD, WHUTiles, b0_probabilities, b3_probabilities, load_polygons  # noqa: E402
from train import build  # noqa: E402

FAIL = 0.75
THRESH = [dict(extent_threshold=e, boundary_threshold=b, seed_threshold=s, simplify=1.0)
          for e in (0.4, 0.5, 0.6) for b in (0.3, 0.5, 0.7) for s in (0.15, 0.3)]


def iou(a, b):
    i = a.intersection(b).area
    return i / (a.area + b.area - i) if i > 0 else 0.0


def classify(g, others, preds, mask_polys):
    """Category and diagnostics for one GT building ``g``."""
    cover = [(p, g.intersection(p).area / g.area) for p in preds if p.intersects(g)]
    cover = [(p, c) for p, c in cover if c > 0]
    best = max(((iou(g, p), p) for p, _ in cover), default=(0.0, None), key=lambda t: t[0])
    m_iou = max((iou(g, p) for p in mask_polys if p.intersects(g)), default=0.0)
    d = {"b3_iou": best[0], "mask_iou": m_iou}
    if best[0] >= FAIL:
        return "ok", d
    if not cover or max(c for _, c in cover) < 0.1:
        return "missed", d
    p = best[1]
    if any(o.intersects(p) and o.intersection(p).area / o.area > 0.3 for o in others):
        return "merged", d
    if sum(1 for _, c in cover if c > 0.2) >= 2:
        return "split", d
    if m_iou >= FAIL:
        return "decoder", d
    d["area_ratio"] = p.area / g.area
    return "mask", d


def main():
    from manyseg.vectors import polygons_from_labels
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "failures")
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
    idx = list(range(3, len(ds), max(1, len(ds) // args.n)))[: args.n]
    cats, cats_b0 = Counter(), Counter()
    loss = defaultdict(float)  # sum of (1 - IoU) per category: share of the total shortfall
    examples = defaultdict(list)
    thr_scores = np.zeros(len(THRESH))
    border_fail = Counter()
    for i in idx:
        x = (ds[i][0][None].to(dev).float() - MEAN.to(dev)) / STD.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o0, o3 = models["B0"](x), models["B3"](x)
        a3 = b3_probabilities(o3, 0)
        b3 = H.decode_b3(a3, p3)
        _, lab3 = H.decode_b0(a3, dict(H.B0_FIXED, simplify=1.0))
        mask3 = polygons_from_labels(lab3, simplify=0.0, min_area=16)
        b0r = H.regularise(H.decode_b0(b0_probabilities(o0, 0), p0)[0])
        raw = [g for g in gt_all[i] if g.area > 150]
        gts = H.regularise(raw)
        on_border = [min(g.bounds[0], g.bounds[1]) < 1 or max(g.bounds[2], g.bounds[3]) > 511 for g in gts]
        # thresholds: mean best-IoU of B3's *instance mask* polygons per setting
        for k, p in enumerate(THRESH):
            _, lab = H.decode_b0(a3, p)
            mp = polygons_from_labels(lab, simplify=0.0, min_area=16)
            thr_scores[k] += sum(max((iou(g, q) for q in mp if q.intersects(g)), default=0.0) for g in gts)
        for j, g in enumerate(gts):
            others = gts[:j] + gts[j + 1:]
            c, d = classify(g, others, b3, mask3)
            c0, _ = classify(g, others, b0r, b0r)
            cats[c] += 1
            cats_b0[c0 if c0 != "decoder" else "mask"] += 1
            loss[c] += 1 - d["b3_iou"]
            if c != "ok" and on_border[j]:
                border_fail[c] += 1
            if c != "ok" and len(examples[c]) < 6 and (len(examples[c]) == 0 or examples[c][-1][0] != i):
                examples[c].append((i, g, d, b3, mask3, b0r, a3["corner"], ds[i][0].permute(1, 2, 0).numpy()))
    n = sum(cats.values())
    total = sum(loss.values())
    summary = {"buildings": n, "tiles": len(idx), "fail_below": FAIL,
               "b3": {c: {"n": cats[c], "share_of_buildings": cats[c] / n, "share_of_iou_shortfall": loss[c] / total}
                      for c in cats},
               "b0_plus_regulariser": {c: cats_b0[c] / n for c in cats_b0},
               "failures_touching_tile_border": dict(border_fail),
               "threshold_sweep_mask_iou": sorted(
                   ({**{k: v for k, v in p.items() if k != "simplify"}, "mean_iou": s / n} for p, s in zip(THRESH, thr_scores)),
                   key=lambda r: -r["mean_iou"])[:5],
               "registered_thresholds": H.B0_FIXED}
    print(json.dumps(summary, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c, rows in examples.items():
        fig, axs = plt.subplots(len(rows), 4, figsize=(16, 4.1 * len(rows)), squeeze=False)
        for ax_row, (i, g, d, b3, mask3, b0r, corner, img) in zip(axs, rows):
            x0, y0, x1, y1 = g.buffer(15).bounds
            s = max(x1 - x0, y1 - y0, 50) / 2
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            panels = (("image + regularised GT", None), ("B3 instance mask (exact)", mask3),
                      ("B3 corner map + final polygon", b3), ("B0 + regulariser", b0r))
            for k, (ax, (title, polys)) in enumerate(zip(ax_row, panels)):
                ax.imshow(img)
                if k == 2:
                    ax.imshow(np.ma.masked_less(corner, 0.15), cmap="autumn", alpha=0.7, vmin=0, vmax=1)
                ax.set_xlim(cx - s, cx + s); ax.set_ylim(cy + s, cy - s); ax.set_xticks([]); ax.set_yticks([])
                cg = np.asarray(g.exterior.coords)
                ax.plot(cg[:, 0], cg[:, 1], "c-", lw=3 if k else 2, alpha=0.6 if k else 1)
                for q in polys or []:
                    if q.intersects(g.buffer(s)):
                        for part in getattr(q, "geoms", [q]):
                            cq = np.asarray(part.exterior.coords)
                            ax.plot(cq[:, 0], cq[:, 1], "r.-", lw=1.4, ms=4)
                extra = {1: f" IoU {d['mask_iou']:.2f}", 2: f" IoU {d['b3_iou']:.2f}"}.get(k, f" (tile {i})" if k == 0 else "")
                ax.set_title(title + extra, fontsize=10)
        fig.suptitle(f"B3 failures: {c} (red = prediction, cyan = regularised GT; orange = corner probability)", fontsize=13, y=1.0)
        plt.tight_layout()
        plt.savefig(args.out / f"{c}.png", dpi=65, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
