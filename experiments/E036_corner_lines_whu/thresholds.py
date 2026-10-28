"""E036 threshold check (Codex review #1): B3's instance thresholds, registered setting included, scored
with one-to-one detection F1, C-IoU x F1 and boundary F against DP GT (E038's metric code); 150 val tiles.

    cd <e036 worktree>/experiments/E030_whu_polygons
    uv run --with buildingregulariser==0.2.5 python <repo>/experiments/E036_corner_lines_whu/thresholds.py
"""
import itertools
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E038_fair_footprint_metrics"))
import evaluate38 as V  # noqa: E402
from common import MEAN, STD, WHUTiles, b3_probabilities, load_polygons  # noqa: E402
from train import build  # noqa: E402

SETTINGS = [dict(extent_threshold=e, boundary_threshold=b, seed_threshold=s)
            for e, b, s in itertools.product((0.4, 0.5, 0.6), (0.3, 0.5, 0.7), (0.15, 0.3))]
P3 = json.load(open("runs/B3_s0_eval.json"))["params"]


def work(item):
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import polygons_from_corners
    torch.set_num_threads(1)
    a, gt = item
    t = {k: torch.from_numpy(a[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
    rows = []
    for s in SETTINGS:
        lab = instances_from_outputs(t["extent"], t["boundary"], t["distance"], logits=False, min_size=16, **s)[0]
        pred = polygons_from_corners(lab, a["corner"].astype(np.float32), a["direction"].astype(np.float32),
                                     corner_threshold=P3["corner_threshold"], big_dev=P3["big_dev"], agree=P3["agree"],
                                     merge_deg=P3["merge_deg"], snap_radius=4.0, simplify=1.0, min_area=16)
        rows.append(V.metrics(gt, pred))
    return rows


def main():
    m = build("B3"); m.load_state_dict(torch.load("runs/B3_s0.pt", map_location="cpu")); m = m.cuda().eval()
    ds = WHUTiles(V.CACHE, "val", "B0", augment=False)
    gts = load_polygons(V.CACHE / "val_poly.npz")
    idx = list(range(0, len(ds), max(1, len(ds) // 150)))[:150]
    accs = [dict.fromkeys(V.SUMS, 0.0) for _ in SETTINGS]
    with Pool(4) as pool:
        for k in range(0, len(idx), 16):
            items = []
            for i in idx[k:k + 16]:
                x = (ds[i][0][None].cuda().float() - MEAN.cuda()) / STD.cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    a = {kk: v.astype(np.float16) for kk, v in b3_probabilities(m(x), 0).items()}
                items.append((a, V.gt_dp(gts[i])))
            for rows in pool.map(work, items):
                for acc, r in zip(accs, rows):
                    for kk in V.SUMS:
                        acc[kk] += r[kk]
    out = [dict(s, **{kk: round(v, 4) for kk, v in V.summarise(a).items() if kk in ("boundary_f", "det_f1", "ciou_x_f1", "miou", "n_pred")})
           for s, a in zip(SETTINGS, accs)]
    out.sort(key=lambda r: -r["boundary_f"])
    reg = next(r for r in out if (r["extent_threshold"], r["boundary_threshold"], r["seed_threshold"]) == (0.6, 0.5, 0.15))
    res = {"tiles": len(idx), "registered": reg, "rank_of_registered": out.index(reg) + 1, "all": out}
    print(json.dumps({k: v for k, v in res.items() if k != "all"}, indent=1)); print(json.dumps(out[:3], indent=1))
    Path(__file__).with_name("thresholds.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
