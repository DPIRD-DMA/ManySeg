"""E030b: per-building fallback (B2F) on the trained E030 models, one seed at a time.

B2 model -> graph polygons and the extent mask (single extent source); B0 model -> instance
polygons for buildings the graph does not cover (recall >= cov and precision >= 0.9, as E028b).
Decoder parameters for B0 / B2 are each run's tuned values (runs/B?_s{seed}_eval.json); cov is
tuned on val by C-IoU x detection F1, then test is decoded once.

    uv run python fallback.py --seed 0 --cache "$E030_CACHE"
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage

from common import DeviceLoader, WHUTiles, b0_probabilities, b2_probabilities, load_polygons
from evaluate import SUMS, decode, summarise
from train import build

HERE = Path(__file__).resolve().parent
COVS = (0.7, 0.8, 0.9, 0.95)
Q_MIN = 0.9
IDENT = Affine(1, 0, 0, 0, 1, 0)
_GT: list = []


def _init(gt):
    global _GT
    _GT = gt
    torch.set_num_threads(1)


def assign(faces, comp, n):
    """Owner = the component a face overlaps most; per component recall and precision of the
    faces it owns (same as experiments/E028_vector_oracle/e028b.py)."""
    if not faces or n == 0:
        return [], np.zeros(n + 1), np.zeros(n + 1)
    lab = rasterize([(f, i + 1) for i, f in enumerate(faces)], out_shape=comp.shape, transform=IDENT, dtype="int32")
    pair = np.bincount(lab.ravel().astype(np.int64) * (n + 1) + comp.ravel(),
                       minlength=(len(faces) + 1) * (n + 1)).reshape(len(faces) + 1, n + 1)[1:]
    owner = [int(np.argmax(r[1:]) + 1) if r[1:].max() > 0 else 0 for r in pair]
    inter, area = np.zeros(n + 1), np.zeros(n + 1)
    for i, o in enumerate(owner):
        if o:
            inter[o] += pair[i, o]
            area[o] += pair[i].sum()
    size = np.bincount(comp.ravel(), minlength=n + 1)
    return owner, inter / np.maximum(size, 1), inter / np.maximum(area, 1)


def work(item):
    from manyseg_bench.evaluation.vectors import polygon_metrics

    i, a0, a2, p0, p2 = item
    t0 = time.perf_counter()
    comp, n = ndimage.label(a2["extent"].astype(np.float32) > 0.5)
    f2 = decode("B2", a2, p2, {})
    own2, R, Q = assign(f2, comp, n)
    f0 = decode("B0", a0, p0, {})
    own0, _, _ = assign(f0, comp, n)
    dt = time.perf_counter() - t0
    rows = []
    for cov in COVS:
        use2 = (R >= cov) & (Q >= Q_MIN)
        pred = [f for f, o in zip(f2, own2) if o and use2[o]] + [f for f, o in zip(f0, own0) if o and not use2[o]]
        m = polygon_metrics(_GT[i], pred, gsd=0.3)
        tp = m["tp"]
        rows.append({"tp": tp, "n_gt": m["n_gt"], "n_pred": m["n_pred"], "ciou": m["ciou"] * tp,
                     "miou": m["miou"] * tp, "polis": (m["polis"] * tp) if tp else 0.0,
                     "sum_n_pred": m["sum_n_pred"], "sum_n_gt": m["sum_n_gt"], "decode_s": dt, "failed": 0,
                     "share_graph": float(use2[1:].mean()) if n else 0.0})
    return rows


@torch.no_grad()
def run_split(models, params, cache, split, procs, limit=None):
    ds = WHUTiles(cache, split, "B0", augment=False)
    gt = load_polygons(cache / f"{split}_poly.npz")
    n = min(len(ds), limit or len(ds))
    dl = DeviceLoader(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=16, num_workers=4),
                      torch.device("cuda"))
    accs = [dict.fromkeys(SUMS, 0.0) | {"tiles": 0, "share_graph": 0.0} for _ in COVS]
    with Pool(procs, initializer=_init, initargs=(gt,)) as pool:
        idx, items = 0, []

        def flush():
            for rows in pool.map(work, items):
                for acc, r in zip(accs, rows):
                    for k in SUMS:
                        acc[k] += r[k]
                    acc["share_graph"] += r["share_graph"]
                    acc["tiles"] += 1

        for x, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o0, o2 = models["B0"](x), models["B2"](x)
            for b in range(x.shape[0]):
                a0 = {k: v.astype(np.float16) for k, v in b0_probabilities(o0, b).items()}
                a2 = {k: v.astype(np.float16) for k, v in b2_probabilities(o2, b).items()}
                items.append((idx, a0, a2, params["B0"], params["B2"]))
                idx += 1
            if len(items) >= 48:
                flush()
                items = []
        flush()
    out = []
    for a in accs:
        s = summarise(a)
        s["share_graph"] = a["share_graph"] / max(1, a["tiles"])
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    models, params = {}, {}
    for arm in ("B0", "B2"):
        run = f"{arm}_s{args.seed}"
        m = build(arm)
        m.load_state_dict(torch.load(HERE / "runs" / f"{run}.pt", map_location="cpu"))
        models[arm] = m.cuda().eval()
        params[arm] = json.loads((HERE / "runs" / f"{run}_eval.json").read_text())["params"]
    val = run_split(models, params, args.cache, "val", args.procs, args.limit)
    best = max(range(len(COVS)), key=lambda k: val[k]["ciou_x_f1"])
    test = run_split(models, params, args.cache, "test", args.procs, args.limit)[best]  # all covs decoded; best reported
    res = {"run": f"B2F_s{args.seed}", "cov": COVS[best], "val": val[best],
           "val_grid": [dict(cov=c, **v) for c, v in zip(COVS, val)], "test": test, "params": params, "limit": args.limit}
    (HERE / "runs" / f"B2F_s{args.seed}_eval{'_smoke' if args.limit else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("run", "cov", "val", "test")}, indent=1))


if __name__ == "__main__":
    main()
