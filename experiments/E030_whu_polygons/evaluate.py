"""E030: tune each run's decoder on val (C-IoU x detection F1), freeze it, score test once.

    uv run python experiments/E030_whu_polygons/evaluate.py --run B2_s0 --cache "$E030_CACHE"
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

from common import DeviceLoader, WHUTiles, b0_probabilities, b2_probabilities, load_polygons
from train import build

HERE = Path(__file__).resolve().parent


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


GRIDS = {
    "B0": grid(extent_threshold=[0.4, 0.5, 0.6], boundary_threshold=[0.3, 0.5],
               seed_threshold=[0.15, 0.3], simplify=[1.0, 2.0]),
    "B2": grid(vertex_threshold=[0.1, 0.2, 0.3], link_radius=[6.0, 10.0], gap=[0.0, 25.0],
               simplify=[1.0, 1.5]),
}
SUMS = ("tp", "n_gt", "n_pred", "ciou", "miou", "polis", "sum_n_pred", "sum_n_gt", "decode_s",
        "failed")

_GT: list = []


def _init(gt):
    global _GT
    _GT = gt
    torch.set_num_threads(1)


def decode(arm: str, a: dict, p: dict, cache: dict) -> list:
    if arm == "B0":
        from manyseg.instances import instances_from_outputs
        from manyseg.vectors import polygons_from_labels

        key = (p["extent_threshold"], p["boundary_threshold"], p["seed_threshold"])
        if key not in cache:
            t = {k: torch.from_numpy(a[k].astype(np.float32))[None, None] for k in a}
            cache[key] = instances_from_outputs(
                t["extent"], t["boundary"], t["distance"], logits=False,
                extent_threshold=key[0], boundary_threshold=key[1], seed_threshold=key[2],
                min_size=16)[0]
        return polygons_from_labels(cache[key], simplify=p["simplify"], min_area=16)
    from manyseg.vectors import polygons_from_tensor

    return polygons_from_tensor(a["vertex"].astype(np.float32), a["edge"].astype(np.float32),
                                a["vector"].astype(np.float32), a["extent"].astype(np.float32),
                                vertex_threshold=p["vertex_threshold"], link_radius=p["link_radius"],
                                gap=p["gap"], simplify=p["simplify"], edge_threshold=0.5, kernel=3)


def work(item):
    from manyseg_bench.evaluation.vectors import polygon_metrics

    arm, i, a, params = item
    cache: dict = {}
    rows = []
    for p in params:
        t0 = time.perf_counter()
        failed = 0
        try:
            pred = decode(arm, a, p, cache)
        except MemoryError:  # counted, never silent; the run's scope caps total memory
            pred, failed = [], 1
        dt = time.perf_counter() - t0
        m = polygon_metrics(_GT[i], pred, gsd=0.3)
        tp = m["tp"]
        rows.append({"tp": tp, "n_gt": m["n_gt"], "n_pred": m["n_pred"],
                     "ciou": m["ciou"] * tp, "miou": m["miou"] * tp,
                     "polis": (m["polis"] * tp) if tp else 0.0,
                     "sum_n_pred": m.get("sum_n_pred", 0), "sum_n_gt": m.get("sum_n_gt", 0),
                     "decode_s": dt, "failed": failed})
    return rows


def summarise(acc: dict) -> dict:
    tp = max(acc["tp"], 1)
    det = 2 * acc["tp"] / max(1, acc["n_gt"] + acc["n_pred"])
    ciou = acc["ciou"] / tp
    return {"ciou": ciou, "det_f1": det, "ciou_x_f1": ciou * det, "miou": acc["miou"] / tp,
            "recall": acc["tp"] / max(1, acc["n_gt"]),  # GT buildings recovered as a matched closed polygon
            "polis_m": acc["polis"] / tp, "n_ratio": acc["sum_n_pred"] / max(1, acc["sum_n_gt"]),
            "tp": acc["tp"], "n_gt": acc["n_gt"], "n_pred": acc["n_pred"],
            "decode_s_per_tile": acc["decode_s"] / max(1, acc["tiles"]),
            "failed_tiles": int(acc["failed"])}


@torch.no_grad()
def run_split(model, arm, ds, gt, params, device, procs, chunk=48, limit=None):
    n = min(len(ds), limit or len(ds))
    accs = [dict.fromkeys(SUMS, 0.0) | {"tiles": 0} for _ in params]
    inter = union = 0
    prob = b0_probabilities if arm == "B0" else b2_probabilities
    dl = DeviceLoader(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=16,
                                                  num_workers=4), device)
    with Pool(procs, initializer=_init, initargs=(gt,)) as pool:
        idx, items = 0, []
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            pr, g = out["extent"][:, 0] > 0, (y > 0)  # y: instance ids (B0 loader) for both arms
            inter += int((pr & g).sum())
            union += int((pr | g).sum())
            for b in range(x.shape[0]):
                a = {k: v.astype(np.float16) for k, v in prob(out, b).items()}
                items.append((arm, idx, a, params))
                idx += 1
            if len(items) >= chunk:
                for rows in pool.map(work, items):
                    for acc, r in zip(accs, rows):
                        for k in SUMS:
                            acc[k] += r[k]
                        acc["tiles"] += 1
                items = []
        for rows in pool.map(work, items):
            for acc, r in zip(accs, rows):
                for k in SUMS:
                    acc[k] += r[k]
                acc["tiles"] += 1
    return [summarise(a) for a in accs], inter / max(1, union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. B2_s0")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N tiles")
    args = ap.parse_args()
    arm = args.run[:2]  # decoder: "B0" or "B2" (E032's "B2w" / "B2e" decode as B2)
    run_arm = args.run.split("_")[0]  # experimental arm, kept in the results
    device = torch.device("cuda")
    model = build(arm)
    model.load_state_dict(torch.load(HERE / "runs" / f"{args.run}.pt", map_location="cpu"))
    model.to(device).eval()
    t0 = time.time()
    val = WHUTiles(args.cache, "val", "B0", augment=False)  # B0 loader: image + instance ids
    val_gt = load_polygons(args.cache / "val_poly.npz")
    val_rows, val_iou = run_split(model, arm, val, val_gt, GRIDS[arm], device, args.procs, limit=args.limit)
    best = max(range(len(val_rows)), key=lambda k: val_rows[k]["ciou_x_f1"])
    test = WHUTiles(args.cache, "test", "B0", augment=False)
    test_gt = load_polygons(args.cache / "test_poly.npz")
    (test_row,), test_iou = run_split(model, arm, test, test_gt, [GRIDS[arm][best]], device, args.procs,
                                      limit=args.limit)
    res = {"run": args.run, "arm": run_arm, "decoder_arm": arm, "params": GRIDS[arm][best], "val": val_rows[best],
           "val_extent_iou": val_iou, "val_grid": [dict(p, **r) for p, r in zip(GRIDS[arm], val_rows)],
           "test": test_row, "test_extent_iou": test_iou, "wall_s": time.time() - t0, "limit": args.limit}
    (HERE / "runs" / f"{args.run}_eval{'_smoke' if args.limit else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("run", "params", "val", "test", "test_extent_iou")}, indent=1))


if __name__ == "__main__":
    main()
