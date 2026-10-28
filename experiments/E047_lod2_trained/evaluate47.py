"""E047 evaluation (README): A0 (plane segmentation of the input nDSM), F0 and F3 (trained faces)
on val (tuning) and test (once). Planes from the input nDSM.

    uv run python evaluate47.py --model runs/B3_s0.pt --seed 0 [--limit N] [--procs 6]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for d in ("E030_whu_polygons", "E031_lod2_oracle", "E043_roof_mesh"):
    sys.path.insert(0, str(HERE.parent / d))
sys.path.insert(0, str(HERE))
import e031 as E  # noqa: E402
import e043  # noqa: E402
from common import b3_probabilities  # noqa: E402
from train47 import CACHE, RoofLoader, RoofTiles, build  # noqa: E402

GRID_A0 = [dict(smooth=s, tg=g, tz=0.5, min_px=m, simplify=si)
           for s, g, m, si in itertools.product((0.5, 1.5), (0.03, 0.06, 0.12), (30, 100), (1.0, 2.0))]
THR = [dict(extent_threshold=e, boundary_threshold=b, seed_threshold=s)
       for e, b, s in itertools.product((0.4, 0.6), (0.3, 0.5), (0.15, 0.3))]
GRID = {"F0": [dict(simplify=s) for s in (0.5, 1.0, 2.0)],
        "F3": [dict(corner_threshold=c, big_dev=b, agree=a, merge_deg=m)
               for c, b, a, m in itertools.product((0.3, 0.5), (4.0, 8.0), (0.4, 0.6), (8.0, 15.0))]}
MASK_M = 2.5  # A0's building mask: nDSM above this (m)
SUMS = ("tp", "n_gt", "n_pred", "sq_err", "n_px", "corner_tp", "corner_gt", "corner_pred", "ciou_tp",
        "normal_sum", "n_normal")


def _tiles(split):
    ts = pickle.loads((CACHE / f"{split}_tiles.pkl").read_bytes())
    for t in ts:
        t.height, t.mask = E.height_map(t.faces, t.planes)
    return ts


def _row(t, faces, planes):
    m = E.metrics(t, faces, planes)
    n = int(t.mask.sum())
    return dict(tp=m["tp"], n_gt=m["n_gt"], n_pred=m["n_pred"], sq_err=m["h_rmse"] ** 2 * n, n_px=n,
                corner_tp=m["corner_tp"], corner_gt=m["corner_gt"], corner_pred=m["corner_pred"],
                ciou_tp=m["ciou"] * m["tp"], n_normal=m["n_normal"],
                normal_sum=m["normal_err_deg"] * m["n_normal"] if m["n_normal"] else 0.0)


def summarise(acc):
    f1 = 2 * acc["tp"] / max(1, acc["n_gt"] + acc["n_pred"])
    rmse = float(np.sqrt(acc["sq_err"] / max(1, acc["n_px"])))
    return {"face_f1": f1, "h_rmse": rmse, "score": f1 * (1 - min(1.0, rmse)),
            "corner_f1": 2 * acc["corner_tp"] / max(1, acc["corner_gt"] + acc["corner_pred"]),
            "ciou": acc["ciou_tp"] / max(1, acc["tp"]),
            "normal_err_deg": acc["normal_sum"] / acc["n_normal"] if acc["n_normal"] else float("nan"), "tp": acc["tp"], "n_gt": acc["n_gt"], "n_pred": acc["n_pred"]}


def work(item):
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import polygons_from_corners, polygons_from_labels

    torch.set_num_threads(1)
    arm, t, nd, prob, thr_list, settings = item
    a = {"height": nd.astype(np.float32), "mask": (nd > MASK_M).astype(np.float32)}
    rows = []
    if arm == "A0":
        for p in settings:
            rows.append(_row(t, *E.decode_a0(a, p)))
        return rows
    if arm == "A0m":  # diagnostic: A0's plane segmentation inside the learned roof extent
        for thr in thr_list:
            am = dict(a, mask=(prob["extent"].astype(np.float32) > thr["extent_threshold"]).astype(np.float32))
            for p in settings:
                rows.append(_row(t, *E.decode_a0(am, p)))
        return rows
    for thr in thr_list:
        tt = {k: torch.from_numpy(prob[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
        lab = instances_from_outputs(tt["extent"], tt["boundary"], tt["distance"], logits=False, min_size=16, **thr)[0]
        for p in settings:
            if arm == "F0":
                faces = polygons_from_labels(lab, simplify=p["simplify"], min_area=16)
            else:
                faces = polygons_from_corners(lab, prob["corner"].astype(np.float32), prob["direction"].astype(np.float32),
                                              corner_threshold=p["corner_threshold"], big_dev=p["big_dev"],
                                              agree=p["agree"], merge_deg=p["merge_deg"], snap_radius=4.0,
                                              simplify=1.0, min_area=16)
            faces = [f for f in faces if f.geom_type == "Polygon"]
            rows.append(_row(t, faces, e043.planes(faces, a, "robust")))
    return rows


@torch.no_grad()
def run(model, arm, split, thr_list, settings, procs, limit):
    tiles = _tiles(split)
    n = min(len(tiles), limit or len(tiles))
    ds = torch.utils.data.Subset(RoofTiles(split, augment=False, targets=False), range(n))
    nds = np.load(CACHE / f"{split}_ndsm.npy", mmap_mode="r")
    per_area = {}
    k = len(settings) * (1 if arm == "A0" else len(thr_list))
    dev = torch.device("cuda")
    idx = 0
    with Pool(procs) as pool:
        if arm == "A0":
            items = [(arm, tiles[i], np.asarray(nds[i]), None, None, settings) for i in range(n)]
            results = list(zip(range(n), pool.map(work, items)))
        else:
            dl = RoofLoader(torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4), dev)
            results = []
            for x, _ in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                items = []
                for b in range(x.shape[0]):
                    pr = {kk: v.astype(np.float16) for kk, v in b3_probabilities(out, b).items()}
                    items.append((arm, tiles[idx], np.asarray(nds[idx]), pr, thr_list, settings))
                    idx += 1
                results += list(zip(range(idx - len(items), idx), pool.map(work, items)))
    for i, rows in results:
        accs = per_area.setdefault(tiles[i].area, [dict.fromkeys(SUMS, 0.0) for _ in range(k)])
        for acc, r in zip(accs, rows):
            for s in SUMS:
                acc[s] += r[s]
    all_ = [dict.fromkeys(SUMS, 0.0) for _ in range(k)]
    for accs in per_area.values():
        for a, acc in zip(all_, accs):
            for s in SUMS:
                a[s] += acc[s]
    return [summarise(a) for a in all_], {ar: [summarise(a) for a in accs] for ar, accs in per_area.items()}


def decide(test):
    """README rule; viability per single trained arm (Codex E047 #1)."""
    areas = ("leidscherijn", "staphorst")
    out = {}
    for arm in ("F0", "F3"):
        g = {ar: test[arm]["areas"][ar]["face_f1"] - test["A0"]["areas"][ar]["face_f1"] for ar in areas}
        h = {ar: test[arm]["areas"][ar]["h_rmse"] - test["A0"]["areas"][ar]["h_rmse"] for ar in areas}
        out[arm] = {"face_f1_gain_vs_a0": g, "h_rmse_vs_a0": h,
                    "viable": all(g[a] >= 0.10 and h[a] <= 0.10 for a in areas)}
    d3 = {ar: test["F3"]["areas"][ar]["face_f1"] - test["F0"]["areas"][ar]["face_f1"] for ar in areas}
    dc = {ar: test["F3"]["areas"][ar]["corner_f1"] - test["F0"]["areas"][ar]["corner_f1"] for ar in areas}
    margins = [abs(v - 0.10) for arm in ("F0", "F3") for v in out[arm]["face_f1_gain_vs_a0"].values()]
    margins += [abs(v - 0.10) for arm in ("F0", "F3") for v in out[arm]["h_rmse_vs_a0"].values()]
    margins += [abs(v - 0.02) for v in d3.values()]
    return {"arms": out, "f3_minus_f0_face_f1": d3, "f3_minus_f0_corner_f1": dc,
            "trained_faces_viable": out["F0"]["viable"] or out["F3"]["viable"],
            "viable_arms": [a for a in ("F0", "F3") if out[a]["viable"]],
            "f3_preferred": all(d3[a] >= 0.02 and dc[a] >= 0 for a in areas),
            "second_seed_needed": min(margins) < 0.02}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = HERE / "runs" / f"eval_{args.model.stem}{'_limit' if args.limit else ''}.json"
    sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    ledger = HERE / "runs" / "test_ledger.json"  # test once per checkpoint (Codex E047 #4)
    led = json.loads(ledger.read_text()) if ledger.exists() else {}
    if out.exists() or (not args.limit and sha in led):
        raise SystemExit(f"test already evaluated for this checkpoint / seed ({out}, {led.get(sha)})")
    m = build()
    m.load_state_dict(torch.load(args.model, map_location="cpu"))
    m = m.to("cuda").eval()
    res = {"seed": args.seed, "model": str(args.model), "sha256": sha}
    t0 = time.time()
    sel = {}
    v, _ = run(None, "A0", "val", None, GRID_A0, args.procs, args.limit)
    k = max(range(len(v)), key=lambda j: v[j]["score"])
    sel["A0"] = {"params": GRID_A0[k], "val": v[k]}
    for arm in ("F0", "F3"):
        v1, _ = run(m, arm, "val", THR, GRID[arm][:1], args.procs, args.limit)
        thr = THR[max(range(len(THR)), key=lambda j: v1[j]["score"])]
        v2, _ = run(m, arm, "val", [thr], GRID[arm], args.procs, args.limit)
        k = max(range(len(v2)), key=lambda j: v2[j]["score"])
        sel[arm] = {"thr": thr, "params": GRID[arm][k], "val": v2[k]}
        print(arm, "val", {x: round(y, 3) for x, y in v2[k].items() if isinstance(y, float)}, f"{time.time() - t0:.0f}s", flush=True)
    v, _ = run(m, "A0m", "val", [sel["F0"]["thr"]], GRID_A0, args.procs, args.limit)
    k = max(range(len(v)), key=lambda j: v[j]["score"])
    sel["A0m"] = {"thr": sel["F0"]["thr"], "params": GRID_A0[k], "val": v[k]}
    res["selected"] = sel
    test = {}
    for arm in ("A0", "F0", "F3", "A0m"):  # test once, after all tuning
        s = sel[arm]
        allm, areas = run(m, arm, "test", [s.get("thr")], [s["params"]], args.procs, args.limit)
        test[arm] = {"all": allm[0], "areas": {a: v[0] for a, v in areas.items()}}
        print(arm, "test", {a: {x: round(y, 3) for x, y in v[0].items() if isinstance(y, float)} for a, v in areas.items()}, flush=True)
    res["test"] = test
    res["decision"] = decide(test)
    out.write_text(json.dumps(res, indent=1))
    if not args.limit:
        led[sha] = str(out)
        ledger.write_text(json.dumps(led, indent=1))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
