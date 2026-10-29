"""E049 evaluation (README): H0 / H1 (RGB only, predicted heights) against R (E047b RGB + nDSM),
all with F3 faces + planar join; tuned on val, test once.

    uv run python evaluate49.py --h0 runs/H0_s0.pt --h1 runs/H1_s0.pt [--procs 6] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E47 = HERE.parent / "E047_lod2_trained"
for d in ("E030_whu_polygons", "E031_lod2_oracle", "E043_roof_mesh"):
    sys.path.insert(0, str(HERE.parent / d))
sys.path[:0] = [str(E47), str(HERE)]
import e043  # noqa: E402
import evaluate47 as V  # noqa: E402
import train47 as T  # noqa: E402
import train49 as T9  # noqa: E402
from common import b3_probabilities  # noqa: E402

THR, GRID3, SUMS = V.THR, V.GRID["F3"], V.SUMS
PLANAR_R = 2.5  # E048b
AREAS = ("leidscherijn", "staphorst")
HSUMS = ("roof_abs", "roof_n", "off_abs", "off_n")


def work(item):
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import planar_faces, polygons_from_corners

    torch.set_num_threads(1)
    t, prob, nd, thr_list, settings, modes, planar = item
    rows = []
    for thr in thr_list:
        tt = {k: torch.from_numpy(prob[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
        lab = instances_from_outputs(tt["extent"], tt["boundary"], tt["distance"], logits=False, min_size=16, **thr)[0]
        for p in settings:
            faces = polygons_from_corners(lab, prob["corner"].astype(np.float32), prob["direction"].astype(np.float32),
                                          snap_radius=4.0, simplify=1.0, min_area=16, **p)
            faces = [f for f in faces if f.geom_type == "Polygon"]
            if planar:
                faces = [f for f in planar_faces(faces, tolerance=PLANAR_R) if f.geom_type == "Polygon"]
            for mode in modes:
                if mode == "pred":
                    P = e043.planes(faces, {"height": prob["height"].astype(np.float32)}, "robust")
                elif mode == "slope":
                    P = e043.planes(faces, {"height": prob["height"].astype(np.float32),
                                            "slope": prob["slope"].astype(np.float32)}, "slope")
                else:  # "ndsm": the real AHN nDSM (R, D)
                    P = e043.planes(faces, {"height": nd.astype(np.float32)}, "robust")
                rows.append(V._row(t, faces, P))
    h = prob["height"].astype(np.float32) if "height" in prob else nd.astype(np.float32)
    hs = dict(roof_abs=float(np.abs(h - t.height)[t.mask].sum()), roof_n=int(t.mask.sum()),
              off_abs=float(np.abs(h)[~t.mask].sum()), off_n=int((~t.mask).sum()))
    return rows, hs


@torch.no_grad()
def predict(model, kind, split, n):
    dev = torch.device("cuda")
    if kind == "R":
        ds = T.RoofTiles(split, augment=False, targets=False)
        norm = lambda x: torch.cat([(x[:, :3] - T.MEAN.to(dev)) / T.STD.to(dev), x[:, 3:] / T.NDSM_SCALE], 1)
    else:
        ds = T9.RGBRoofTiles(split, augment=False, targets=False)
        norm = lambda x: (x - T.MEAN.to(dev)) / T.STD.to(dev)
    for i in range(n):
        x = norm(ds[i][0][None].to(dev).float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        p = b3_probabilities(out, 0) if kind == "R" else T9.probabilities(out, 0)
        yield i, {k: v.astype(np.float16) for k, v in p.items()}


def run(model, kind, split, thr_list, settings, modes, planar, procs, limit):
    tiles = V._tiles(split)
    n = min(len(tiles), limit or len(tiles))
    nds = np.load(T.CACHE / f"{split}_ndsm.npy", mmap_mode="r")
    k = len(thr_list) * len(settings) * len(modes)
    per_area, hacc = {}, {}
    items = ((tiles[i], p, np.asarray(nds[i]), thr_list, settings, modes, planar) for i, p in predict(model, kind, split, n))
    with Pool(procs, maxtasksperchild=16) as pool:
        for i, (rows, hs) in enumerate(pool.imap(work, items)):
            a = tiles[i].area
            accs = per_area.setdefault(a, [dict.fromkeys(SUMS, 0.0) for _ in range(k)])
            for acc, r in zip(accs, rows):
                for s in SUMS:
                    acc[s] += r[s]
            ha = hacc.setdefault(a, dict.fromkeys(HSUMS, 0.0))
            for s in HSUMS:
                ha[s] += hs[s]
    all_ = [dict.fromkeys(SUMS, 0.0) for _ in range(k)]
    for accs in per_area.values():
        for x, acc in zip(all_, accs):
            for s in SUMS:
                x[s] += acc[s]
    hmae = {a: {"roof_mae": h["roof_abs"] / max(1, h["roof_n"]), "off_mae": h["off_abs"] / max(1, h["off_n"])}
            for a, h in hacc.items()}
    return [V.summarise(x) for x in all_], {a: [V.summarise(x) for x in accs] for a, accs in per_area.items()}, hmae


def tune(model, procs, limit):
    v1, _, _ = run(model, "H", "val", THR, GRID3[:1], ["pred"], False, procs, limit)
    thr = THR[max(range(len(THR)), key=lambda j: v1[j]["score"])]
    v2, _, _ = run(model, "H", "val", [thr], GRID3, ["pred"], False, procs, limit)
    k = max(range(len(v2)), key=lambda j: v2[j]["score"])
    return {"thr": thr, "params": GRID3[k], "val": v2[k]}


def decide(test, best):
    areas = [a for a in AREAS if a in test["R"]["areas"]] or list(test["R"]["areas"])
    b, r = test[best]["areas"], test["R"]["areas"]
    df = {a: b[a]["face_f1"] - r[a]["face_f1"] for a in areas}
    dh = {a: b[a]["h_rmse"] - r[a]["h_rmse"] for a in areas}
    viable = all(df[a] >= -0.10 and dh[a] <= 0.75 for a in areas)
    close = viable and all(df[a] >= -0.03 and dh[a] <= 0.25 for a in areas)
    h0, h1 = test["H0"]["areas"], test["H1"]["areas"]
    dn = {a: h1[a]["normal_err_deg"] - h0[a]["normal_err_deg"] for a in areas}
    dr = {a: h1[a]["h_rmse"] - h0[a]["h_rmse"] for a in areas}
    slope = all(dn[a] <= -1.0 and dr[a] <= 0.05 for a in areas)
    margins = [abs(df[a] + 0.10) for a in areas] + [abs(df[a] + 0.03) for a in areas]
    hm = [abs(dh[a] - 0.75) for a in areas] + [abs(dh[a] - 0.25) for a in areas] + [abs(dr[a] - 0.05) for a in areas]
    return {"best_arm": best, "face_f1_vs_R": df, "h_rmse_vs_R": dh, "rgb_only_viable": viable,
            "close_to_dsm": close, "h1_minus_h0_normal_deg": dn, "h1_minus_h0_rmse": dr, "slope_head_worth_it": slope,
            "second_seed_needed": min(margins) < 0.02 or min(hm) < 0.05 or min(abs(dn[a] + 1.0) for a in areas) < 0.25}


def load(path, arm):
    ck = torch.load(path, map_location="cpu")
    if ck.get("arm") != arm:
        raise SystemExit(f"{path} holds arm {ck.get('arm')}, expected {arm}")
    m, _ = T9.build(arm)
    m.load_state_dict(ck["state_dict"])
    return m.cuda().eval(), ck["seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", type=Path, required=True)
    ap.add_argument("--h1", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sha = hashlib.sha256(args.h0.read_bytes() + args.h1.read_bytes()).hexdigest()
    ledger = HERE / "runs" / "test_ledger.json"
    led = json.loads(ledger.read_text()) if ledger.exists() else {}
    out = HERE / "runs" / f"eval_{args.h0.stem}_{args.h1.stem}{'_limit' if args.limit else ''}.json"
    if not args.limit and (out.exists() or sha in led):
        raise SystemExit(f"test already evaluated for these checkpoints ({out}, {led.get(sha)})")
    h0, s0 = load(args.h0, "H0")
    h1, s1 = load(args.h1, "H1")
    ev47 = json.loads((E47 / "runs" / "eval_B3_s0_e150.json").read_text())
    r = T.build()
    r.load_state_dict(torch.load(E47 / "runs" / "B3_s0_e150.pt", map_location="cpu"))
    r = r.cuda().eval()
    res = {"seeds": [s0, s1], "sha256": sha, "r_sha256": ev47["sha256"]}
    t0 = time.time()
    sel = {"H0": tune(h0, args.procs, args.limit)}
    print("H0 val", {x: round(y, 3) for x, y in sel["H0"]["val"].items() if isinstance(y, float)}, f"{time.time() - t0:.0f}s", flush=True)
    sel["H1"] = tune(h1, args.procs, args.limit)
    v, _, _ = run(h1, "H", "val", [sel["H1"]["thr"]], [sel["H1"]["params"]], ["slope", "pred"], False, args.procs, args.limit)
    sel["H1"]["val_slope"], sel["H1"]["val_pred"] = v[0], v[1]
    print("H1 val", {x: round(y, 3) for x, y in v[0].items() if isinstance(y, float)}, f"{time.time() - t0:.0f}s", flush=True)
    sel["R"] = {"thr": ev47["selected"]["F3"]["thr"], "params": ev47["selected"]["F3"]["params"]}
    vs = {"H0": sel["H0"]["val"]["score"], "H1": sel["H1"]["val_slope"]["score"], "H1r": sel["H1"]["val_pred"]["score"]}
    best = max(vs, key=vs.get)
    res["selected"], res["val_scores"] = sel, vs
    test, hm = {}, {}
    test_split = "val" if args.limit else "test"  # smoke runs never touch test
    for name, model, kind, s, modes, keys in (("H0", h0, "H", sel["H0"], ["pred", "ndsm"], ["H0", "D"]),
                                              ("H1", h1, "H", sel["H1"], ["slope", "pred"], ["H1", "H1r"]),
                                              ("R", r, "R", sel["R"], ["ndsm"], ["R"])):
        allm, areas, hmae = run(model, kind, test_split, [s["thr"]], [s["params"]], modes, True, args.procs, args.limit)
        for j, key in enumerate(keys):
            test[key] = {"all": allm[j], "areas": {a: v[j] for a, v in areas.items()}}
            print(key, "test", {a: {x: round(y, 3) for x, y in v[j].items() if isinstance(y, float)} for a, v in areas.items()}, flush=True)
        hm[name] = hmae
    res["test"], res["height_mae"] = test, hm
    res["decision"] = decide(test, best)
    res["wall_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    if not args.limit:
        led[sha] = str(out)
        ledger.write_text(json.dumps(led, indent=1))
    print(json.dumps(res["decision"], indent=1), json.dumps(hm, indent=1))


if __name__ == "__main__":
    main()
