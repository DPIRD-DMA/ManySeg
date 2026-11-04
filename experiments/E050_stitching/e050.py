"""E050 (README): blend vs cutline stitching of tiled predictions, scored by decoded-vector agreement
with a single-pass prediction of the whole scene.

    uv run python e050.py scenes            # fetch / build the roof scenes (PDOK, cached)
    uv run python e050.py run [--procs 6]   # tune on val, test once; writes runs/e050.json
"""

from __future__ import annotations

import os
import argparse
import itertools
import json
import pickle
import sys
import time
import types
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / d) for d in ("E030_whu_polygons", "E031_lod2_oracle", "E043_roof_mesh", "E047_lod2_trained",
                                       "E044_cityscale_unified", "E039_unified_roads", "E028_vector_oracle")]
sys.path.insert(0, os.environ.get("MANYPRED_SRC", "../ManyPred/src"))
from manypred.blend import gradient_weights  # noqa: E402
from manypred.windows import make_patch_indexes  # noqa: E402

DATA = Path(os.environ.get("MANYSEG_BENCH_DATA", "data"))
SCENES = DATA / "cache" / "e050_roof_scenes"
RUNS_E044 = Path(os.environ.get("MANYSEG_RUNS", "../ManySeg-runs"), "e044-90736d4/experiments/E044_cityscale_unified/runs")
S = 2048
CFG = {"roads": {"patch": 1024, "overlap": 256, "support": "mask"},
       "roofs": {"patch": 512, "overlap": 128, "support": "extent"}}
SEAM_GRID = [dict(mu=m, lam=0.0) for m in (0.1, 0.5)]
OBJ_GRID = [dict(mu=m, lam=l) for m, l in itertools.product((0.1, 0.5), (1.0, 4.0))]


# ----------------------------------------------------------------------------- scenes

def roof_windows(area: str, n: int):
    """Top-left (x0, y1) in RD metres of the n densest non-overlapping 512 m windows (128 m grid)."""
    import e031 as E
    faces = E.load_faces(DATA / "raw" / "3dbag" / area)
    c = np.array([f["poly"].centroid.coords[0] for f in faces])
    x0s = np.arange(np.floor(c[:, 0].min() / 128) * 128, c[:, 0].max(), 128)
    y1s = np.arange(np.ceil(c[:, 1].max() / 128) * 128, c[:, 1].min(), -128)
    cand = []
    for x0 in x0s:
        for y1 in y1s:
            k = ((c[:, 0] >= x0) & (c[:, 0] < x0 + 512) & (c[:, 1] <= y1) & (c[:, 1] > y1 - 512)).sum()
            cand.append((k, x0, y1))
    out = []
    for k, x0, y1 in sorted(cand, reverse=True):
        if all(abs(x0 - a) >= 512 or abs(y1 - b) >= 512 for a, b in out):
            out.append((x0, y1))
        if len(out) == n:
            break
    return out


def build_scenes():
    import prepare47 as P
    SCENES.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, areas in (("val", [("haarlem", 2)]), ("test", [("leidscherijn", 2), ("staphorst", 2)])):
        for area, n in areas:
            for j, (x0, y1) in enumerate(roof_windows(area, n)):
                rgb, nd = np.zeros((S, S, 3), np.uint8), np.zeros((S, S), np.float32)
                for r in range(4):
                    for q in range(4):
                        t = types.SimpleNamespace(area=f"e050_{area}", x0=x0 + 128 * q, y1=y1 - 128 * r)
                        a, b = P.inputs(t)
                        rgb[512 * r:512 * (r + 1), 512 * q:512 * (q + 1)], nd[512 * r:512 * (r + 1), 512 * q:512 * (q + 1)] = a, b
                name = f"{area}_{j}"
                np.save(SCENES / f"{name}_rgb.npy", rgb)
                np.save(SCENES / f"{name}_ndsm.npy", nd.astype(np.float16))
                meta[name] = {"split": split, "area": area, "x0": float(x0), "y1": float(y1)}
                print(name, x0, y1, flush=True)
    (SCENES / "meta.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- models

class Task:
    def __init__(self, name):
        self.name = name
        dev = torch.device("cuda")
        if name == "roads":
            from train44 import build
            self.model = build()
            self.model.load_state_dict(torch.load(RUNS_E044 / "U_s0.pt", map_location="cpu"))
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
            self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
        else:
            import train47 as T
            self.model = T.build()
            self.model.load_state_dict(torch.load(EXP / "E047_lod2_trained/runs/B3_s0_e150.pt", map_location="cpu"))
            self.T = T
        self.model = self.model.to(dev).eval()

    def scenes(self, split):
        if self.name == "roads":
            ids = json.loads((DATA / "cache" / "e044_cityscale" / f"{split}_ids.json").read_text())
            return [f"r{rid}" for rid in ids]
        meta = json.loads((SCENES / "meta.json").read_text())
        return [k for k, v in meta.items() if v["split"] == split]

    def image(self, scene):
        if self.name == "roads":
            import data as D
            from PIL import Image
            return np.asarray(Image.open(D.IMAGES / f"region_{scene[1:]}_sat.png").convert("RGB"))
        return np.load(SCENES / f"{scene}_rgb.npy")

    def x(self, scene):
        dev = torch.device("cuda")
        rgb = torch.from_numpy(self.image(scene)).permute(2, 0, 1).float().to(dev)
        if self.name == "roads":
            return (rgb[None] - self.mean) / self.std
        nd = torch.from_numpy(np.load(SCENES / f"{scene}_ndsm.npy").astype(np.float32)).to(dev)
        T = self.T
        return torch.cat([(rgb[None] - T.MEAN.to(dev)) / T.STD.to(dev), nd[None, None] / T.NDSM_SCALE], 1)

    @torch.no_grad()
    def fields(self, x):
        """Activated heads as {name: (C, H, W) float32}."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.model(x)
        if self.name == "roads":
            return {"mask": torch.sigmoid(o["mask"][0].float()).cpu().numpy(), "line": torch.sigmoid(o["line"][0].float()).cpu().numpy(),
                    "kp": torch.sigmoid(o["kp"][0].float()).cpu().numpy(), "direction": o["direction"][0].float().cpu().numpy()}
        return {"extent": torch.sigmoid(o["extent"][0].float()).cpu().numpy(), "boundary": torch.sigmoid(o["boundary"][0].float()).cpu().numpy(),
                "distance": o["distance"][0].float().cpu().numpy(), "corner": torch.sigmoid(o["corner"][0].float()).cpu().numpy(),
                "direction": o["direction"][0].float().cpu().numpy()}


def windows(task):
    c = CFG[task]
    return list(dict.fromkeys(make_patch_indexes(S, S, c["patch"], c["overlap"])))  # drop clamped duplicates


# ----------------------------------------------------------------------------- stitching

def blend(patches, wins, overlap):
    P = wins[0][1] - wins[0][0]
    w = gradient_weights(P, overlap).numpy()
    acc = {k: np.zeros((v.shape[0], S, S), np.float32) for k, v in patches[0].items()}
    ws = np.zeros((S, S), np.float32)
    for (t, b, l, r), p in zip(wins, patches):
        for k in acc:
            acc[k][:, t:b, l:r] += p[k] * w
        ws[t:b, l:r] += w
    return {k: v / ws for k, v in acc.items()}, None


def assemble(patches, wins, owner):
    out = {k: np.zeros((v.shape[0], S, S), np.float32) for k, v in patches[0].items()}
    for i, ((t, b, l, r), p) in enumerate(zip(wins, patches)):
        m = owner[t:b, l:r] == i
        for k in out:
            out[k][:, t:b, l:r][:, m] = p[k][:, m]
    return out


def midline(patches, wins, overlap):
    P = wins[0][1] - wins[0][0]
    w = gradient_weights(P, overlap).numpy()
    best, owner = np.full((S, S), -1.0, np.float32), np.full((S, S), -1, np.int32)
    for i, (t, b, l, r) in enumerate(wins):
        m = w > best[t:b, l:r]
        best[t:b, l:r][m] = w[m]
        owner[t:b, l:r][m] = i
    return assemble(patches, wins, owner), owner


def _path(C: np.ndarray) -> np.ndarray:
    """Min-cost top-to-bottom path through C (rows x cols), moving at most one column per row."""
    n = C.shape[1]
    acc, back = C[0].astype(np.float64).copy(), np.zeros(C.shape, np.int8)
    for r in range(1, C.shape[0]):
        cand = np.stack([np.r_[np.inf, acc[:-1]], acc, np.r_[acc[1:], np.inf]])  # from col-1, col, col+1
        k = cand.argmin(0)
        acc = cand[k, np.arange(n)] + C[r]
        back[r] = k - 1
    p = np.zeros(C.shape[0], np.int64)
    p[-1] = int(acc.argmin())
    for r in range(C.shape[0] - 1, 0, -1):
        p[r - 1] = p[r] + back[r, p[r]]
    return p


def _prior(n: int) -> np.ndarray:
    i = np.arange(n)
    return np.abs(i - (n - 1) / 2) / max((n - 1) / 2, 1)


def seam(patches, wins, overlap, mu, lam, support):
    owner = np.full((S, S), -1, np.int32)
    cur = {k: np.zeros((v.shape[0], S, S), np.float32) for k, v in patches[0].items()}
    for i, ((t, b, l, r), p) in enumerate(zip(wins, patches)):
        own = owner[t:b, l:r] >= 0
        P = b - t
        take = np.ones((P, P), bool)
        if own.any():
            c = {k: v[:, t:b, l:r] for k, v in cur.items()}
            cost = sum(np.abs(p[k] - c[k]).mean(0) * (0.5 if k == "direction" else 1.0) for k in p)
            cost = cost + lam * np.maximum(p[support][0], c[support][0])
            hT = int(np.argmin(np.r_[own.all(1), False]))  # leading fully owned rows
            wL = int(np.argmin(np.r_[own[hT:].all(0) if hT < P else np.zeros(P, bool), False]))  # leading owned cols below
            if hT >= P:  # window already fully owned (duplicate)
                continue
            if wL > 0:
                v = _path(cost[:, :wL] + mu * _prior(wL)[None, :])
                take &= np.arange(P)[None, :] >= v[:, None]
            if hT > 0:
                h = _path((cost[:hT, :] + mu * _prior(hT)[:, None]).T)
                take &= np.arange(P)[:, None] >= h[None, :]
            take |= ~own
        owner[t:b, l:r][take] = i
        for k in cur:
            cur[k][:, t:b, l:r][:, take] = p[k][:, take]
    return cur, owner


# ----------------------------------------------------------------------------- decoding and agreement

_DEC = {}


def decode(task, f):
    if task == "roads":
        import data as D
        if "p" not in _DEC:
            ev = json.loads((RUNS_E044 / "eval_U_s0.json").read_text())
            _DEC["p"] = (ev["r0_params"], ev["r4_params"])
        p0, p4 = _DEC["p"]
        a = {"mask": f["mask"][0], "line": f["line"][0], "direction": f["direction"],
             "junction": f["kp"][0], "end": f["kp"][1], "crossing": f["kp"][2]}
        g = D.R.decode_r4(a, dict(p4, r0=p0, gap=p0["gap"]))
        return [(np.asarray(g.nodes[u]["xy"]), np.asarray(g.nodes[v]["xy"])) for u, v in g.edges()], \
            np.array([g.nodes[n]["xy"] for n in g if g.degree(n) >= 3]).reshape(-1, 2)
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import polygons_from_corners
    from rasterio.features import rasterize
    if "s" not in _DEC:
        _DEC["s"] = json.loads((EXP / "E047_lod2_trained/runs/eval_B3_s0_e150.json").read_text())["selected"]["F3"]
    sel = _DEC["s"]
    tt = {k: torch.from_numpy(f[k].astype(np.float32))[None] for k in ("extent", "boundary", "distance")}
    lab = instances_from_outputs(tt["extent"], tt["boundary"], tt["distance"], logits=False, min_size=16, **sel["thr"])[0]
    faces = [g for g in polygons_from_corners(lab, f["corner"][0].astype(np.float32), f["direction"].astype(np.float32),
                                              snap_radius=4.0, simplify=1.0, min_area=16, **sel["params"]) if g.geom_type == "Polygon"]
    return rasterize([(g, k + 1) for k, g in enumerate(faces)], out_shape=(S, S), dtype="int32") if faces else np.zeros((S, S), np.int32)


def _sample(segs, step=1.0):
    pts = [a + (b - a) * t for a, b in segs for t in np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / step) + 1))]
    return np.array(pts).reshape(-1, 2)


def _zone(xy, zone):
    if len(xy) == 0:
        return np.zeros(0, bool)
    c = np.clip(xy.astype(int), 0, S - 1)
    return zone[c[:, 1], c[:, 0]]


def agree(task, ref, test, zone):
    """Counts per zone ("ov" overlap, "single"): tp_p, n_p (test side), tp_r, n_r (reference side)."""
    from scipy.spatial import cKDTree
    out = {}
    if task == "roads":
        (sr, jr), (st, jt) = ref, test
        pr, pt = _sample(sr), _sample(st)
        hit_t = cKDTree(pr).query(pt)[0] <= 3 if len(pr) and len(pt) else np.zeros(len(pt), bool)
        hit_r = cKDTree(pt).query(pr)[0] <= 3 if len(pr) and len(pt) else np.zeros(len(pr), bool)
        from scipy.optimize import linear_sum_assignment
        mt, mr = np.zeros(len(jt), bool), np.zeros(len(jr), bool)
        if len(jt) and len(jr):
            d = np.hypot(jt[:, None, 0] - jr[None, :, 0], jt[:, None, 1] - jr[None, :, 1])
            a, b = linear_sum_assignment(np.where(d <= 5, d, 1e6))
            ok = d[a, b] <= 5
            mt[a[ok]], mr[b[ok]] = True, True
        zt, zr, zjt, zjr = _zone(pt, zone), _zone(pr, zone), _zone(jt, zone), _zone(jr, zone)
        for name, sel_t, sel_r, sj_t, sj_r in (("ov", zt, zr, zjt, zjr), ("single", ~zt, ~zr, ~zjt, ~zjr)):
            out[name] = {"len": [int(hit_t[sel_t].sum()), int(sel_t.sum()), int(hit_r[sel_r].sum()), int(sel_r.sum())],
                         "junc": [int(mt[sj_t].sum()), int(sj_t.sum()), int(mr[sj_r].sum()), int(sj_r.sum())]}
        return out
    lr, lt = ref, test
    nr, nt = lr.max() + 1, lt.max() + 1
    pair = lr.astype(np.int64) * nt + lt
    inter = np.bincount(pair.ravel(), minlength=nr * nt).reshape(nr, nt)
    ar, at = inter.sum(1), inter.sum(0)
    iou = inter / np.maximum(ar[:, None] + at[None, :] - inter, 1)
    iou[0, :], iou[:, 0] = 0, 0
    mr, mt = (iou > 0.5).any(1), (iou > 0.5).any(0)
    zr_ = np.bincount(lr[zone].ravel(), minlength=nr) > 0
    zt_ = np.bincount(lt[zone].ravel(), minlength=nt) > 0
    pres_r, pres_t = ar > 0, at > 0
    pres_r[0] = pres_t[0] = False
    for name, sr_, st_ in (("ov", zr_, zt_), ("single", ~zr_, ~zt_), ("all", np.ones(nr, bool), np.ones(nt, bool))):
        a_, b_ = pres_t & st_, pres_r & sr_
        out[name] = {"face": [int((mt & a_).sum()), int(a_.sum()), int((mr & b_).sum()), int(b_.sum())]}
    return out


def mae(ref, f, zone):
    return {k: [float(np.abs(f[k] - ref[k]).mean(0)[zone].sum()), int(zone.sum()),
                float(np.abs(f[k] - ref[k]).mean(0)[~zone].sum()), int((~zone).sum())] for k in ref}


def f1(c):
    p, r = c[0] / max(c[1], 1), c[2] / max(c[3], 1)
    return 2 * p * r / max(p + r, 1e-9)


def work(item):
    torch.set_num_threads(1)
    task, scene, method, fields, ref_dec, zone, keep = item
    d = decode(task, fields)
    return scene, method, agree(task, ref_dec, d, zone), (d if keep else None)


def methods_for(stage, sel=None):
    if stage == "val":
        return [("blend", {}), ("midline", {})] + [(f"seam|{m['mu']}|{m['lam']}", m) for m in SEAM_GRID + OBJ_GRID]
    return [("blend", {}), ("midline", {}), (f"seam|{sel['seam']['mu']}|0.0", sel["seam"]),
            (f"seam_obj|{sel['seam_obj']['mu']}|{sel['seam_obj']['lam']}", sel["seam_obj"])]


def stitch(name, params, patches, wins, task):
    ov = CFG[task]["overlap"]
    if name == "blend":
        return blend(patches, wins, ov)
    if name == "midline":
        return midline(patches, wins, ov)
    return seam(patches, wins, ov, params["mu"], params["lam"], CFG[task]["support"])


def run_stage(task, T, stage, sel, procs, keep_vis=None):
    wins = windows(task)
    cov = np.zeros((S, S), np.int32)
    for t, b, l, r in wins:
        cov[t:b, l:r] += 1
    zone = cov >= 2
    acc, maes, vis = {}, {}, {}
    ms = methods_for(stage, sel)
    with Pool(procs, maxtasksperchild=8) as pool:
        for scene in T.scenes(stage):  # one scene at a time: bounded memory
            x = T.x(scene)
            ref = T.fields(x)
            patches = [T.fields(x[:, :, t:b, l:r]) for t, b, l, r in wins]
            ref_dec = decode(task, ref)
            keep = keep_vis == scene
            items = []
            for name, params in ms:
                f, owner = stitch(name.split("|")[0], params, patches, wins, task)
                m = maes.setdefault(name, {})
                for k, v in mae(ref, f, zone).items():
                    m[k] = [a + b for a, b in zip(m.get(k, [0, 0, 0, 0]), v)]
                if keep:
                    vis[name] = {"owner": owner}
                items.append((task, scene, name, {k: v.astype(np.float16) for k, v in f.items()}, ref_dec, zone, keep))
            if keep:
                vis["ref"] = {"dec": ref_dec}
            for sc, name, a, d in pool.imap_unordered(work, items):
                if d is not None:
                    vis[name]["dec"] = d
                s = acc.setdefault(name, {})
                for z, dd in a.items():
                    for k, c in dd.items():
                        key = f"{z}_{k}"
                        s[key] = [p + q for p, q in zip(s.get(key, [0, 0, 0, 0]), c)]
                print(task, stage, sc, name, {k: round(f1(v), 3) for k, v in a.get("ov", {}).items()}, flush=True)
            del patches, items
    res = {n: {k: {"f1": f1(c), "counts": c} for k, c in s.items()} for n, s in acc.items()}
    for n, m in maes.items():
        res[n]["mae"] = {k: {"ov": v[0] / max(v[1], 1), "single": v[2] / max(v[3], 1)} for k, v in m.items()}
    return res, vis


PRIMARY = {"roads": "ov_junc", "roofs": "ov_face"}
GUARD = {"roads": "ov_len", "roofs": "all_face"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scenes", "run"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tasks", default="roads,roofs")
    ap.add_argument("--limit-scenes", type=int, default=None)  # smoke runs (val only)
    args = ap.parse_args()
    if args.cmd == "scenes":
        return build_scenes()
    out = HERE / "runs" / ("e050.json" if not args.limit_scenes else "e050_smoke.json")
    if out.exists() and not args.limit_scenes:
        raise SystemExit(f"{out} exists (test once)")
    out.parent.mkdir(exist_ok=True)
    res = {}
    for task in args.tasks.split(","):
        t0 = time.time()
        T = Task(task)
        if args.limit_scenes:
            sc = T.scenes
            T.scenes = lambda split, sc=sc: sc("val")[: args.limit_scenes]
        val, _ = run_stage(task, T, "val", None, args.procs)
        pk = PRIMARY[task]
        best = lambda names: max(names, key=lambda n: val[n][pk]["f1"])
        sel = {"seam": SEAM_GRID[[f"seam|{m['mu']}|{m['lam']}" for m in SEAM_GRID].index(best([f"seam|{m['mu']}|{m['lam']}" for m in SEAM_GRID]))],
               "seam_obj": OBJ_GRID[[f"seam|{m['mu']}|{m['lam']}" for m in OBJ_GRID].index(best([f"seam|{m['mu']}|{m['lam']}" for m in OBJ_GRID]))]}
        cuts = {"midline": val["midline"][pk]["f1"], "seam": val[f"seam|{sel['seam']['mu']}|0.0"][pk]["f1"],
                "seam_obj": val[f"seam|{sel['seam_obj']['mu']}|{sel['seam_obj']['lam']}"][pk]["f1"]}
        best_cut = max(cuts, key=cuts.get)
        stage = "val" if args.limit_scenes else "test"
        first = T.scenes(stage)[0]
        test, vis = run_stage(task, T, stage, sel, args.procs, keep_vis=first)
        key = {"blend": "blend", "midline": "midline", "seam": f"seam|{sel['seam']['mu']}|0.0",
               "seam_obj": f"seam_obj|{sel['seam_obj']['mu']}|{sel['seam_obj']['lam']}"}
        b, c = test["blend"], test[key[best_cut]]
        dec = {"best_cut_on_val": best_cut, "val_primary": cuts, "primary_gain": c[pk]["f1"] - b[pk]["f1"],
               "guard_change": c[GUARD[task]]["f1"] - b[GUARD[task]]["f1"]}
        dec["cut_preferred"] = bool(dec["primary_gain"] >= 0.01 and dec["guard_change"] >= -0.005)
        res[task] = {"selected": sel, "val": val, "test": test, "decision": dec, "wall_s": time.time() - t0,
                     "vis_scene": first, "keys": key}
        with open(HERE / "runs" / f"vis_{task}{'_smoke' if args.limit_scenes else ''}.pkl", "wb") as fh:
            pickle.dump(vis, fh)
        print(task, json.dumps(dec, indent=1), flush=True)
        del T
        torch.cuda.empty_cache()
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
