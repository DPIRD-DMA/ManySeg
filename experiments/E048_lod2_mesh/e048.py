"""E048 (README): vertex snapping before the J2 mesh join, on E047b's trained faces."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import shapely
import torch
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
E47 = HERE.parent / "E047_lod2_trained"
sys.path[:0] = [str(HERE.parent / d) for d in ("E030_whu_polygons", "E031_lod2_oracle", "E043_roof_mesh")] + [str(E47)]
import e031 as E  # noqa: E402
import e043  # noqa: E402
import evaluate47 as V  # noqa: E402
import train47 as T  # noqa: E402
from common import b3_probabilities  # noqa: E402

RADII = (1.0, 1.5, 2.5)


def snap(faces, r: float):
    if r <= 0 or len(faces) < 2:
        return list(faces)
    pts = [np.asarray(f.exterior.coords)[:-1] for f in faces]
    allp = np.concatenate(pts)
    owner = np.concatenate([[i] * len(p) for i, p in enumerate(pts)])
    parent = np.arange(len(allp))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(allp).query_pairs(r):
        if owner[i] != owner[j]:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(allp))])
    sums = {}
    for k, rt in enumerate(roots):
        s = sums.setdefault(rt, [np.zeros(2), 0])
        s[0] += allp[k]
        s[1] += 1
    out, k = [], 0
    for f, p in zip(faces, pts):
        q = np.array([sums[roots[k + j]][0] / sums[roots[k + j]][1] for j in range(len(p))])
        k += len(p)
        g = shapely.Polygon(q)
        out.append(g if g.is_valid and g.area > 1 else f)
    return out


def planarize(faces, r: float):
    """Snap faces onto their neighbours, node all outlines, re-polygonise, give each cell to the face
    covering most of it (E048b)."""
    if r <= 0 or len(faces) < 2:
        return list(faces)
    tree = shapely.STRtree(faces)
    snapped: list = []  # sequential: each face snaps onto faces already placed (larger first)
    order = sorted(range(len(faces)), key=lambda i: -faces[i].area)
    placed: dict = {}
    for i in order:
        f = faces[i]
        nb = [placed[j] for j in tree.query(f, predicate="dwithin", distance=r) if j in placed]
        g = shapely.snap(f, shapely.union_all([n.exterior for n in nb]), r) if nb else f
        g = g if g.is_valid else g.buffer(0)
        placed[i] = g if g.geom_type == "Polygon" and g.area > 1 else f
    snapped = [placed[i] for i in range(len(faces))]
    cells = list(shapely.polygonize([shapely.union_all([g.exterior for g in snapped])]).geoms)
    otree = shapely.STRtree(faces)
    parts: dict = {}
    for c in cells:
        cand = otree.query(c, predicate="intersects")
        if not len(cand):
            continue
        ov = [(faces[j].intersection(c).area, int(j)) for j in cand]
        a, j = max(ov)
        if a >= 0.5 * c.area:
            parts.setdefault(j, []).append(c)
    out = []
    for j, f in enumerate(faces):
        if j not in parts:
            continue
        u = shapely.union_all(parts[j])
        if u.geom_type == "MultiPolygon":
            u = max(u.geoms, key=lambda g: g.area)
        if u.geom_type == "Polygon" and u.area > 1:
            out.append(shapely.Polygon(u.exterior))
    return out


def seam_closure(faces, near: float = 2.0, on: float = 0.1, step: float = 0.5) -> tuple[int, int]:
    if len(faces) < 2:
        return 0, 0
    rings = [f.exterior for f in faces]
    tree = shapely.STRtree(rings)
    hit = tot = 0
    for i, rg in enumerate(rings):
        pts = shapely.line_interpolate_point(rg, np.arange(0, rg.length, step))
        for j in tree.query(rg, predicate="dwithin", distance=near):
            if j == i:
                continue
            d = shapely.distance(pts, rings[j])
            m = d <= near
            tot += int(m.sum())
            hit += int((d[m] <= on).sum())
    return hit, tot


def face_sets(split: str, sel: dict, model):
    from manyseg.instances import instances_from_outputs
    from manyseg.vectors import polygons_from_corners, polygons_from_labels

    ds = T.RoofTiles(split, augment=False, targets=False)
    mean, std = T.MEAN.cuda(), T.STD.cuda()
    for i in range(len(ds)):
        x = ds[i][0][None].cuda().float()
        x = torch.cat([(x[:, :3] - mean) / std, x[:, 3:] / T.NDSM_SCALE], 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = b3_probabilities(model(x), 0)
        out = {}
        for arm in ("F0", "F3"):
            tt = {k: torch.from_numpy(p[k].astype(np.float32))[None, None] for k in ("extent", "boundary", "distance")}
            lab = instances_from_outputs(tt["extent"], tt["boundary"], tt["distance"], logits=False, min_size=16,
                                         **sel[arm]["thr"])[0]
            if arm == "F0":
                f = polygons_from_labels(lab, simplify=sel[arm]["params"]["simplify"], min_area=16)
            else:
                f = polygons_from_corners(lab, p["corner"].astype(np.float32), p["direction"].astype(np.float32),
                                          snap_radius=4.0, simplify=1.0, min_area=16, **sel[arm]["params"])
            out[arm] = [g for g in f if g.geom_type == "Polygon"]
        yield i, out


def score(t, faces, nd, r, join=None):
    sf = (join or snap)(faces, r)
    a = {"height": nd}
    P = e043.planes(sf, a, "robust")
    rings, st = e043.mesh(sf, P, "J2")
    mm = e043.mesh_metrics(t, sf, rings, P)
    h, n = seam_closure(sf)
    fm = E.metrics(t, sf, P)
    return dict(seam_hit=h, seam_n=n, z_sum=mm["vertex_z_err"] * mm["n_vertices"] if mm["n_vertices"] else 0.0,
                z_n=mm["n_vertices"], false_steps=mm["false_steps"], shared=st["shared_vertices"],
                tp=fm["tp"], n_gt=fm["n_gt"], n_pred=fm["n_pred"])


def pooled(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    return {"seam_closure": s["seam_hit"] / max(1, s["seam_n"]), "vertex_z_err": s["z_sum"] / max(1, s["z_n"]),
            "false_steps_per_shared": s["false_steps"] / max(1, s["shared"]), "shared_vertices": s["shared"],
            "face_f1": 2 * s["tp"] / max(1, s["n_gt"] + s["n_pred"])}


def main():
    out = HERE / "runs" / "e048.json"
    if out.exists():
        raise SystemExit(f"{out} exists (test once)")
    out.parent.mkdir(exist_ok=True)
    ev = json.loads((E47 / "runs" / "eval_B3_s0_e150.json").read_text())
    model = T.build()
    model.load_state_dict(torch.load(E47 / "runs" / "B3_s0_e150.pt", map_location="cpu"))
    model = model.cuda().eval()
    res = {"model_sha256": ev["sha256"], "selected": {a: ev["selected"][a] for a in ("F0", "F3")}}
    # tune r on val (F3, seam closure)
    tiles = V._tiles("val")
    nds = np.load(V.CACHE / "val_ndsm.npy", mmap_mode="r")
    val = {r: [] for r in RADII}
    for i, fs in face_sets("val", res["selected"], model):
        for r in RADII:
            val[r].append(score(tiles[i], fs["F3"], np.asarray(nds[i], np.float32), r))
    vp = {r: pooled(v) for r, v in val.items()}
    r_best = max(RADII, key=lambda r: vp[r]["seam_closure"])
    res["val"] = {str(r): v for r, v in vp.items()}
    res["r"] = r_best
    print("val", {r: round(v["seam_closure"], 3) for r, v in vp.items()}, "-> r", r_best, flush=True)
    tiles = V._tiles("test")
    nds = np.load(V.CACHE / "test_ndsm.npy", mmap_mode="r")
    acc: dict = {}
    for i, fs in face_sets("test", res["selected"], model):
        nd = np.asarray(nds[i], np.float32)
        for arm in ("F0", "F3"):
            for name, r in (("J2", 0.0), ("snap+J2", r_best)):
                acc.setdefault((tiles[i].area, arm, name), []).append(score(tiles[i], fs[arm], nd, r))
    res["test"] = {f"{a}|{arm}|{n}": pooled(v) for (a, arm, n), v in acc.items()}
    dec = {}
    for area in ("leidscherijn", "staphorst"):
        b, s = res["test"][f"{area}|F3|J2"], res["test"][f"{area}|F3|snap+J2"]
        dec[area] = {"seam_gain": s["seam_closure"] - b["seam_closure"], "z_change": s["vertex_z_err"] - b["vertex_z_err"],
                     "face_f1_change": s["face_f1"] - b["face_f1"]}
    res["decision"] = {"areas": dec, "snap_adopted": all(d["seam_gain"] >= 0.20 and d["z_change"] <= 0.05
                                                         and d["face_f1_change"] >= -0.01 for d in dec.values())}
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["test"].items():
        print(k, {x: round(y, 3) for x, y in v.items()})
    print(json.dumps(res["decision"], indent=1))


def main_b():
    out = HERE / "runs" / "e048b.json"
    if out.exists():
        raise SystemExit(f"{out} exists (test once)")
    ev = json.loads((E47 / "runs" / "eval_B3_s0_e150.json").read_text())
    e48 = json.loads((HERE / "runs" / "e048.json").read_text())
    model = T.build()
    model.load_state_dict(torch.load(E47 / "runs" / "B3_s0_e150.pt", map_location="cpu"))
    model = model.cuda().eval()
    sel = {a: ev["selected"][a] for a in ("F0", "F3")}
    res = {"model_sha256": ev["sha256"], "selected": sel, "snap_r": e48["r"]}
    tiles = V._tiles("val")
    nds = np.load(V.CACHE / "val_ndsm.npy", mmap_mode="r")
    val = {r: [] for r in RADII}
    for i, fs in face_sets("val", sel, model):
        for r in RADII:
            val[r].append(score(tiles[i], fs["F3"], np.asarray(nds[i], np.float32), r, planarize))
    vp = {r: pooled(v) for r, v in val.items()}
    r_best = max(RADII, key=lambda r: vp[r]["seam_closure"])
    res["val"], res["r"] = {str(r): v for r, v in vp.items()}, r_best
    print("val", {r: round(v["seam_closure"], 3) for r, v in vp.items()}, "-> r", r_best, flush=True)
    tiles = V._tiles("test")
    nds = np.load(V.CACHE / "test_ndsm.npy", mmap_mode="r")
    acc: dict = {}
    for i, fs in face_sets("test", sel, model):
        nd = np.asarray(nds[i], np.float32)
        for name, r, j in (("J2", 0.0, None), ("snap+J2", e48["r"], None), ("planar+J2", r_best, planarize)):
            acc.setdefault((tiles[i].area, name), []).append(score(tiles[i], fs["F3"], nd, r, j))
    res["test"] = {f"{a}|F3|{n}": pooled(v) for (a, n), v in acc.items()}
    dec = {}
    for area in ("leidscherijn", "staphorst"):
        b, s_ = res["test"][f"{area}|F3|J2"], res["test"][f"{area}|F3|planar+J2"]
        dec[area] = {"seam_gain": s_["seam_closure"] - b["seam_closure"], "z_change": s_["vertex_z_err"] - b["vertex_z_err"],
                     "face_f1_change": s_["face_f1"] - b["face_f1"]}
    res["decision"] = {"areas": dec, "planar_adopted": all(d["seam_gain"] >= 0.20 and d["z_change"] <= 0.05
                                                           and d["face_f1_change"] >= -0.01 for d in dec.values())}
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["test"].items():
        print(k, {x: round(y, 3) for x, y in v.items()})
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main_b() if sys.argv[1:] == ["b"] else main()
