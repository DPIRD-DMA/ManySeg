"""E037 A1: L0 inventory (perfect targets, fixed decoders) -> runs/a1.jsonl (one row per tile x arm).

    E028_DATA=... E031_DATA=... uv run python run_a1.py [--limit N] [--procs 6] [--only roads|buildings|roofs]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree

import e037 as P

E = P.E
HERE = Path(__file__).resolve().parent
DATA = "$MANYSEG_BENCH_DATA"
OSM_AREAS = ("murdoch", "northam")
CITYSCALE = Path(DATA) / "cache/cityscale"
WHU = Path(DATA) / "cache/whu_aerial/val_poly.npz"
ROOF_AREAS = ("leidscherijn", "staphorst")
N_WHU = 100

# L0 decoder settings fixed from the earlier tuning (E028 runs/best.json L0, E035b best_35 L0, E031 best L0)
R0_P = {"thr": 0.3, "spur": 10, "gap": 0, "simplify": 1, "min_len": 20}
R2_P = {"thr_v": 0.2, "thr_e": 0.3, "link_r": 4, "gap": 0, "mutual": False, "kernel": 5}
B0_P = {"thr": 0.4, "thr_b": 0.3, "thr_s": 0.15, "simplify": 2}
B2_P = {"thr_v": 0.2, "link_r": 4, "gap": 0, "simplify": 0.5, "thr_e": 0.5, "mutual": False, "kernel": 3, "face_thr": 0.5}
B3R_P = {"thr_c": 0.3, "big_dev": 4.0, "agree": 0.4, "merge_deg": 8.0, "snap_r": 4.0}
A1_P = {"vertex_threshold": 0.1, "link_radius": 6.0, "gap": 0.0, "simplify": 1.0}
P8_ROAD = {"simplify": 1.0, "min_len": 0.0}
R3_P = dict(R0_P)

_T: dict = {}


# --------------------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------------------


def cityscale_tiles() -> list:
    split = json.loads((CITYSCALE / "split.json").read_text())
    ids = split.get("valid") or split.get("val")
    out = []
    for rid in ids:
        z = np.load(CITYSCALE / f"region_{rid}.npz")
        xy, edges, over = z["xy"].astype(float), z["edges"], z["overpass"].astype(float)
        for ty in range(2):
            for tx in range(2):
                ox, oy = tx * E.TILE, ty * E.TILE
                t = E.Tile(f"cityscale_{rid}", "roads", ty * 2 + tx, 0.0, 0.0, 1.0)
                g = nx.Graph()
                for k, (a, b) in enumerate(edges):
                    p, q = xy[a] - (ox, oy), xy[b] - (ox, oy)
                    tt = E._clip_t(p, q)
                    if tt is None:
                        continue
                    ua = int(a) if tt[0] == 0 else ("b", k, 0)
                    vb = int(b) if tt[1] == 1 else ("b", k, 1)
                    pa, qb = p + tt[0] * (q - p), p + tt[1] * (q - p)
                    if np.allclose(pa, qb):
                        continue
                    g.add_node(ua, xy=pa)
                    g.add_node(vb, xy=qb)
                    g.add_edge(ua, vb, bridge=False, layer=0, width=6.0)
                if g.number_of_edges() < 10:
                    continue
                t.graph = g
                c = over - (ox, oy)
                t.crossings = [p for p in c if 0 <= p[0] < E.TILE and 0 <= p[1] < E.TILE]
                out.append(t)
    return out


def whu_tiles() -> list:
    import sys
    sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
    from common import load_polygons
    polys = load_polygons(WHU)
    nonempty = [i for i, p in enumerate(polys) if len(p) >= 3]
    pick = nonempty[:: max(1, len(nonempty) // N_WHU)][:N_WHU]
    return [E.Tile("whu", "buildings", i, 0.0, 0.0, 0.3, polygons=list(polys[i])) for i in pick]


def tiles(key: str) -> list:
    if key not in _T:
        if key == "roads":
            ts = []
            for a in OSM_AREAS:
                ts += E.road_tiles(a, E.load_area(Path(os.environ["E028_DATA"]) / f"{a}.json"))
            _T[key] = ts + cityscale_tiles()
        elif key == "buildings":
            ts = []
            for a in OSM_AREAS:
                ts += E.building_tiles(a, E.load_area(Path(os.environ["E028_DATA"]) / f"{a}.json"))
            _T[key] = ts + whu_tiles()
        else:
            import e031
            ts = []
            for a in ROOF_AREAS:
                ts += e031.make_tiles(a, e031.load_faces(Path(os.environ["E031_DATA"]) / a))
            _T[key] = ts
    return _T[key]


# --------------------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------------------


def _arm_point(g, n, m, reach=8.0):
    prev, cur, d = n, m, 0.0
    while True:
        d += float(np.hypot(*(g.nodes[cur]["xy"] - g.nodes[prev]["xy"])))
        if d >= reach or g.degree(cur) != 2:
            return g.nodes[cur]["xy"]
        nxt = [x for x in g.neighbors(cur) if x != prev]
        prev, cur = cur, nxt[0]


def gt_junctions(g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT junctions (degree >= 3): xy, degree, minimum angle between arms measured 8 px out."""
    xy, deg, ang = [], [], []
    for n in g:
        if g.degree(n) < 3:
            continue
        c = g.nodes[n]["xy"]
        th = sorted(math.atan2(*(_arm_point(g, n, m) - c)[::-1]) for m in g.neighbors(n))
        gaps = np.diff(th + [th[0] + 2 * math.pi])
        xy.append(c)
        deg.append(g.degree(n))
        ang.append(math.degrees(gaps.min()))
    return np.array(xy).reshape(-1, 2), np.array(deg, int), np.array(ang)


def junction_hits(g_gt, g_pred, radius: float) -> dict:
    """Per GT junction: matched (a predicted junction within ``radius``, one-to-one greedy) and
    degree-correct; returned with the junction's degree and minimum angle."""
    xy, deg, ang = gt_junctions(g_gt)
    pj = [n for n in g_pred if g_pred.degree(n) >= 3]
    hit = np.zeros(len(xy), bool)
    hit_deg = np.zeros(len(xy), bool)
    if len(xy) and pj:
        pxy = np.array([g_pred.nodes[n]["xy"] for n in pj])
        pdeg = np.array([g_pred.degree(n) for n in pj])
        d, j = cKDTree(pxy).query(xy, k=min(3, len(pj)))
        d, j = d.reshape(len(xy), -1), j.reshape(len(xy), -1)
        used = set()
        for i in np.argsort(d[:, 0]):
            for dd, jj in zip(d[i], j[i]):
                if dd <= radius and jj not in used:
                    used.add(jj)
                    hit[i] = True
                    hit_deg[i] = pdeg[jj] == deg[i]
                    break
    return {"j_ang": ang.round(1).tolist(), "j_deg": deg.tolist(), "j_hit": hit.tolist(), "j_hit_deg": hit_deg.tolist()}


def building_hits(gt: list, pred: list) -> dict:
    per = [float(p.exterior.length) for p in gt]
    sep = []
    tree = shapely.STRtree(gt)
    for i, p in enumerate(gt):
        near = [j for j in tree.query(p.buffer(3.0)) if j != i]
        sep.append(min([float(p.distance(gt[j])) for j in near], default=99.0))
    hit = []
    ptree = shapely.STRtree(pred) if pred else None
    for p in gt:
        best = 0.0
        if ptree is not None:
            for j in ptree.query(p, predicate="intersects"):
                inter = p.intersection(pred[j]).area
                best = max(best, inter / (p.area + pred[j].area - inter))
        hit.append(best > 0.5)
    return {"b_per": np.round(per, 1).tolist(), "b_sep": np.round(sep, 2).tolist(), "b_hit": hit}


# --------------------------------------------------------------------------------------
# work
# --------------------------------------------------------------------------------------


def _enc_stats(a) -> dict:
    enc = a.get("_enc")
    return {} if enc is None else {"collisions": enc.collisions, "moved": enc.moved,
                                   "pos_links": int(enc.link.sum())}


def work(job: dict) -> list:
    key, ti = job["key"], job["tile"]
    t = tiles(key)[ti]
    rows = []
    for arm in job["arms"]:
        t0 = time.perf_counter()
        E.CACHE.clear()
        extra = {}
        if key == "roads":
            if arm in ("R0", "R2"):
                a = E.encode(t, arm).arrays
                pred = (E.decode_r0 if arm == "R0" else E.decode_r2)(a, R0_P if arm == "R0" else R2_P)
            else:
                a = P.encode_roads(t, arm)
                pred = P.decode_r3(a, R3_P) if arm.startswith("R3") else P.decode_p8_roads(a, P8_ROAD)
                extra = _enc_stats(a)
            m = E.road_metrics(t, pred)
            m.update(junction_hits(t.graph, pred, 5.0 / t.gsd))
        elif key == "buildings":
            if arm == "B0":
                pred = E.decode_b0(E.encode(t, "B0").arrays, B0_P)
            elif arm == "B2":
                pred = E.decode_b2(E.encode(t, "B2").arrays, B2_P)
            elif arm == "B3r":
                import e035 as B
                a3 = B.encode_b3(t).arrays
                pred = B.decode_b3r(a3, dict(B3R_P, b0=B0_P, kernel=3, simplify=B0_P["simplify"]))
            else:
                a = P.encode_buildings(t, arm)
                pred = P.decode_p8_faces(a, {"simplify": 0.5})
                extra = _enc_stats(a)
            m = E.polygon_metrics(t, pred)
            m.update(building_hits(t.polygons, pred))
        else:
            import e031
            if arm == "A1":
                faces, planes = e031.decode_a1(e031.encode(t), A1_P)
            else:
                g = P.lines_to_graph(P.roof_lines(t.faces, e031.TILE))
                enc = P.encode_p8(g, (e031.TILE, e031.TILE), **P.p8_opts(arm))
                a = {"node": enc.node, "link": enc.link, "offset": enc.offset, "mask": t.mask.astype(np.float32), "_enc": enc}
                faces = P.decode_p8_faces(a, {"simplify": 0.5})
                planes = e031.planes_for(faces, t.height)
                extra = _enc_stats(a)
            m = e031.metrics(t, faces, planes)
        rows.append(dict(key=key, area=t.area, tile=ti, arm=arm, s=round(time.perf_counter() - t0, 2), **extra, **m))
    return rows


ARMS = {"roads": ("R0", "R2", "P8", "P8r", "R3"), "buildings": ("B0", "B2", "B3r", "P8", "P8r"),
        "roofs": ("A1", "P8", "P8r")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--arms", default=None, help="override, e.g. P8g,P8rg,R3g (amendment 1)")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "a1.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(exist_ok=True)
    keys = [args.only] if args.only else list(ARMS)
    jobs = []
    for k in keys:
        n = len(tiles(k))
        arms = ARMS[k] if not args.arms else tuple(a for a in args.arms.split(",") if a != "R3g" or k == "roads")
        jobs += [dict(key=k, tile=i, arms=arms) for i in range(n if args.limit is None else min(n, args.limit))]
    print(f"{len(jobs)} jobs", flush=True)
    done = 0
    with open(args.out, "w") as f, Pool(args.procs, maxtasksperchild=4) as pool:
        for rows in pool.imap_unordered(work, jobs):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            done += 1
            if done % 10 == 0:
                print(f"{done}/{len(jobs)} {time.strftime('%H:%M')}", flush=True)


if __name__ == "__main__":
    main()
