"""E037 A2: target stability and chain survival for A1 survivors -> runs/a2_stab.jsonl, runs/a2_chain.jsonl.

    uv run python run_a2.py stability|chain --arms P8r,R3 [--procs 6]
Tile subsets (compute): roads 20, buildings 40 (20 OSM + 20 WHU), roofs 30, evenly spaced.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

import e037 as P
import run_a1 as R

E = P.E
HERE = Path(__file__).resolve().parent
QS = (0.001, 0.005, 0.01, 0.02, 0.05)
SEEDS = (0, 1, 2)
N_SUB = {"roads": 20, "buildings": 40, "roofs": 30}


def subset(key):
    ts = R.tiles(key)
    if key == "buildings":
        osm = [i for i, t in enumerate(ts) if t.area != "whu"]
        whu = [i for i, t in enumerate(ts) if t.area == "whu"]
        return osm[:: max(1, len(osm) // 20)][:20] + whu[:: max(1, len(whu) // 20)][:20]
    return list(range(len(ts)))[:: max(1, len(ts) // N_SUB[key])][:N_SUB[key]]


def gt_graph(key, t):
    if key == "roads":
        return P.road_graph_px(t)
    if key == "buildings":
        return P.lines_to_graph(E.building_lines(t.polygons))
    import e031
    return P.lines_to_graph(P.roof_lines(t.faces, e031.TILE))


def shape_of(key):
    if key == "roofs":
        import e031
        return (e031.TILE, e031.TILE)
    return (E.TILE, E.TILE)


# --------------------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------------------


def undirected_links(link) -> set:
    out = set()
    for k, (dr, dc) in enumerate(P.DIRS[:4]):
        rr, cc = np.nonzero(link[k] > 0.5)
        out |= {((r, c), (r + dr, c + dc)) for r, c in zip(rr, cc)}
    for k in range(4, 8):  # links stored only on the other end (one-sided) still count
        dr, dc = P.DIRS[k]
        rr, cc = np.nonzero(link[k] > 0.5)
        out |= {((r + dr, c + dc), (r, c)) for r, c in zip(rr, cc)}
    return {tuple(sorted(e)) for e in out}


def dihedral(k: int, S: int):
    """Map of (x, y) under the k-th dihedral transform of an S x S tile."""
    def fwd(xy):
        x, y = xy[..., 0], xy[..., 1]
        a = [(x, y), (S - y, x), (S - x, S - y), (y, S - x), (S - x, y), (x, S - y), (y, x), (S - y, S - x)][k]
        return np.stack(a, -1)

    return fwd


def _inverse_k(k):
    # inverse of each element of the dihedral group in the order used by dihedral()
    return [0, 3, 2, 1, 4, 5, 6, 7][k]


def transform_graph(g, f):
    h = g.copy()
    for n in h:
        h.nodes[n]["xy"] = f(np.asarray(h.nodes[n]["xy"], float))
    return h


def links_back(links: set, kinv: int, S: int) -> set:
    f = dihedral(kinv, S)
    out = set()
    for a, b in links:
        ca = f(np.array([a[1] + 0.5, a[0] + 0.5]))
        cb = f(np.array([b[1] + 0.5, b[0] + 0.5]))
        ra = (int(np.floor(ca[1])), int(np.floor(ca[0])))
        rb = (int(np.floor(cb[1])), int(np.floor(cb[0])))
        out.add(tuple(sorted((ra, rb))))
    return out


def disagreement(a: set, b: set) -> float:
    u = a | b
    return len(a ^ b) / max(len(u), 1)


def work_stab(job):
    key, ti, arm = job["key"], job["tile"], job["arm"]
    t = R.tiles(key)[ti]
    S = shape_of(key)
    g = gt_graph(key, t)
    o = P.p8_opts(arm)
    base = undirected_links(P.encode_p8(g, S, **o).link)
    rows = []
    for k in range(1, 8):
        enc = P.encode_p8(transform_graph(g, dihedral(k, S[0])), S, **o)
        back = links_back(undirected_links(enc.link), _inverse_k(k), S[0])
        rows.append(dict(key=key, tile=ti, arm=arm, kind=f"dihedral{k}", dis=disagreement(base, back), n=len(base)))
    for dx, dy in ((0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25)):
        enc = P.encode_p8(transform_graph(g, lambda xy: xy + np.array([dx, dy])), S, **o)
        rows.append(dict(key=key, tile=ti, arm=arm, kind=f"shift{dx:+.2f},{dy:+.2f}",
                         dis=disagreement(base, undirected_links(enc.link)), n=len(base)))
    return rows


# --------------------------------------------------------------------------------------
# chain survival
# --------------------------------------------------------------------------------------


def delete_links(a: dict, q: float, corr: bool, rng, region=None) -> dict:
    link = a["link"].copy()
    if not corr:
        kill = rng.random(link.shape) < q
        if region is not None:
            kill &= region[None]
        link[kill] = 0
    else:  # runs of 3 consecutive undirected links (both directions) at the same expected rate
        und = sorted(undirected_links(link))
        if region is not None:
            und = [e for e in und if region[e[0]] and region[e[1]]]
        adj: dict = {}
        for e in und:
            for c in e:
                adj.setdefault(c, []).append(e)
        dead = set()
        for i in np.nonzero(rng.random(len(und)) < q / 3)[0]:
            e = und[i]
            dead.add(e)
            for c in e:
                for f in adj.get(c, []):
                    dead.add(f)
        for (r0, c0), (r1, c1) in dead:
            k = P.DIRS.index((r1 - r0, c1 - c0))
            link[k, r0, c0] = 0
            link[P.OPP[k], r1, c1] = 0
    return dict(a, link=link)


def score(key, t, pred_or_faces):
    if key == "roads":
        return E.road_metrics(t, pred_or_faces)["apls"]
    if key == "buildings":
        m = E.polygon_metrics(t, pred_or_faces)
        return m["ciou"] * m["det_f1"]
    import e031
    faces = pred_or_faces
    return e031.metrics(t, faces, e031.planes_for(faces, t.height))["face_f1"]


def decode(key, arm, a, rule, gap, heal=False):
    if key == "roads":
        if arm.startswith("R3"):
            return P.decode_r3(a, dict(R.R3_P, rule=rule, gap=gap))
        return P.decode_p8_roads(a, dict(R.P8_ROAD, rule=rule, gap=gap, heal=heal))
    return P.decode_p8_faces(a, {"simplify": 0.5, "rule": rule, "gap": gap, "heal": heal})


DECODERS = (("and", 0, False), ("or", 0, False), ("and", 10, False), ("or", 10, False), ("or", 0, True), ("or", 10, True))


def encode(key, arm, t):
    if key == "roads":
        return P.encode_roads(t, arm)
    if key == "buildings":
        return P.encode_buildings(t, arm)
    import e031
    enc = P.encode_p8(gt_graph(key, t), shape_of(key), **P.p8_opts(arm))
    return {"node": enc.node, "link": enc.link, "offset": enc.offset, "mask": t.mask.astype(np.float32)}


def work_chain(job):
    key, ti, arm = job["key"], job["tile"], job["arm"]
    t = R.tiles(key)[ti]
    a = encode(key, arm, t)
    a.pop("_enc", None)
    region = a.get("zone") if arm.startswith("R3") else None
    nlinks = int((a["link"] > 0.5).sum())
    rows = []
    l0 = score(key, t, decode(key, arm, a, "and", 0))
    for q in QS:
        for kind in ("link", "link3", "node"):
            for seed in SEEDS:
                rng = np.random.default_rng([seed, ti, int(q * 1e4), len(kind), sum(map(ord, key))])
                if kind == "node":  # missing pixels: their neighbours still point at them
                    kill = (rng.random(a["node"].shape) < q) & (a["node"] > 0.5)
                    if region is not None:
                        kill &= region
                    b = dict(a, node=np.where(kill, 0.0, a["node"]))
                else:
                    b = delete_links(a, q, kind == "link3", rng, region)
                for rule, gap, hl in DECODERS:
                    if hl and arm.startswith("R3"):
                        continue
                    rows.append(dict(key=key, tile=ti, arm=arm, q=q, kind=kind, seed=seed, rule=rule, gap=gap, heal=hl,
                                     l0=l0, score=score(key, t, decode(key, arm, b, rule, gap, hl)), n_links=nlinks))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stability", "chain"])
    ap.add_argument("--arms", required=True, help="e.g. roads:P8r,roads:R3,buildings:P8r,roofs:P8r")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    pairs = [s.split(":") for s in args.arms.split(",")]
    jobs = [dict(key=k, tile=i, arm=a) for k, a in pairs for i in subset(k)]
    if args.what == "stability":
        jobs = [j for j in jobs if not j["arm"].startswith("R3")]  # R3's links are P8r's
    out = HERE / "runs" / f"a2_{'stab' if args.what == 'stability' else 'chain'}.jsonl"
    fn = work_stab if args.what == "stability" else work_chain
    print(f"{len(jobs)} jobs", flush=True)
    done = 0
    with open(out, "w") as f, Pool(args.procs, maxtasksperchild=4) as pool:
        for rows in pool.imap_unordered(fn, jobs):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            done += 1
            if done % 5 == 0:
                print(f"{done}/{len(jobs)} {time.strftime('%H:%M')}", flush=True)


if __name__ == "__main__":
    main()
