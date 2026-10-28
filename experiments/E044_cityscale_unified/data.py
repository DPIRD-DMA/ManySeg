"""E044 data: City-Scale regions (2048 px, 1 m) cut into four 1024 tiles; unified road targets
(E039 encode_r4 logic via `encode_fast`, bbox-restricted: 28 of ~4M direction values differ from encode_r4 by window rasterisation; surface mask, 3 px centreline, junction / end / crossing heatmaps, two-mode
(cos 2t, sin 2t) direction) precomputed once into memory-mapped uint8 / int8 arrays.

    uv run python data.py prepare --out <cache dir> [--procs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "E039_unified_roads"), str(HERE.parent / "E028_vector_oracle")]
import e039 as R  # noqa: E402

E = R.E
GRAPHS = Path("$MANYSEG_BENCH_DATA/cache/cityscale")
IMAGES = Path("$MANYSEG_BENCH_DATA/raw/rngdet_roads/cityscale/data/20cities")
T = E.TILE  # 1024
WIDTH_PX = 6.0  # City-Scale has no road widths; 6 m at 1 m/px
CH = ("mask", "line", "junction", "end", "crossing")


def split() -> dict:
    s = json.loads((GRAPHS / "split.json").read_text())
    return {"train": s["train"], "val": s.get("valid") or s["val"], "test": s["test"]}


def region_tiles(rid: int) -> list:
    """Four 1024 tiles of one region as E028 Tiles (graph clipped to the tile, crossings), in
    (ty, tx) order; empty tiles kept (graph with no edges)."""
    z = np.load(GRAPHS / f"region_{rid}.npz")
    xy, edges, over = z["xy"].astype(float), z["edges"], z["overpass"].astype(float)
    out = []
    for ty in range(2):
        for tx in range(2):
            ox, oy = tx * T, ty * T
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
                g.add_edge(ua, vb, bridge=False, layer=0, width=WIDTH_PX)
            t.graph = g
            c = over - (ox, oy)
            t.crossings = [q for q in c if 0 <= q[0] < T and 0 <= q[1] < T]
            out.append(t)
    return out


def image_tiles(rid: int) -> np.ndarray:
    from PIL import Image
    im = np.asarray(Image.open(IMAGES / f"region_{rid}_sat.png").convert("RGB"))
    return np.stack([im[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] for ty in range(2) for tx in range(2)])


def encode_fast(t) -> dict:
    """``e039.encode_r4`` with the per-edge direction updates done inside each edge's bounding box
    (all channels identical except 28 of ~4M direction values on a test tile, from window rasterisation
    at buffer boundaries; the full-tile array operations per edge took ~90% of the time)."""
    import math
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely.geometry import LineString
    g = t.graph
    S = (T, T)
    arr = {"mask": E.road_mask(t)}
    arr["line"] = rasterize([(LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]).buffer(1.5), 1) for u, v in g.edges()],
                            out_shape=S, transform=E.IDENT, dtype="uint8").astype(np.float32)
    pts = {"junction": np.array([g.nodes[n]["xy"] for n in g if g.degree(n) >= 3]).reshape(-1, 2),
           "end": np.array([g.nodes[n]["xy"] for n in g if g.degree(n) == 1 and not E._near_border(g.nodes[n]["xy"])]).reshape(-1, 2),
           "crossing": np.array(t.crossings, float).reshape(-1, 2)}
    for c in R.CLASSES:
        arr[c] = E.gaussian_points(pts[c], 2.0) if len(pts[c]) else np.zeros(S, np.float32)
    d = np.zeros((4, *S), np.float32)
    owner = np.full(S, -1, np.int64)
    ang1 = np.full(S, np.nan, np.float32)
    cid = R._chains(g)
    for u, v, data in g.edges(data=True):
        p, q = g.nodes[u]["xy"], g.nodes[v]["xy"]
        th = math.atan2(q[1] - p[1], q[0] - p[0])
        poly = LineString([p, q]).buffer(data["width"] / 2)
        x0, y0, x1, y1 = poly.bounds
        c0, r0 = max(int(math.floor(x0)) - 1, 0), max(int(math.floor(y0)) - 1, 0)
        c1, r1 = min(int(math.ceil(x1)) + 1, T), min(int(math.ceil(y1)) + 1, T)
        if c0 >= c1 or r0 >= r1:
            continue
        m = rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=Affine(1, 0, c0, 0, 1, r0),
                      dtype="uint8").astype(bool)
        c = cid[frozenset((u, v))]
        ow, a1 = owner[r0:r1, c0:c1], ang1[r0:r1, c0:c1]
        first = m & (ow < 0)
        d[0, r0:r1, c0:c1][first], d[1, r0:r1, c0:c1][first] = math.cos(2 * th), math.sin(2 * th)
        ow[first], a1[first] = c, th
        dang = np.abs(((a1 - th) + math.pi / 2) % math.pi - math.pi / 2)
        second = m & (ow >= 0) & (ow != c) & (dang > math.radians(20))
        d[2, r0:r1, c0:c1][second], d[3, r0:r1, c0:c1][second] = math.cos(2 * th), math.sin(2 * th)
    arr["direction"] = d
    return arr


def encode_tile(t) -> tuple[np.ndarray, np.ndarray]:
    if t.graph.number_of_edges() == 0:
        return np.zeros((len(CH), T, T), np.uint8), np.zeros((4, T, T), np.int8)
    a = encode_fast(t)
    probs = np.stack([np.clip(a[c], 0, 1) for c in CH])
    return (probs * 255).round().astype(np.uint8), (np.clip(a["direction"], -1, 1) * 127).round().astype(np.int8)


def _job(args):
    rid = args
    ts = region_tiles(rid)
    enc = [encode_tile(t) for t in ts]
    return rid, image_tiles(rid), np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc])


def prepare(out: Path, procs: int):
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in split().items():
        n = 4 * len(ids)
        img = np.lib.format.open_memmap(out / f"{name}_img.npy", "w+", np.uint8, (n, T, T, 3))
        prb = np.lib.format.open_memmap(out / f"{name}_prob.npy", "w+", np.uint8, (n, len(CH), T, T))
        dr = np.lib.format.open_memmap(out / f"{name}_dir.npy", "w+", np.int8, (n, 4, T, T))
        order = {rid: k for k, rid in enumerate(ids)}
        with Pool(procs, maxtasksperchild=4) as pool:
            for rid, im, p, d in pool.imap_unordered(_job, ids):
                k = order[rid] * 4
                img[k:k + 4], prb[k:k + 4], dr[k:k + 4] = im, p, d
                print(name, rid, flush=True)
        img.flush(); prb.flush(); dr.flush()
        (out / f"{name}_ids.json").write_text(json.dumps(ids))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    prepare(a.out, a.procs)
