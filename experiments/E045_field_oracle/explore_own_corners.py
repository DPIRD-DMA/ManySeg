"""Exploratory (val only): corner ownership for B3 on fields. Each peak goes to the instance whose
ring is nearest; compare pooled boundary F on NL val with E045's tuned settings."""
import sys, json, numpy as np, shapely
from multiprocessing import Pool
sys.path.insert(0, "../E028_vector_oracle"); sys.path.insert(0, ".")
import e028 as E, e035 as B, e045, run45
from manyseg.vectors import polygons_from_labels
from shapely.geometry import Polygon
b0, b3 = json.load(open("runs/best0.json")), json.load(open("runs/best3.json"))

def dec(a, p, own):
    lab = E._labels_b0(a, p["b0"])
    exact = polygons_from_labels(lab, simplify=0.0, min_area=16)
    pk = E._peaks(a["corner"], p["thr_c"], p["kernel"])
    rings = [np.asarray(shapely.segmentize(q.exterior, 1.0).coords) for q in exact]
    owner = None
    if own and len(pk) and exact:
        tree = shapely.STRtree([q.exterior for q in exact])
        owner = tree.nearest(shapely.points(pk))
    out = []
    for i, ring in enumerate(rings):
        cp = pk if owner is None else pk[owner == i]
        q = B.polygon_b3r(ring, cp, a["direction"], p)
        if q is not None:
            out.append(q)
    return out

def work(job):
    ti, lv, s = job
    t = run45.tiles(run45.TUNE)[ti]
    a = B.corrupt_b3(B.encode_b3(t), t, E.LEVELS[lv], s)
    p0 = b0[f"L{lv}"]["params"]
    p = dict(b3[f"L{lv}"]["params"], b0=p0, kernel=3, simplify=p0["simplify"])
    lab = E._labels_b0(a, p0)
    P0 = [Polygon(q.exterior) for q in polygons_from_labels(lab, simplify=p0["simplify"], min_area=1)]
    res = {}
    for name, P in (("B0", P0), ("B3", dec(a, p, False)), ("B3own", dec(a, p, True))):
        res[name] = e045.boundary_counts(t.polygons, P)
    return lv, res

if __name__ == "__main__":
    jobs = [(ti, lv, 0) for ti in range(0, 42, 3) for lv in (0, 1, 2)]
    acc = {}
    with Pool(8) as pool:
        for lv, res in pool.imap_unordered(work, jobs):
            for k, v in res.items():
                acc.setdefault((lv, k), np.zeros(4))
                acc[(lv, k)] += v
    for (lv, k), v in sorted(acc.items()):
        p, r = v[0] / v[1], v[2] / v[3]
        print(f"L{lv} {k:6s} bf {2 * p * r / (p + r):.4f}  P {p:.3f} R {r:.3f}")
