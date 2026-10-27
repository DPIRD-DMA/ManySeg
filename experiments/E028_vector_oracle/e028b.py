"""E028b: decoder-only fixes for graph-tensor footprints, on the E028 oracle.

B2F: per building (extent-mask component), B2's faces if they cover >= cov of it without
spilling outside it, else B0's polygons. B2M (diagnostic): for buildings that fail that gate,
add the component's outline to the graph's edges before polygonizing. B0 / B2 keep their frozen E028
parameters; only cov is tuned (on innaloo).
"""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

import e028 as E
from manyseg.vectors import polygons_from_labels


def _faces_inside(a2: dict, p2: dict, extra_lines=()) -> list[Polygon]:
    """E028's decode_b2 with optional extra lines added before polygonizing."""
    g = E.decode_graph_tensor(a2, dict(p2, snap_border=True))
    lines = [LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]) for u, v in g.edges()]
    lines += E.tile_border_lines() + list(extra_lines)
    faces = list(polygonize(unary_union(lines)))
    if not faces:
        return []
    lab = rasterize([(f, i + 1) for i, f in enumerate(faces)], out_shape=(E.TILE, E.TILE),
                    transform=E.IDENT, dtype="int32")
    n = len(faces) + 1
    cnt = np.bincount(lab.ravel(), minlength=n)
    s = np.bincount(lab.ravel(), weights=(a2["mask"] > 0.5).ravel(), minlength=n)
    out = []
    for i, f in enumerate(faces, start=1):
        if cnt[i] >= 16 and s[i] / cnt[i] > p2["face_thr"]:
            q = f.simplify(p2["simplify"])
            q = Polygon(q.exterior) if q.geom_type == "Polygon" else None
            if q is not None and q.is_valid and q.area > 1:
                out.append(q)
    return out


def _assign(faces: list, comp: np.ndarray, n: int):
    """Each face's owner = the component it overlaps most (0 = none), and per component the
    recall R = |C ∩ P| / |C| and precision Q = |C ∩ P| / |P| of the faces it owns."""
    if not faces or n == 0:
        return [], np.zeros(n + 1), np.zeros(n + 1)
    lab = rasterize([(f, i + 1) for i, f in enumerate(faces)], out_shape=comp.shape,
                    transform=E.IDENT, dtype="int32")
    pair = np.bincount(lab.ravel() * (n + 1) + comp.ravel(), minlength=(len(faces) + 1) * (n + 1))
    pair = pair.reshape(len(faces) + 1, n + 1)[1:]  # (faces, components incl. 0)
    owner = [int(np.argmax(r[1:]) + 1) if r[1:].max() > 0 else 0 for r in pair]
    inter, area = np.zeros(n + 1), np.zeros(n + 1)
    for i, o in enumerate(owner):
        if o:
            inter[o] += pair[i, o]
            area[o] += pair[i].sum()
    size = np.bincount(comp.ravel(), minlength=n + 1)
    return owner, inter / np.maximum(size, 1), inter / np.maximum(area, 1)


Q_MIN = 0.9  # the graph's faces must not spill outside the building (Codex review)


def decode_b2f(a2: dict, a0: dict, p: dict) -> list[Polygon]:
    """Per extent-mask component (B2's mask is the single extent source): B2's faces if their
    recall >= cov and precision >= Q_MIN, else B0's polygons for it; never both."""
    comp, n = ndimage.label(a2["mask"] > 0.5)
    f2 = _faces_inside(a2, p["b2"])
    own2, R, Q = _assign(f2, comp, n)
    use2 = (R >= p["cov"]) & (Q >= Q_MIN)
    out = [f for f, o in zip(f2, own2) if o and use2[o]]
    f0 = E.decode_b0(a0, p["b0"])
    own0, _, _ = _assign(f0, comp, n)
    out += [f for f, o in zip(f0, own0) if o and not use2[o]]
    return out


def decode_b2m(a2: dict, a0: dict, p: dict) -> list[Polygon]:
    """Diagnostic: buildings B2 does not cover (same gate as B2F) get their mask outline added to
    the graph's edges before polygonizing."""
    comp, n = ndimage.label(a2["mask"] > 0.5)
    f2 = _faces_inside(a2, p["b2"])
    _, R, Q = _assign(f2, comp, n)
    weak = np.nonzero(~((R >= p["cov"]) & (Q >= Q_MIN))[1:])[0] + 1
    if len(weak) == 0:
        return f2
    sub = np.where(np.isin(comp, weak), comp, 0)
    extra = [poly.exterior for poly in polygons_from_labels(sub, simplify=1.0, min_area=16)]
    return _faces_inside(a2, p["b2"], [LineString(r.coords) for r in extra])
