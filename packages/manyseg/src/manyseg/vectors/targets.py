"""Training targets for the vector presets from pixel-coordinate geometry (one call per task).

Start from vector data in a CRS with :func:`manyseg.vectors.to_pixels` (the tile's geotransform
and shape), then:

* footprints: :func:`footprint_targets` (corner-line preset, ``CornerLineTask``);
* roof faces: :func:`roof_face_targets` (the same preset, touching faces kept apart);
* fields: :func:`field_targets` (instance heads only, decoded as outlines);
* roads: :func:`road_graph`, then :func:`manyseg.vectors.road_targets`.

Every function returns float32 / int32 / bool numpy arrays per tile, keyed by the names the
presets and bench tasks use (``manyseg.vectors.presets`` and the tasks in ``manyseg-bench``,
the benchmark package in the GitHub repository, not on PyPI).
Rasterising follows rasterio (pixel centres inside; where polygons overlap, the later one wins),
as the experiments' encoders did.
"""

from __future__ import annotations

import numpy as np

from .corners import corner_targets
from .graph import _deps


def rasterize_instances(polygons, shape: tuple[int, int], *, min_area: int = 0,
                        fill_holes: bool = False, split: bool = False):
    """Instance-ID raster ``(H, W)`` int32 (0 = background) from pixel-coordinate polygons.

    Polygon ``k`` burns id ``k + 1`` (pixel centres inside; later polygons win where they
    overlap), then, in id order:

    * ``split``: each 4-connected piece of an id becomes its own instance;
    * ``min_area``: instances (pieces) under ``min_area`` pixels are dropped;
    * ``fill_holes``: holes of an instance that hold only background are filled (a hole holding
      another instance is kept around it).

    Ids are renumbered ``1..K`` without gaps. Returns ``(labels, source)``: ``source[k - 1]`` is
    the index of the polygon instance ``k`` came from (a polygon covering no pixel centre gets
    no instance).
    """
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy import ndimage

    H, W = shape
    shapes = [(p, k + 1) for k, p in enumerate(polygons) if not p.is_empty]
    m = (rasterize(shapes, out_shape=(H, W), transform=Affine.identity(), dtype="int32")
         if shapes else np.zeros((H, W), np.int32))
    out = np.zeros((H, W), np.int32)
    source = []
    for v, sl in enumerate(ndimage.find_objects(m), start=1):
        if sl is None:
            continue
        # a hole lies inside its instance's bounding box, so the crop fills it exactly
        mine = m[sl] == v
        pieces = ndimage.label(mine)[0] if split else mine.astype(np.int32)
        for j in range(1, int(pieces.max()) + 1):
            c = pieces == j
            if c.sum() < max(min_area, 1):
                continue
            if fill_holes:
                c = ndimage.binary_fill_holes(c) & ((m[sl] == 0) | c)
            o = out[sl]
            o[c & (o == 0)] = len(source) + 1
            source.append(v - 1)
    return out, np.asarray(source, np.int64)


def _instance_heads(labels: np.ndarray, boundary_width: float, distance: str) -> dict:
    import torch

    from ..instances import instance_targets

    t = instance_targets(torch.from_numpy(labels.astype(np.int64)), boundary_width=boundary_width,
                         distance=distance)
    return {k: v[0, 0].numpy().astype(np.float32) for k, v in t.items()}


def footprint_targets(polygons, shape: tuple[int, int], *, boundary_width: float = 2.0,
                      distance: str = "normalized", min_area: int = 0, instance_heads: bool = True,
                      **corner_kw) -> dict[str, np.ndarray]:
    """Every target of the corner-line preset (:func:`corner_line_heads`) for building
    footprints, from pixel-coordinate polygons (e.g. :func:`to_pixels`):

    * ``instances`` ``(H, W)`` int32: the instance ids (:func:`rasterize_instances`; holes such
      as courtyards kept);
    * ``corner`` ``(H, W)``, ``direction`` ``(2, H, W)``, ``direction_mask`` ``(H, W)`` bool:
      :func:`corner_targets` of the polygons that got an instance, as visible in the instance
      map (a polygon overlapped by a later one loses the covered part, so no wall or corner is
      drawn inside the later instance; ``corner_kw`` passed on);
    * ``extent``, ``boundary``, ``distance`` ``(H, W)`` float32: the instance heads'
      targets, :func:`manyseg.instances.instance_targets` with ``CornerLineTask``'s settings
      (``boundary_width`` 2, normalised distance). ``instance_heads=False`` leaves them out:
      ``CornerLineTask`` recomputes them from ``instances`` on the device.

    The first four are the labels ``CornerLineTask`` (in ``manyseg-bench``) reads; save them
    as a label dict and augment with :func:`dihedral` (``direction=True`` for ``direction``).
    """
    polygons = list(polygons)
    labels, source = rasterize_instances(polygons, shape, min_area=min_area)
    out = {"instances": labels,
           **corner_targets(_visible(polygons, np.unique(source)), shape, **corner_kw)}
    if instance_heads:
        out.update(_instance_heads(labels, boundary_width, distance))
    return out


def _visible(polygons, keep, min_overlap: float = 0.5) -> list:
    """The kept polygons minus the later polygons overlapping them by more than ``min_overlap``
    px² (the instance map's "later wins"), split into parts. Polygons that only touch (shared
    roof-face edges) are left exactly as given."""
    shapely = _deps()[1]
    tree = shapely.STRtree(polygons)
    out = []
    for i in keep:
        p = polygons[i]
        later = [polygons[j] for j in tree.query(p, predicate="intersects")
                 if j > i and p.intersection(polygons[j]).area > min_overlap]
        if later:
            p = p.difference(shapely.union_all(later))
        out += [q for q in getattr(p, "geoms", [p]) if q.geom_type == "Polygon" and q.area > 0]
    return out


def roof_face_targets(faces, shape: tuple[int, int], **kwargs) -> dict[str, np.ndarray]:
    """Targets of the corner-line preset for roof faces: :func:`footprint_targets` with
    each planar face as an instance. Faces sharing an edge stay separate instances (the
    ``boundary`` target runs along the shared edge on both sides, so the instance decoder
    splits them), and both faces' walls carry the edge direction. Give faces of one building
    without overlaps (where they do overlap, the later face wins).

    Roof heights are not a target: the trained roof model takes a normalised DSM
    (nDSM, metres above ground) as an input channel next to the image, and the roof planes
    are fitted to it after decoding the faces."""
    return footprint_targets(faces, shape, **kwargs)


def field_targets(polygons, shape: tuple[int, int], *, boundary_width: float = 2.0,
                  distance: str = "absolute", min_area: int = 16,
                  fill_holes: bool = True) -> dict[str, np.ndarray]:
    """Targets for field boundaries with the instance heads only (outline decoding with
    :func:`polygons_from_labels`; trained on Fields of The World), from pixel-coordinate polygons:
    ``instances`` ``(H, W)`` int32 and ``extent``, ``boundary``, ``distance`` ``(H, W)``
    float32 (``InstanceTask(boundary_width=2, distance="absolute")`` in ``manyseg-bench``).

    ``distance`` is ``"absolute"`` (px) by default (as good as normalised on trained FTW
    models, and an instance cut by a tile edge keeps its whole-scene values, so large scenes
    stitch without seam errors); train with the same setting and decode with a
    ``seed_threshold`` in pixels (``block_decoder("fields")`` does).

    Which fields count: every 4-connected piece of a field is its own
    instance, pieces under ``min_area`` pixels are dropped, and holes holding only background
    are filled (a field is trained as one solid parcel; a hole around another field stays).
    For the reference polygons that match the targets, decode the instances themselves
    (``polygons_from_labels(out["instances"])``).
    """
    labels, _ = rasterize_instances(polygons, shape, min_area=min_area, fill_holes=fill_holes,
                                    split=True)
    return {"instances": labels, **_instance_heads(labels, boundary_width, distance)}


def road_graph(lines, *, widths=None, width: float = 6.0, node: bool = False,
               snap: float = 1e-3, cross_merge: float = 0.0):
    """A road graph for :func:`road_targets` from pixel-coordinate lines (e.g.
    ``to_pixels(roads, transform, shape, kind="lines")``). Returns ``(graph, crossings)``:
    a ``networkx.Graph`` with node ``xy`` and edge ``width`` (px) and the grade-separated
    crossings ``(K, 2)``.

    Each line vertex is a node; vertices within ``snap`` px are one node, so lines meeting at a
    shared vertex form a junction (OSM ways share their junction nodes). Two edges that cross
    without a shared node are a grade-separated crossing (a bridge or tunnel), and
    their intersection points are returned for the crossing heatmap; points closer than
    ``cross_merge`` px are merged. ``node=True`` is for line
    data without that topology: all intersections become junctions and there are no crossings.

    Args:
        lines: shapely ``LineString``\\ s in pixel coordinates.
        widths: per-line road width in pixels (e.g. ``to_pixels(..., return_index=True)`` to
            map feature attributes; metres / pixel size); ``width`` where ``None``.
    """
    nx, shapely, cKDTree = _deps()[:3]
    lines = [shapely.LineString(np.asarray(ln.coords)[:, :2]) for ln in lines]
    w = np.broadcast_to(np.asarray(width if widths is None else widths, np.float64),
                        (len(lines),))
    if node and lines:
        noded = shapely.union_all(lines)
        tree = shapely.STRtree(lines)
        pieces = [p for p in getattr(noded, "geoms", [noded]) if p.geom_type == "LineString"]
        pw = []
        for p in pieces:  # width of the source line covering the piece's midpoint
            mid = p.interpolate(0.5, normalized=True)
            hit = tree.query(mid, predicate="dwithin", distance=1e-6)
            pw.append(float(w[hit].max()) if len(hit) else float(width))
        lines, w = pieces, np.asarray(pw)
    coords = [np.asarray(ln.coords, np.float64) for ln in lines]
    g = nx.Graph()
    if not coords:
        return g, np.zeros((0, 2))
    allxy = np.concatenate(coords)
    # union-find over vertices within snap px
    parent = np.arange(len(allxy))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(allxy).query_pairs(snap):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(i) for i in range(len(allxy))])
    for r in np.unique(root):
        g.add_node(int(r), xy=allxy[root == r].mean(0))
    start = 0
    for c, wk in zip(coords, w):
        ids = root[start:start + len(c)]
        start += len(c)
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            old = g.edges[a, b]["width"] if g.has_edge(a, b) else 0.0
            g.add_edge(int(a), int(b), width=max(float(wk), old))
    return g, _crossings(g, shapely, cross_merge)


def _crossings(g, shapely, merge: float) -> np.ndarray:
    es = list(g.edges())
    if not es:
        return np.zeros((0, 2))
    segs = [shapely.LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]) for u, v in es]
    left, right = shapely.STRtree(segs).query(segs, predicate="crosses")
    pts = []
    for i, j in zip(left, right):
        if i >= j or set(es[i]) & set(es[j]):
            continue
        p = segs[i].intersection(segs[j])
        if p.geom_type == "Point":
            pts.append((p.x, p.y))
    out: list = []
    for p in pts:
        if merge <= 0 or all(np.hypot(p[0] - o[0], p[1] - o[1]) > merge for o in out):
            out.append(p)
    return np.asarray(out, np.float64).reshape(-1, 2)
