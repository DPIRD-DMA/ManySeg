"""Vector outputs end to end, one test per task, on CPU in a few seconds.

Each test starts from vector data in a projected CRS (EPSG:28350, 0.5 m pixels), encodes the
training targets with the library's own encoders, uses them as the model's outputs (plus mild
noise), decodes a 320 x 320 px scene block by block (3 x 3 blocks of 128 px; objects cross seams
and block corners) and checks that:

* the block decode equals the whole-scene decode (same objects, identical geometry: nothing is
  duplicated or cut at a seam);
* the decoded objects match the input vectors;
* the written GeoJSON (and GeoPackage, when geopandas is installed) holds world coordinates in
  the scene's CRS.
"""

import json
import math

import numpy as np
import pytest
import torch

shapely = pytest.importorskip("shapely")
pytest.importorskip("skimage")
pytest.importorskip("rasterio")
pytest.importorskip("networkx")

from manyseg.points import (  # noqa: E402
    PointBlockDecoder,
    find_peaks,
    gaussian_heatmap,
    points_from_xy,
)
from manyseg.vectors import (  # noqa: E402
    Scene,
    block_decoder,
    decode_scene,
    field_targets,
    footprint_targets,
    graph_from_road_heads,
    road_graph,
    road_targets,
    roof_face_targets,
    to_pixels,
    write_records,
)

H = W = 320
BLOCK, HALO = 128, 32  # 3 x 3 blocks; seams at x, y = 128 and 256
PX, X0, Y0 = 0.5, 391_000.0, 6_466_000.0
TRANSFORM = (PX, 0.0, X0, 0.0, -PX, Y0)  # rasterio order; north up
CRS = "EPSG:28350"
SCENE = Scene(H, W, TRANSFORM, CRS)


def world(g):
    """Pixel (x, y) -> metres, written out by hand (independent of the library's transforms)."""
    return shapely.transform(g, lambda xy: np.c_[X0 + PX * xy[:, 0], Y0 - PX * xy[:, 1]])


def pixel(g):
    return shapely.transform(g, lambda xy: np.c_[(xy[:, 0] - X0) / PX, (Y0 - xy[:, 1]) / PX])


def noisy(a, seed, sd=0.02, clip=True):
    """``a`` plus smooth noise (std ``sd``, ~8 px blobs), as a model's errors are smooth."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.normal(0, 1, np.shape(a)), 4, axes=(-2, -1))
    a = np.asarray(a, np.float32) + (n * sd / n.std()).astype(np.float32)
    return np.clip(a, 0, 1) if clip else a


def iou(a, b):
    return a.intersection(b).area / a.union(b).area


def assert_same_records(got, ref):
    assert [r["id"] for r in got] == [r["id"] for r in ref]
    for a, b in zip(got, ref):
        assert shapely.equals_exact(a["geometry"], b["geometry"], 0), a["id"]


def match(decoded, truth, min_iou):
    """One-to-one (optimal assignment by IoU): as many decoded as true polygons, and every
    pair at IoU >= min_iou."""
    from scipy.optimize import linear_sum_assignment

    assert len(decoded) == len(truth), (len(decoded), len(truth))
    m = np.array([[iou(t, d) for d in decoded] for t in truth])
    rows, cols = linear_sum_assignment(-m)
    bad = [(truth[r].wkt, m[r, c]) for r, c in zip(rows, cols) if m[r, c] < min_iou]
    assert not bad, bad


def check_written(records, tmp_path, name, mesh=False):
    """GeoJSON (and GPKG with geopandas) in world coordinates, CRS recorded. With ``mesh``
    the geometries are the 3-D mesh rings, checked by the caller."""
    path = write_records(records, tmp_path / f"{name}.geojson", SCENE, mesh=mesh)
    fc = json.loads(path.read_text())
    assert fc["crs"]["properties"]["name"] == CRS
    assert len(fc["features"]) == len(records)
    for f, r in zip(fc["features"], records):
        g = shapely.geometry.shape(f["geometry"])
        assert f["properties"]["id"] == r["id"]
        assert mesh or shapely.equals_exact(g, world(r["geometry"]), 1e-6)
    try:  # optional (not in the vectors extra): GeoPackage only where geopandas is installed
        import geopandas as gpd
    except ImportError:
        return fc
    gdf = gpd.read_file(write_records(records, tmp_path / f"{name}.gpkg", SCENE, mesh=mesh))
    assert gdf.crs.to_epsg() == 28350 and len(gdf) == len(records)
    for g, r in zip(gdf.geometry, records):
        assert mesh or shapely.equals_exact(g, world(r["geometry"]), 1e-6)
    return fc


# ---------------------------------------------------------------- footprints (B3, E038)

def _buildings():
    rot = shapely.affinity.rotate
    px = [
        shapely.box(30, 30, 52, 48),  # inside one core
        shapely.box(110, 40, 150, 70),  # across the x = 128 seam
        rot(shapely.box(112, 110, 148, 146), 15),  # over the (128, 128) block corner
        shapely.Polygon([(180, 240), (230, 240), (230, 252), (200, 252), (200, 280),
                         (180, 280)]),  # L-shape across y = 256
        shapely.box(60, 200, 290, 222),  # long: across x = 128 and x = 256
        shapely.box(300, 150, 330, 180),  # cut by the scene edge (clipped by to_pixels)
        rot(shapely.box(240, 60, 275, 95), -30),  # rotated, near a seam
    ]
    return [world(p) for p in px]


def test_footprints_blocks_equal_whole_and_input(tmp_path):
    buildings = _buildings()  # e.g. a cadastre layer in metres
    polys = to_pixels(buildings, TRANSFORM, (H, W), crs=CRS, raster_crs=CRS)
    t = footprint_targets(polys, (H, W))
    fields = {"extent": noisy(t["extent"], 0), "boundary": noisy(t["boundary"], 1),
              "distance": noisy(t["distance"], 2), "corner": noisy(t["corner"], 3),
              "direction": noisy(t["direction"], 4, 0.05, clip=False)}
    dec = block_decoder("footprints")  # B3: corners + wall directions
    recs = decode_scene(fields, dec, BLOCK, HALO, scene=SCENE)
    whole = decode_scene(fields, dec, block=max(H, W), scene=SCENE)
    assert recs.stats["blocks"] == 9 and recs.stats["pieces"] > 0  # seams were crossed
    assert_same_records(recs, whole)
    match([r["geometry"] for r in recs], polys, 0.9)
    # straight walls: the rectangles keep 4 corners, the L-shape 6
    nv = sorted(len(r["geometry"].exterior.coords) - 1 for r in recs)
    assert nv.count(4) >= 5 and 6 in nv
    assert sum(r["scene_edge"] for r in recs) == 1
    fc = check_written(recs, tmp_path, "buildings")
    got = [shapely.geometry.shape(f["geometry"]) for f in fc["features"]]
    inside = [b.intersection(world(shapely.box(0, 0, W, H))) for b in buildings]
    match(got, inside, 0.9)  # in metres, against the original layer


# ---------------------------------------------------------------- fields (B0, E046)

def _parcels():
    """Six touching fields: three columns with slanted borders, cut by a slanted row border."""
    ys, xs = np.linspace(0, H, 9), np.linspace(0, W, 9)
    cols = [lambda y: 0 * y, lambda y: 100 + 0.1 * y, lambda y: 215 - 0.05 * y, lambda y: W + 0 * y]
    rows = [lambda x: 0 * x, lambda x: 140 + 0.08 * x, lambda x: H + 0 * x]
    out = []
    for left, right in zip(cols[:-1], cols[1:]):
        col = shapely.Polygon(np.r_[np.c_[left(ys), ys], np.c_[right(ys), ys][::-1]])
        for top, bot in zip(rows[:-1], rows[1:]):
            row = shapely.Polygon(np.r_[np.c_[xs, top(xs)], np.c_[xs, bot(xs)][::-1]])
            out.append(col.intersection(row))
    return [world(p) for p in out]


def test_fields_blocks_equal_whole_and_input(tmp_path):
    parcels = _parcels()
    polys = to_pixels(parcels, TRANSFORM, (H, W))
    assert len(polys) == 6 and abs(sum(p.area for p in polys) - H * W) < 1  # a partition
    t = field_targets(polys, (H, W))
    fields = {k: noisy(t[k], i) for i, k in enumerate(("extent", "boundary", "distance"))}
    dec = block_decoder("fields")  # B0: outlines of the instance masks
    recs = decode_scene(fields, dec, BLOCK, HALO, scene=SCENE)
    whole = decode_scene(fields, dec, block=max(H, W), scene=SCENE)
    assert recs.stats["merged_objects"] == 6  # every field crosses a seam
    assert_same_records(recs, whole)
    match([r["geometry"] for r in recs], polys, 0.95)
    check_written(recs, tmp_path, "fields")


# ---------------------------------------------------------------- roads (R4, E044)

def _ways():
    def way(*xy):
        return world(shapely.LineString(xy))

    # OSM-style: ways meeting at a shared vertex form a junction; D crosses C without one
    return [way((0, 100), (60, 100), (200, 101), (320, 98)),  # A, across both x seams
            way((200, 0), (200, 101)),  # B: T-junction with A
            way((200, 101), (205, 320)),  # C
            way((60, 100), (120, 230)),  # E: junction with A, dead end, across y = 128
            way((30, 300), (300, 180))]  # D: overpass over C, across x = 128 and 256


def test_roads_blocks_equal_whole_and_input(tmp_path):
    lines = to_pixels(_ways(), TRANSFORM, (H, W))
    g, crossings = road_graph(lines, width=8)
    assert len(crossings) == 1
    t = road_targets(g, (H, W), crossings=crossings)
    fields = {"mask": noisy(t["mask"], 0), "keypoints": noisy(t["keypoints"], 1),
              "direction": noisy(t["direction"], 2, 0.05, clip=False)}
    dec = block_decoder("roads")  # R4: RoadBlockDecoder
    assert dec.min_halo <= HALO
    graph = decode_scene(fields, dec, BLOCK, HALO, scene=SCENE)
    ref = graph_from_road_heads(fields["mask"], fields["keypoints"], fields["direction"])
    assert np.array_equal(graph.xy, ref.xy) and np.array_equal(graph.edges, ref.edges)
    # against the input: junctions at the two shared vertices, none at the overpass
    junctions = graph.junctions()
    for p in ((60, 100), (200, 101)):
        assert np.hypot(*(junctions - p).T).min() < 4, (p, junctions)
    assert np.hypot(*(junctions - crossings[0]).T).min() > 8
    truth = shapely.union_all(lines)
    decoded = shapely.MultiLineString([q for q in graph.polylines()])
    assert max(truth.distance(shapely.Point(v)) for v in graph.xy) < 3  # all on a road
    along = [ln.interpolate(d) for ln in lines for d in np.arange(0, ln.length, 4.0)]
    near = np.mean([decoded.distance(p) < 4 for p in along])
    assert near > 0.95, near  # and the roads are covered
    # written in metres
    fc = json.loads(write_records(graph, tmp_path / "roads.geojson", SCENE).read_text())
    assert len(fc["features"]) == len(graph.polylines())
    for f, q in zip(fc["features"], graph.polylines()):
        metres = np.c_[X0 + PX * q[:, 0], Y0 - PX * q[:, 1]]
        assert np.allclose(f["geometry"]["coordinates"], metres)
    try:
        import geopandas as gpd
    except ImportError:
        return
    gdf = gpd.read_file(write_records(graph, tmp_path / "roads.gpkg", SCENE))
    assert gdf.crs.to_epsg() == 28350 and len(gdf) == len(graph.polylines())


# ---------------------------------------------------------------- roofs (E047b + E048b)

def _houses():
    """Roof faces (pixel coordinates) with their planes z = a x + b y + c (metres)."""
    faces = []

    def gable(x0, y0, x1, y1, eave=3.0, rise=0.15):
        ym = (y0 + y1) / 2  # ridge along x
        ridge = eave + rise * (ym - y0)
        faces.append((shapely.box(x0, y0, x1, ym), (0.0, rise, eave - rise * y0)))
        faces.append((shapely.box(x0, ym, x1, y1), (0.0, -rise, ridge + rise * ym)))

    def hip(x0, y0, x1, y1, eave=4.0, rise=0.15):
        d = (y1 - y0) / 2
        a, b, c, e = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
        r1, r2 = (x0 + d, y0 + d), (x1 - d, y0 + d)
        faces.append((shapely.Polygon([a, b, r2, r1]), (0.0, rise, eave - rise * y0)))
        faces.append((shapely.Polygon([r1, r2, c, e]), (0.0, -rise, eave + rise * y1)))
        faces.append((shapely.Polygon([a, r1, e]), (rise, 0.0, eave - rise * x0)))
        faces.append((shapely.Polygon([b, c, r2]), (-rise, 0.0, eave + rise * x1)))

    gable(100, 30, 160, 70)  # across x = 128
    hip(225, 225, 295, 285)  # over the (256, 256) block corner
    gable(30, 170, 80, 210)  # inside one core
    faces.append((shapely.box(150, 140, 200, 180), (0.0, 0.0, 6.0)))  # flat, near a corner
    return faces


def test_roofs_blocks_equal_whole_and_input(tmp_path):
    truth = _houses()
    faces = to_pixels([world(f) for f, _ in truth], TRANSFORM, (H, W))
    t = roof_face_targets(faces, (H, W))
    yy, xx = np.mgrid[0:H, 0:W] + 0.5  # pixel centres, the planes' coordinates
    ndsm = np.zeros((H, W), np.float32)  # metres above ground: the model's extra input
    for f, (a, b, c) in truth:
        inside = shapely.contains_xy(f, xx, yy)
        ndsm[inside] = (a * xx + b * yy + c)[inside]
    fields = {"extent": noisy(t["extent"], 0), "boundary": noisy(t["boundary"], 1),
              "distance": noisy(t["distance"], 2), "corner": noisy(t["corner"], 3),
              "direction": noisy(t["direction"], 4, 0.05, clip=False),
              "height": noisy(ndsm, 5, 0.05, clip=False)}
    dec = block_decoder("roofs")  # corners decoder, planar_faces join, robust planes, J2 mesh
    recs = decode_scene(fields, dec, BLOCK, HALO, scene=SCENE)
    whole = decode_scene(fields, dec, block=max(H, W), scene=SCENE)
    assert recs.stats["joined_in_combine"] > 0  # some clusters were joined across blocks
    assert [(r["id"], r["part"]) for r in recs] == [(r["id"], r["part"]) for r in whole]
    for a, b in zip(recs, whole):
        assert shapely.equals_exact(a["geometry"], b["geometry"], 0)
        assert a["plane"] == b["plane"] and np.array_equal(a["mesh"], b["mesh"])
    assert recs.stats["nan_planes"] == 0
    match([r["geometry"] for r in recs], [f for f, _ in truth], 0.85)
    for f, (a, b, c) in truth:  # each face's height at its centroid, within 0.25 m
        r = max(recs, key=lambda r: iou(f, r["geometry"]))
        p = f.centroid
        za = r["plane"][0] * p.x + r["plane"][1] * p.y + r["plane"][2]
        assert abs(za - (a * p.x + b * p.y + c)) < 0.25, (f.wkt, r["plane"], (a, b, c))
    # 2-D faces, then the 3-D mesh: z in metres from the planes
    check_written(recs, tmp_path, "roofs")
    fc = check_written(recs, tmp_path, "roofs3d", mesh=True)
    for feat, r in zip(fc["features"], recs):
        ring = np.asarray(feat["geometry"]["coordinates"][0])
        assert ring.shape[1] == 3 and np.allclose(ring[:-1, 2], r["mesh"][:, 2])
        assert np.allclose(ring[:-1, 0], X0 + PX * r["mesh"][:, 0])
    assert {"plane_a", "plane_b", "plane_c", "part"} <= set(fc["features"][0]["properties"])


# ---------------------------------------------------------------- points

def test_points_blocks_equal_whole_and_input():
    rng = np.random.default_rng(0)
    pts = [(127.6, 40.2), (128.3, 128.4), (255.5, 255.9), (40.7, 255.2), (300.1, 128.9)]
    while len(pts) < 40:  # trees at least 8 px apart, some on seams and block corners
        p = rng.uniform(4, 316, 2)
        if all(math.dist(p, q) >= 8 for q in pts):
            pts.append(tuple(p))
    trees = [world(shapely.Point(p)) for p in pts]  # e.g. a GeoDataFrame of tree points
    xy = to_pixels(trees, TRANSFORM, (H, W), kind="points")
    hm = gaussian_heatmap([points_from_xy(xy)], (H, W), sigma=2.0)[0].numpy()
    fields = {"heatmap": noisy(hm, 0)}
    dec = PointBlockDecoder(threshold=0.5)  # block_decoder("points") builds the same
    assert dec.reach <= HALO
    got = decode_scene(fields, dec, BLOCK, HALO, scene=SCENE)
    ((yx, scores),) = find_peaks(torch.from_numpy(fields["heatmap"])[None], 0.5)
    o = np.lexsort((yx[:, 1], yx[:, 0], -scores.double()))  # ties (clipped at 1) in raster order
    assert np.array_equal(got.points, yx.numpy()[o]) and np.array_equal(got.scores, scores[o])
    assert len(got) == len(pts)
    d = np.hypot(*(got.xy[:, None] - np.asarray(pts)[None]).transpose(2, 0, 1)).min(1)
    assert d.max() <= math.sqrt(0.5) + 1e-6  # each peak is the pixel holding its tree
    fc = got.to_geojson()  # through the scene's transform
    w = np.asarray([f["geometry"]["coordinates"] for f in fc["features"]])
    ref = np.asarray([(t.x, t.y) for t in trees])
    dw = np.hypot(*(w[:, None] - ref[None]).transpose(2, 0, 1)).min(1)
    assert dw.max() <= PX * math.sqrt(0.5) + 1e-6
    assert all(f["properties"]["score"] > 0.5 for f in fc["features"])
