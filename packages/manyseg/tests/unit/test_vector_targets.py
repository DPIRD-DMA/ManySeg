"""Training-target encoders for vector outputs: georeferencing (world features -> tile pixels,
the inverse of ``to_geojson``) and round trips per task (the E028 oracle as unit tests):
synthetic vectors -> targets -> the shipped decoder -> the same vectors within tolerance."""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("rasterio")

import shapely  # noqa: E402
from manyseg.instances import instance_targets  # noqa: E402
from manyseg.vectors import (  # noqa: E402
    corner_targets,
    field_targets,
    footprint_targets,
    from_gdal,
    graph_from_road_heads,
    pixel_to_world,
    polygons_from_corners,
    polygons_from_labels,
    rasterize_instances,
    road_graph,
    road_targets,
    roof_face_targets,
    to_geojson,
    to_pixels,
    world_to_pixel,
)
from rasterio.transform import Affine, from_origin  # noqa: E402

# A north-up 0.3 m tile in MGA zone 50 (Perth) and a rotated, sheared one
NORTH_UP = from_origin(391_000.0, 6_466_000.0, 0.3, 0.3)
ROTATED = Affine(0.25, -0.12, 391_000.0, -0.09, -0.31, 6_466_000.0)


def _match(pred, true):
    """IoU of each true polygon with its prediction under a one-to-one (Hungarian) assignment;
    0 for a true polygon left without one."""
    from scipy.optimize import linear_sum_assignment

    iou = np.array([[p.intersection(t).area / p.union(t).area for p in pred] for t in true])
    out = [0.0] * len(true)
    if iou.size:
        for r, c in zip(*linear_sum_assignment(-iou)):
            out[r] = float(iou[r, c])
    return out


# --- georeferencing ----------------------------------------------------------------------------

def test_pixel_corner_convention():
    """Not just a round trip (a shared half-pixel error would cancel): the world footprint of
    pixel (row 2, col 3), as rasterio defines it, becomes the unit box at x 3..4, y 2..3 and
    burns exactly that pixel."""
    x0, y1 = NORTH_UP * (3, 2)
    cell = shapely.box(x0, y1 - 0.3, x0 + 0.3, y1)
    (p,) = to_pixels([cell], NORTH_UP, (8, 8))
    assert np.allclose(p.bounds, (3, 2, 4, 3))
    lab, _ = rasterize_instances([p], (8, 8))
    assert lab.sum() == 1 and lab[2, 3] == 1


@pytest.mark.parametrize("transform", [NORTH_UP, ROTATED])
def test_pixel_world_round_trip(transform):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-10, 600, (50, 2))
    w = pixel_to_world(xy, transform)
    assert np.allclose(world_to_pixel(w, transform), xy, atol=1e-6)
    # same convention as rasterio (pixel corners at integers) and to_geojson
    assert np.allclose(w[0], transform @ tuple(xy[0]))
    assert np.allclose(pixel_to_world([[0.5, 0.5]], transform)[0], transform @ (0.5, 0.5))
    poly = shapely.Polygon(xy[:4]).convex_hull
    fc = to_geojson([poly], transform=transform)
    back = to_pixels(fc, transform, (700, 700))
    assert len(back) == 1 and back[0].symmetric_difference(poly).area < 1e-6
    # a GDAL geotransform and a plain 6-tuple give the same mapping
    gdal = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    assert np.allclose(world_to_pixel(w, from_gdal(gdal)), xy, atol=1e-6)
    assert np.allclose(world_to_pixel(w, tuple(transform)[:6]), xy, atol=1e-6)


def test_north_up_y_flip():
    """Row 0 is the tile's top (largest northing); the world y axis points up."""
    top_left = pixel_to_world([[0.0, 0.0]], NORTH_UP)[0]
    bottom_left = pixel_to_world([[0.0, 100.0]], NORTH_UP)[0]
    assert np.allclose(top_left, (391_000.0, 6_466_000.0))
    assert np.isclose(bottom_left[1], 6_466_000.0 - 30.0)
    # a building 3 m east and 6 m south of the corner: pixel (10, 20) onwards
    b = shapely.box(391_003.0, 6_465_988.0, 391_006.0, 6_465_994.0)
    (p,) = to_pixels([b], NORTH_UP, (100, 100))
    assert np.allclose(p.bounds, (10, 20, 20, 40))


def test_inputs_and_clipping():
    H = W = 100
    inside = shapely.box(10, 10, 30, 30)
    cut = shapely.box(90, 40, 120, 60)  # crosses the right border
    outside = shapely.box(150, 150, 160, 160)
    bowtie = shapely.Polygon([(40, 40), (60, 60), (60, 40), (40, 60)])  # invalid
    multi = shapely.MultiPolygon([shapely.box(5, 70, 15, 80), shapely.box(20, 70, 30, 80)])
    feats = [pixel_to_world(g, NORTH_UP) for g in (inside, cut, outside, bowtie, multi)]
    mappings = [{"type": "Feature", "properties": {"id": i},
                 "geometry": shapely.geometry.mapping(g)} for i, g in enumerate(feats)]
    mappings.insert(2, {"type": "Feature", "properties": {}, "geometry": None})
    for src in (feats, mappings, {"type": "FeatureCollection", "features": mappings}):
        out, idx = to_pixels(src, NORTH_UP, (H, W), return_index=True)
        assert all(p.geom_type == "Polygon" and p.is_valid for p in out)
        areas = sorted(round(p.area, 6) for p in out)
        assert areas == sorted([400.0, 200.0, 100.0, 100.0, 100.0, 100.0]), areas
        assert max(p.bounds[2] for p in out) <= W
    assert list(to_pixels(feats, NORTH_UP, (H, W), return_index=True)[1]) == [0, 1, 3, 3, 4, 4]
    # lines: a line leaving and re-entering the tile becomes two parts; vertices kept
    ln = shapely.LineString([(10.25, 50.5), (110, 50.5), (110, 70), (50, 70), (50, 90)])
    parts = to_pixels([pixel_to_world(ln, NORTH_UP)], NORTH_UP, (H, W))
    assert len(parts) == 2 and all(p.geom_type == "LineString" for p in parts)
    assert np.allclose(parts[0].coords[0], (10.25, 50.5), atol=1e-6)
    assert math.isclose(sum(p.length for p in parts), 89.75 + 50 + 20, abs_tol=1e-6)
    with pytest.raises(ValueError, match="mix"):
        to_pixels([feats[0], pixel_to_world(ln, NORTH_UP)], NORTH_UP, (H, W))
    assert len(to_pixels([feats[0], pixel_to_world(ln, NORTH_UP)], NORTH_UP, (H, W),
                         kind="lines")) == 2  # polygons dropped from a lines request
    assert to_pixels([], NORTH_UP, (H, W)) == []


def test_crs():
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("pyproj")
    px = [shapely.box(10.2, 20.3, 50.7, 60.1), shapely.Polygon([(70, 10), (95, 30), (80, 45)])]
    world = list(pixel_to_world(px, NORTH_UP))
    gdf = gpd.GeoDataFrame({"h": [3.0, 7.5]}, geometry=world, crs="EPSG:28350")
    for src in (gdf, gdf.to_crs(4326), gdf.to_crs(4326).geometry):
        back = to_pixels(src, NORTH_UP, (100, 100), raster_crs="EPSG:28350")
        assert len(back) == 2
        for p, q in zip(back, px):  # reprojection error well under a pixel
            assert p.symmetric_difference(q).area < 1e-3 * q.area
    # a features CRS given explicitly; same CRS (in another spelling) skips reprojection
    wgs = [g for g in gdf.to_crs(4326).geometry]
    assert len(to_pixels(wgs, NORTH_UP, (100, 100), crs=4326, raster_crs=28350)) == 2
    same = to_pixels(gdf, NORTH_UP, (100, 100), raster_crs="EPSG:28350")
    assert np.allclose(np.asarray(same[0].exterior.coords), np.asarray(px[0].exterior.coords))


# --- footprints: corner-line preset ------------------------------------------------------------

def _buildings(rng, size, n=12):
    """Rotated rectangles and L shapes, 15-50 px (5-15 m at 0.3 m), apart from each other."""
    out = []
    while len(out) < n:
        cx, cy = rng.uniform(30, size - 30, 2)
        w, h = rng.uniform(15, 50, 2)
        p = shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if len(out) % 3 == 2:  # L shape: a corner bitten out
            p = p.difference(shapely.box(cx, cy, cx + w, cy + h))
        p = shapely.affinity.rotate(p, rng.uniform(0, 90), origin="centroid")
        if all(p.distance(q) > 4 for q in out):
            out.append(p)
    return out


def test_footprint_targets_match_corner_line_task():
    """The arrays CornerLineTask reads, and the instance heads with its settings."""
    size, rng = 256, np.random.default_rng(1)
    polys = _buildings(rng, size)
    t = footprint_targets(polys, (size, size))
    assert {k: (v.shape, v.dtype.name) for k, v in t.items()} == {
        "instances": ((size, size), "int32"), "corner": ((size, size), "float32"),
        "direction": ((2, size, size), "float32"), "direction_mask": ((size, size), "bool"),
        "extent": ((size, size), "float32"), "boundary": ((size, size), "float32"),
        "distance": ((size, size), "float32")}
    assert int(t["instances"].max()) == len(polys)
    ref = instance_targets(torch.from_numpy(t["instances"]).long(), boundary_width=2)
    for k in ("extent", "boundary", "distance"):
        assert np.array_equal(t[k], ref[k][0, 0].numpy()), k
    ct = corner_targets(polys, (size, size))
    for k in ct:
        assert np.array_equal(t[k], ct[k]), k
    assert set(footprint_targets(polys, (size, size), instance_heads=False)) == {
        "instances", "corner", "direction", "direction_mask"}


def test_footprint_round_trip():
    """Targets -> instances_from_outputs-style labels -> polygons_from_corners: the polygons
    come back with their corners (E035b oracle: straight walls, sub-pixel corners)."""
    size, rng = 256, np.random.default_rng(2)
    polys = _buildings(rng, size)
    t = footprint_targets(polys, (size, size))
    pred = polygons_from_corners(t["instances"], t["corner"], t["direction"])
    assert len(pred) == len(polys)
    ious = _match(pred, polys)
    assert min(ious) > 0.93 and np.mean(ious) > 0.96, ious
    nv = [len(p.exterior.coords) - 1 for p in polys]
    got = [len(max(pred, key=lambda q: q.intersection(p).area).exterior.coords) - 1
           for p in polys]
    assert np.mean(np.array(nv) == np.array(got)) >= 0.9, (nv, got)
    for p in polys:  # each true corner has a decoded corner within 1.5 px
        q = max(pred, key=lambda r: r.intersection(p).area)
        dv = np.asarray(q.exterior.coords)[:-1]
        for c in np.asarray(p.exterior.coords)[:-1]:
            assert np.hypot(*(dv - c).T).min() < 1.5
    # the outline decoder (B0) from the same instances
    out = polygons_from_labels(t["instances"])
    assert len(out) == len(polys) and min(_match(out, polys)) > 0.9


def test_footprint_from_world_with_courtyard_and_border():
    """World features -> to_pixels -> targets: a courtyard stays a hole; a building cut by the
    tile border is clipped (its border corners are corners, as the experiments trained). The
    corner decoder returns outer rings (holes are not straightened); the outline decoder keeps
    the hole."""
    size = 128
    court = shapely.Polygon(shapely.box(10, 10, 70, 70).exterior.coords,
                            [shapely.box(30, 30, 50, 50).exterior.coords])
    cut = shapely.box(100, 20, 150, 60)
    world = [pixel_to_world(g, NORTH_UP) for g in (court, cut)]
    px = to_pixels(world, NORTH_UP, (size, size))
    t = footprint_targets(px, (size, size))
    assert t["instances"][40, 40] == 0 and t["instances"][20, 20] == 1  # hole kept
    assert t["instances"][40, size - 1] == 2
    clipped = cut.intersection(shapely.box(0, 0, size, size))
    pred = polygons_from_corners(t["instances"], t["corner"], t["direction"])
    ious = _match(pred, [shapely.Polygon(court.exterior), clipped])
    assert min(ious) > 0.95, ious
    out = polygons_from_labels(t["instances"])
    assert min(_match(out, [court, clipped])) > 0.95 and len(out[0].interiors) == 1


def test_overlapping_footprints_draw_only_visible_walls():
    """Where a later polygon covers an earlier one, the earlier one's hidden wall and corners
    are not targets (they would lie inside the later instance)."""
    size = 64
    a, b = shapely.box(10, 10, 30, 30), shapely.box(20, 10, 40, 30)
    t = footprint_targets([a, b], (size, size))
    assert (t["instances"][12:28, 21:39] == 2).all()
    assert not t["direction_mask"][14:26, 28:32].any()  # a's right wall (x = 30) is hidden
    assert t["corner"][10, 30] < 0.05 and t["corner"][29, 30] < 0.05
    assert t["corner"][10, 20] > 0.5  # b's corners and a's visible ones stay
    assert t["corner"][10, 10] > 0.5 and t["corner"][29, 39] > 0.5
    # touching faces are unaffected: same targets as corner_targets on the faces as given
    faces = _hip_roof(32, 32, 40, 24, 10)
    ct = corner_targets(faces, (size, size))
    t = roof_face_targets(faces, (size, size))
    assert all(np.array_equal(t[k], ct[k]) for k in ct)


def test_rasterize_instances_rules():
    size = 64
    big = shapely.box(4, 4, 40, 40)
    ring_field = big.difference(shapely.box(14, 14, 30, 30))  # a hole of background
    inner = shapely.box(18, 18, 26, 26)  # a field inside another's hole
    neck = shapely.box(44, 4, 60, 12).union(shapely.box(44, 30, 60, 40)).union(
        shapely.box(51.6, 12, 52.4, 30))  # two parts joined by a neck no pixel centre lies on
    tiny = shapely.box(2, 50, 5, 53)  # 9 px
    polys = [ring_field, inner, neck, tiny, shapely.box(200, 200, 210, 210)]
    lab, src = rasterize_instances(polys, (size, size))
    assert list(src) == [0, 1, 2, 3] and lab[20, 16] == 0
    lab, src = rasterize_instances(polys, (size, size), fill_holes=True, split=True,
                                   min_area=16)
    assert list(src) == [0, 1, 2, 2]  # neck split in two, tiny dropped
    assert lab[16, 16] == 1 and lab[20, 20] == 2  # hole filled around the inner field
    assert lab[51, 3] == 0
    # later polygons win where they overlap
    lab, _ = rasterize_instances([shapely.box(0, 0, 10, 10), shapely.box(5, 5, 15, 15)], (16, 16))
    assert lab[7, 7] == 2 and lab[2, 2] == 1


# --- fields: instance heads (B0) ---------------------------------------------------------------

def _fields(rng, size, n=14):
    """A partition of the tile into Voronoi parcels (shared boundaries), minus 2 px tracks."""
    from shapely.ops import voronoi_diagram

    pts = shapely.MultiPoint(rng.uniform(0, size, (n, 2)))
    tile = shapely.box(0, 0, size, size)
    cells = [c.intersection(tile) for c in voronoi_diagram(pts, envelope=tile).geoms]
    return [c.buffer(-1.0, join_style="mitre") for c in cells if c.area > 400]


def test_field_round_trip():
    size, rng = 256, np.random.default_rng(3)
    fields = _fields(rng, size)
    # one field with a background hole (a farmhouse yard): trained filled
    fields[0] = fields[0].difference(fields[0].centroid.buffer(6))
    t = field_targets(fields, (size, size))
    assert set(t) == {"instances", "extent", "boundary", "distance"}
    ref = instance_targets(torch.from_numpy(t["instances"]).long(), boundary_width=2)
    for k in ("extent", "boundary", "distance"):
        assert np.array_equal(t[k], ref[k][0, 0].numpy()), k
    c = fields[0].centroid
    assert t["instances"][int(c.y), int(c.x)] > 0  # hole filled
    pred = polygons_from_labels(t["instances"], simplify=1.0)
    want = [shapely.Polygon(f.exterior) for f in fields]
    assert len(pred) == len(fields)
    ious = _match(pred, want)
    assert min(ious) > 0.95, ious


# --- roof faces: corner decoder on touching faces ----------------------------------------------

def _hip_roof(cx, cy, w, h, angle):
    """Four faces of a hip roof (two trapezoids, two triangles) sharing edges."""
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    r = (h / 2) * 0.9
    a, b = (x0 + r, cy), (x1 - r, cy)
    faces = [shapely.Polygon([(x0, y0), (x1, y0), b, a]), shapely.Polygon([(x1, y0), (x1, y1), b]),
             shapely.Polygon([(x1, y1), (x0, y1), a, b]), shapely.Polygon([(x0, y1), (x0, y0), a])]
    return [shapely.affinity.rotate(f, angle, origin=(cx, cy)) for f in faces]


def test_roof_face_round_trip():
    size = 256
    faces = (_hip_roof(70, 70, 90, 50, 12) + _hip_roof(180, 170, 70, 46, -30)
             + _hip_roof(70, 190, 60, 40, 60))
    t = roof_face_targets(faces, (size, size))
    assert int(t["instances"].max()) == len(faces)  # touching faces stay distinct
    # the boundary target separates touching faces: no instance pixel without a boundary
    # pixel on its side of a shared edge
    a, b = t["instances"][70, 69], t["instances"][60, 69]
    assert a != b and a > 0 and b > 0
    pred = polygons_from_corners(t["instances"], t["corner"], t["direction"])
    assert len(pred) == len(faces)
    ious = _match(pred, faces)
    assert min(ious) > 0.88 and np.mean(ious) > 0.93, ious
    assert np.mean([len(p.exterior.coords) - 1 for p in pred]) <= 4.5


# --- roads: graph_from_road_heads --------------------------------------------------------------

def _road_lines():
    """A T junction (three ways sharing a node), a bend, a dead end and an overpass that crosses
    the main road without a node (a 512 px tile at 1 m)."""
    main_w = shapely.LineString([(-20, 250.3), (120.4, 248.1), (255.7, 252.2)])
    main_e = shapely.LineString([(255.7, 252.2), (400.2, 262.5), (530, 270)])
    branch = shapely.LineString([(255.7, 252.2), (262.1, 380.4), (340.6, 450.9)])  # dead end
    over = shapely.LineString([(160.3, -10), (190.5, 200.2), (205.1, 530)])  # bridge
    side = shapely.LineString([(400.2, 262.5), (420.3, 120.7), (470.1, 60.2)])  # 2nd junction
    return [main_w, main_e, branch, over, side]


def _covered(a_lines, b_graph, tol):
    """Fraction of points sampled every 2 px along a_lines within tol px of b_graph's edges."""
    segs = shapely.MultiLineString([b_graph.xy[e] for e in b_graph.edges])
    pts = [ln.interpolate(s) for ln in a_lines for s in np.arange(0, ln.length, 2.0)]
    return float(np.mean([segs.distance(p) <= tol for p in pts]))


def test_road_graph_topology():
    size = 512
    lines = to_pixels([pixel_to_world(ln, NORTH_UP) for ln in _road_lines()], NORTH_UP,
                      (size, size))
    g, cr = road_graph(lines, widths=[6, 6, 5, 8, 5])
    deg3 = [g.nodes[n]["xy"] for n in g if g.degree(n) >= 3]
    assert len(deg3) == 2
    assert min(np.hypot(*(np.array(deg3) - (255.7, 252.2)).T)) < 1e-6
    assert cr.shape == (1, 2)
    p = shapely.LineString([(120.4, 248.1), (255.7, 252.2)]).intersection(
        shapely.LineString([(160.3, -10), (190.5, 200.2), (205.1, 530)]))
    assert np.allclose(cr[0], (p.x, p.y))
    widths = sorted({d["width"] for *_, d in g.edges(data=True)})
    assert widths == [5.0, 6.0, 8.0]
    # node=True: the overpass becomes a junction, no crossings
    g2, cr2 = road_graph(lines, node=True)
    assert len(cr2) == 0 and sum(g2.degree(n) >= 3 for n in g2) == 3
    assert road_graph([])[0].number_of_nodes() == 0


def test_road_round_trip():
    """World lines -> to_pixels -> road_graph -> road_targets -> graph_from_road_heads: the
    roads come back, the junctions are junctions, the overpass stays two roads."""
    size = 512
    lines = to_pixels([pixel_to_world(ln, NORTH_UP) for ln in _road_lines()], NORTH_UP,
                      (size, size))
    g, cr = road_graph(lines, widths=[6, 6, 6, 8, 6])
    t = road_targets(g, (size, size), crossings=cr)
    assert t["keypoints"][2].max() > 0.9 and t["keypoints"][0].max() > 0.9
    vg = graph_from_road_heads(t["mask"], t["keypoints"], t["direction"])
    assert _covered(lines, vg, 3.0) > 0.97  # recall
    assert _covered([shapely.LineString(vg.xy[e]) for e in vg.edges], _as_vg(g), 3.0) > 0.97
    j = vg.junctions()
    for want in ((255.7, 252.2), (400.2, 262.5)):  # both junctions found within 4 px
        assert len(j) and np.hypot(*(j - want).T).min() < 4, j
    near = np.hypot(*(vg.xy - cr[0]).T) < 10  # no junction at the overpass
    assert not (vg.degree[near] >= 3).any()
    assert len(j) == 2
    # the dead end decodes as an end near (340.6, 450.9)
    ends = vg.xy[vg.degree == 1]
    assert np.hypot(*(ends - (340.6, 450.9)).T).min() < 4


def _as_vg(g):
    from manyseg.vectors import VectorGraph

    return VectorGraph.from_networkx(g)


# --- points: manyseg.points <-> GeoJSON / CRS -------------------------------------------------

@pytest.mark.parametrize("transform", [NORTH_UP, ROTATED])
def test_points_to_geojson_round_trip(transform):
    """find_peaks (y, x) -> points_to_xy -> to_geojson -> world; checked against rasterio's own
    pixel-centre mapping (not just a round trip, where a shared half-pixel error cancels)."""
    import json

    from manyseg.points import find_peaks, gaussian_heatmap, points_from_xy, points_to_xy
    from rasterio.transform import xy as rio_xy

    yx = torch.tensor([[2.0, 3.0], [40.0, 7.0], [0.0, 63.0]])
    ((peaks, scores),) = find_peaks(gaussian_heatmap([yx], (50, 64)), 0.5)
    xy = points_to_xy(peaks)
    fc = to_geojson(xy, transform, properties={"score": scores.numpy()})
    json.dumps(fc)  # numpy values became Python numbers
    world = np.array([f["geometry"]["coordinates"] for f in fc["features"]])
    assert all(f["geometry"]["type"] == "Point" for f in fc["features"])
    rows, cols = peaks[:, 0].numpy(), peaks[:, 1].numpy()
    rx, ry = rio_xy(transform, rows, cols, offset="center")
    assert np.allclose(world, np.stack([rx, ry], 1))
    assert np.allclose(world, pixel_to_world(xy, transform))
    assert [f["properties"]["score"] for f in fc["features"]] == pytest.approx([1.0] * 3)
    # and back: the same features in, the same pixels out
    back = to_pixels(fc, transform, (50, 64))
    assert np.allclose(back, xy, atol=1e-6)
    assert torch.allclose(points_from_xy(back), peaks.float(), atol=1e-6)


def test_to_geojson_point_inputs():
    pts = [(1.5, 2.5), (3.0, 4.0)]
    t = (2, 0, 100, 0, -2, 50)
    fc = to_geojson(pts, t, properties=[{"id": 1}, {"id": 2}])
    assert [f["geometry"]["coordinates"] for f in fc["features"]] == [[103, 45], [106, 42]]
    assert fc == to_geojson(np.array(pts), t, properties=[{"id": 1}, {"id": 2}])
    assert fc == to_geojson((p for p in pts), t, properties=[{"id": 1}, {"id": 2}])
    g = to_geojson([shapely.Point(1.5, 2.5), shapely.MultiPoint(pts),
                    shapely.box(0, 0, 1, 1)], t)
    assert [f["geometry"]["type"] for f in g["features"]] == ["Point", "MultiPoint", "Polygon"]
    assert g["features"][1]["geometry"]["coordinates"] == [[103, 45], [106, 42]]
    assert to_geojson(shapely.Point(0, 0))["features"][0]["geometry"]["coordinates"] == [0, 0]
    assert to_geojson(np.zeros((0, 2)))["features"] == []
    with pytest.raises(TypeError, match="points_to_xy"):  # (y, x) tensors would be swapped
        to_geojson(torch.zeros(3, 2))
    with pytest.raises(ValueError, match="2 values for 1"):
        to_geojson([(0, 0)], properties={"score": [1, 2]})
    with pytest.raises(ValueError, match="shape"):
        to_geojson(np.zeros((3, 3)))


def test_to_pixels_points():
    H, W = 40, 60
    px = np.array([[0.5, 0.5], [59.9, 39.9], [0.0, 0.0], [12.25, 30.75],
                   [60.001, 10.0], [10.0, 40.001], [-0.01, 5.0], [200.0, 200.0]])
    inside = px[:4]
    world = pixel_to_world(px, ROTATED)
    feats = [shapely.Point(*w) for w in world[:3]] + [shapely.MultiPoint(world[3:6])] + \
        [shapely.Point(*w) for w in world[6:]]
    xy, idx = to_pixels(feats, ROTATED, (H, W), return_index=True)
    assert xy.shape == (4, 2) and xy.dtype == np.float64
    assert np.allclose(xy, inside, atol=1e-6)
    assert idx.tolist() == [0, 1, 2, 3]  # the MultiPoint's one inside part keeps index 3
    assert np.allclose(to_pixels(feats, ROTATED, (H, W), kind="points"), inside, atol=1e-6)
    empty = to_pixels([], ROTATED, (H, W), kind="points")
    assert empty.shape == (0, 2)
    xy, idx = to_pixels([shapely.Point(*world[7])], ROTATED, (H, W), return_index=True)
    assert xy.shape == (0, 2) and idx.shape == (0,)
    with pytest.raises(ValueError, match="mix"):
        to_pixels([feats[0], shapely.box(*world[0], *world[1])], ROTATED, (H, W))
    # with kind=, other geometry types are ignored, as for lines / polygons
    assert len(to_pixels([feats[0], pixel_to_world(shapely.box(1, 1, 5, 5), ROTATED)],
                         ROTATED, (H, W), kind="points")) == 1


def test_to_pixels_points_reprojected():
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("pyproj")
    px = np.array([[10.5, 20.5], [99.5, 0.5], [50.0, 99.99], [100.5, 50.0], [-3.0, 3.0]])
    gdf = gpd.GeoDataFrame({"h": range(5)}, geometry=gpd.points_from_xy(
        *pixel_to_world(px, NORTH_UP).T), crs="EPSG:28350")
    for src in (gdf.to_crs(4326), gdf.to_crs(3857).geometry):
        xy, idx = to_pixels(src, NORTH_UP, (100, 100), raster_crs="EPSG:28350",
                            return_index=True)
        assert idx.tolist() == [0, 1, 2]
        assert np.allclose(xy, px[:3], atol=1e-4)  # 1e-4 px = 0.03 mm
    # raster CRS unknown: taken to be in the features' CRS (degrees here: all outside the tile)
    assert len(to_pixels(gdf.to_crs(4326), NORTH_UP, (100, 100))) == 0


def test_scene_points_to_geojson_uses_scene_transform():
    from manyseg.points import ScenePoints

    sp = ScenePoints(np.array([[2.0, 3.0], [4.25, 5.5]]), np.array([0.9, 0.8], np.float32),
                     np.array([[2, 3], [4, 5]]), transform=NORTH_UP)
    fc = sp.to_geojson(properties={"id": [7, 8]})
    world = np.array([f["geometry"]["coordinates"] for f in fc["features"]])
    assert np.allclose(world, [NORTH_UP @ (3.5, 2.5), NORTH_UP @ (6.0, 4.75)])
    assert fc["features"][1]["properties"] == {"score": pytest.approx(0.8), "id": 8}
    assert sp.to_geojson(transform=(1, 0, 0, 0, 1, 0))["features"][0]["geometry"][
        "coordinates"] == [3.5, 2.5]
