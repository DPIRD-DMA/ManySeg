"""Regressions: huge instance ids, to_geojson geometry types, empty batches, missing extras."""

import sys

import numpy as np
import pytest
import torch
from manyseg.instances import instance_targets, instances_from_outputs
from manyseg.instances.targets import edge_pixels
from manyseg.points import offset_targets

HUGE = 1_000_000_000  # a scene-wide id kept by a small crop


@pytest.fixture
def small_tables(monkeypatch):
    """Fail (instead of allocating gigabytes) if a table sized by the max id is created."""
    zeros, full = torch.zeros, torch.full

    def check(size):
        n = 1
        for s in (size[0] if len(size) == 1 and not isinstance(size[0], int) else size):
            n *= int(s)
        assert n <= 10**6, f"table of {n} elements for a 12 x 12 mask"

    def guarded_zeros(*size, **kw):
        check(size)
        return zeros(*size, **kw)

    def guarded_full(size, *a, **kw):
        check((size,) if isinstance(size, int) else tuple(size))
        return full(size, *a, **kw)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    monkeypatch.setattr(torch, "full", guarded_full)


def _two_instances(a: int, b: int) -> torch.Tensor:
    m = torch.zeros(2, 12, 12, dtype=torch.long)
    m[0, 2:7, 2:9] = a
    m[0, 7:11, 2:9] = b  # touches instance a
    m[1, 3:9, 3:9] = a
    return m


@pytest.mark.parametrize("kw", [{}, {"boundary_mode": "both", "distance": "absolute"}])
def test_instance_targets_huge_ids_compacted(kw, small_tables):
    small = instance_targets(_two_instances(1, 2), **kw)
    huge = instance_targets(_two_instances(HUGE, HUGE + 1), **kw)
    for k in small:
        assert torch.equal(small[k], huge[k]), k


def test_offset_targets_huge_ids_compacted(small_tables):
    assert torch.equal(offset_targets(_two_instances(1, 2)),
                       offset_targets(_two_instances(HUGE, HUGE + 1)))


def test_edge_pixels_keep_adjacent_ids_above_float32_precision():
    # 16777216 and 16777217 are the same float32: a float round trip merges them
    small = edge_pixels(_two_instances(1, 2))
    big = edge_pixels(_two_instances(16777216, 16777217))
    assert torch.equal(small, big)
    assert big[0, 6, 4] and big[0, 7, 4]  # the shared boundary is there


@pytest.mark.parametrize("backend", ["torch", "auto"])
@pytest.mark.parametrize("ids", ["sequential", "anchor"])
def test_instances_from_outputs_empty_batch(backend, ids):
    z = torch.zeros(0, 1, 16, 16)
    out = instances_from_outputs(z, z, z, logits=False, backend=backend, ids=ids)
    assert out.shape == (0, 16, 16) and out.dtype == torch.long


# --- vectors -------------------------------------------------------------------------------

def test_pixel_to_world_without_networkx_or_skimage(monkeypatch):
    from manyseg.vectors import pixel_to_world, world_to_pixel

    for name in ("networkx", "skimage", "skimage.morphology"):
        monkeypatch.setitem(sys.modules, name, None)
    t = (0.5, 0.0, 100.0, 0.0, -0.5, 200.0)
    xy = np.array([[0.0, 0.0], [4.0, 2.0]])
    w = pixel_to_world(xy, t)
    np.testing.assert_allclose(w, [[100.0, 200.0], [102.0, 199.0]])
    np.testing.assert_allclose(world_to_pixel(w, t), xy)


def test_polygons_from_labels_missing_scipy_message(monkeypatch):
    pytest.importorskip("shapely")
    from manyseg.vectors import polygons_from_labels

    for name in ("scipy", "scipy.ndimage", "scipy.spatial"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"pip install 'manyseg\[vectors\]'"):
        polygons_from_labels(np.ones((4, 4), np.int64))


def test_instances_skimage_backend_missing_message(monkeypatch):
    for name in ("skimage", "skimage.segmentation"):
        monkeypatch.setitem(sys.modules, name, None)
    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(ImportError, match=r"pip install 'manyseg\[instances\]'"):
        instances_from_outputs(x, x, x, logits=False, backend="skimage")


def test_to_geojson_all_geometry_types():
    shapely = pytest.importorskip("shapely")
    from manyseg.vectors import to_geojson

    geoms = [
        shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)]),
        shapely.LineString([(0, 0), (2, 1)]),
        shapely.MultiLineString([[(0, 0), (1, 0)], [(0, 1), (1, 1)]]),
        shapely.GeometryCollection([shapely.Point(1, 2), shapely.LineString([(0, 0), (1, 1)])]),
        shapely.box(0, 0, 2, 2).difference(shapely.box(0.5, 0.5, 1, 1)),
        shapely.Point(1, 1),
    ]
    t = (2.0, 0.0, 10.0, 0.0, -2.0, 50.0)
    fc = to_geojson(geoms, t)
    got = [f["geometry"] for f in fc["features"]]
    assert [g["type"] for g in got] == [g.geom_type for g in geoms]
    assert got[1]["coordinates"] == [[10.0, 50.0], [14.0, 48.0]]
    assert got[3]["geometries"][0] == {"type": "Point", "coordinates": [12.0, 46.0]}
    assert got[5]["coordinates"] == [12.0, 48.0]
    mp = shapely.geometry.shape(got[0])
    assert mp.equals(shapely.MultiPolygon([shapely.box(10, 48, 12, 50),
                                           shapely.box(14, 44, 16, 46)]))
    assert len(got[4]["coordinates"]) == 2  # exterior + hole
    # pixel coordinates without a transform
    line = to_geojson([geoms[1]])["features"][0]["geometry"]
    assert line["coordinates"] == [[0.0, 0.0], [2.0, 1.0]]
