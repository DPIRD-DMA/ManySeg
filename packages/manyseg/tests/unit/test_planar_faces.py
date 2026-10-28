"""planar_faces (E048b): neighbouring faces end up sharing edges and vertices."""
import pytest

shapely = pytest.importorskip("shapely")
from manyseg.vectors import planar_faces  # noqa: E402


def _shared_vertices(faces):
    seen: dict = {}
    for i, f in enumerate(faces):
        for x, y in list(f.exterior.coords)[:-1]:
            seen.setdefault((round(x, 6), round(y, 6)), set()).add(i)
    return sum(1 for s in seen.values() if len(s) >= 2)


def test_gapped_neighbours_share_edges():
    a = shapely.box(0, 0, 10, 10)
    b = shapely.Polygon([(10.8, 0), (20, 0), (20, 10), (10.6, 10)])
    c = shapely.box(0, 10.5, 20, 15)
    out = planar_faces([a, b, c], tolerance=1.5)
    assert len(out) == 3
    assert _shared_vertices(out) >= 4
    assert out[0].intersection(out[1]).length > 9  # a shared wall, not a crack
    assert sum(p.area for p in out) == pytest.approx(shapely.union_all(out).area)  # no overlaps


def test_far_faces_unchanged():
    a, b = shapely.box(0, 0, 10, 10), shapely.box(20, 0, 30, 10)
    out = planar_faces([a, b], tolerance=2.5)
    assert [p.equals(q) for p, q in zip(out, [a, b])] == [True, True]
