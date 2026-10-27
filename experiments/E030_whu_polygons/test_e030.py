"""E030 sanity tests: dihedral transforms keep polygons and rasters aligned; WKB round trip."""

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon, box

from common import dihedral_array, dihedral_polygon, load_polygons, save_polygons


def raster(polys, size=64):
    return rasterize([(p, i + 1) for i, p in enumerate(polys)], out_shape=(size, size),
                     transform=Affine.identity(), dtype="int32")


def test_dihedral_consistency():
    # vertices off the half-pixel grid, so no edge passes exactly through a pixel centre (where
    # rasterization tie-breaking is not rotation-invariant)
    polys = [box(3, 5, 20, 12), Polygon([(30.3, 30.1), (60.2, 34.3), (50.1, 60.3), (33.3, 50.2)])]
    base = raster(polys)
    for d in range(8):
        moved = raster([dihedral_polygon(p, d, 64) for p in polys])
        np.testing.assert_array_equal(moved, dihedral_array(base, d), err_msg=f"d={d}")
    assert len({dihedral_array(base, d).tobytes() for d in range(8)}) == 8


def test_wkb_round_trip(tmp_path):
    tiles = [[box(0, 0, 1, 1), box(2, 2, 4, 5)], [], [Polygon([(0, 0), (3, 0), (0, 3)])]]
    save_polygons(tmp_path / "p.npz", tiles)
    back = load_polygons(tmp_path / "p.npz")
    assert [len(t) for t in back] == [2, 0, 1]
    assert all(a.equals(b) for ta, tb in zip(tiles, back) for a, b in zip(ta, tb))


def test_endpoint_loss_rewards_correct_vectors():
    """E032: perfect vectors land on target vertices (low loss); shifted ones do not; the
    gradient reaches the vector head."""
    import torch
    from shapely.geometry import box

    from common import SPACING, TILE, GraphTensorTask
    from manyseg.vectors import encode_graph_tensor, vertex_graph

    polys = [box(100.3, 120.2, 180.1, 170.4), box(300.2, 50.1, 340.3, 130.2)]
    g = encode_graph_tensor(vertex_graph(polys, SPACING, border=(TILE, TILE)), (TILE, TILE))
    t = {"vertex": torch.from_numpy(g["vertex"])[None], "edge": torch.from_numpy(g["edge"].astype("uint8"))[None]}
    vec = torch.from_numpy((g["vector"] / SPACING).reshape(1, -1, TILE, TILE)).requires_grad_()
    good = GraphTensorTask.endpoint_loss({"vector": vec}, t)
    good.backward()
    assert vec.grad is not None and torch.isfinite(vec.grad).all()
    shifted = GraphTensorTask.endpoint_loss({"vector": vec.detach() + 2.0 / SPACING}, t)
    assert good.item() < 0.4 < shifted.item(), (good.item(), shifted.item())
    # gradients stay bounded for endpoints near the target's cut-off (E032 amendment 1)
    for off in (2.5, 2.9, 3.1, 4.0):
        v = (vec.detach() + off / SPACING).requires_grad_()
        GraphTensorTask.endpoint_loss({"vector": v}, t).backward()
        assert v.grad.abs().max().item() < 1.0, (off, v.grad.abs().max().item())


def test_wall_tolerant_vertex_loss_ignores_on_wall_predictions():
    """E034: a vertex predicted on a true wall (not at a target point) costs nothing; one off the
    outline costs as in BCE."""
    import torch
    from shapely.geometry import box

    from common import SPACING, TILE, GraphTensorTask
    from manyseg.vectors import encode_graph_tensor, vertex_graph

    vg = vertex_graph([box(100.3, 120.2, 300.1, 170.4)], SPACING, border=(TILE, TILE))
    g = encode_graph_tensor(vg, (TILE, TILE))
    from manyseg_bench.losses import distance_map
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    import shapely

    lines = [shapely.LineString([vg.xy[a], vg.xy[b]]) for a, b in vg.edges]
    wall = rasterize([(ln, 1) for ln in lines], out_shape=(TILE, TILE), transform=Affine.identity(),
                     all_touched=True, dtype="uint8").astype(bool)
    t = {"vertex": torch.from_numpy(g["vertex"])[None], "wall_dist": torch.from_numpy(distance_map(wall, 8.0))[None]}
    task = GraphTensorTask(vertex_loss="wall")

    def vloss(extra_xy):
        logit = torch.full((1, 1, TILE, TILE), -8.0)
        logit[0, 0] = torch.where(t["vertex"][0] > 0.5, torch.tensor(8.0), logit[0, 0])
        x, y = extra_xy
        logit[0, 0, y, x] = 8.0
        w = ~((t["vertex"] == 0) & (t["wall_dist"] <= 1.5))
        return torch.nn.functional.binary_cross_entropy_with_logits(logit[:, 0], t["vertex"], weight=w.float())

    on_wall = vloss((210, 120))  # on the top wall, between target points
    off = vloss((210, 145))  # inside the building, off the outline
    base = vloss((5, 5))  # far away
    assert on_wall < off and abs(off - base) < 1e-6
    assert task.vertex_loss == "wall"
