"""Tile-invariant target options (experimental): ``corner_targets(tile_edges="ignore")`` and
``CornerLineTask``'s ``corner_ignore`` label, ``distance="absolute"``. Measured end to end in
``tests/integration/test_vector_pipeline.py``."""

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("rasterio")

import shapely  # noqa: E402
from manyseg.vectors import corner_targets, footprint_targets  # noqa: E402


def _clipped():
    """A 40 x 20 building cut by the tile's right edge at x = 64 (a 64 x 64 tile)."""
    return shapely.box(40, 20, 80, 40).intersection(shapely.box(0, 0, 64, 64))


def test_keep_is_unchanged():
    t = corner_targets([_clipped()], (64, 64))
    assert set(t) == {"corner", "direction", "direction_mask"}
    assert t["corner"][19:21, 62:64].max() > 0.7  # clip-edge corners, as trained so far
    assert t["direction_mask"][25:35, 63].all()


def test_ignore_drops_clip_corners_and_walls():
    keep = corner_targets([_clipped()], (64, 64))
    t = corner_targets([_clipped()], (64, 64), tile_edges="ignore")
    # the real corners (x = 40) are untouched
    np.testing.assert_array_equal(t["corner"][:, :50], keep["corner"][:, :50])
    # no corner where the polygon meets the tile edge; those pixels are ignored instead
    assert t["corner"][14:26, 56:64].max() < 1e-6
    assert t["corner_ignore"][20, 63] and t["corner_ignore"][40, 63]
    assert not t["corner_ignore"][:, :50].any()
    # no wall along the tile edge, the real walls (top, bottom, left) keep their direction
    assert not t["direction_mask"][25:35, 63].any()
    assert t["direction_mask"][20, 50] and t["direction_mask"][30, 40]
    # footprint_targets passes the option through
    f = footprint_targets([_clipped()], (64, 64), tile_edges="ignore")
    np.testing.assert_array_equal(f["corner_ignore"], t["corner_ignore"])
    with pytest.raises(ValueError, match="tile_edges"):
        corner_targets([_clipped()], (64, 64), tile_edges="drop")


def test_ignore_keeps_a_polygon_inside_the_tile():
    p = shapely.box(10, 10, 30, 25)
    keep = corner_targets([p], (64, 64))
    t = corner_targets([p], (64, 64), tile_edges="ignore")
    for k in keep:
        np.testing.assert_array_equal(t[k], keep[k])
    assert not t["corner_ignore"].any()


def test_corner_line_task_ignores_corner_ignore_pixels():
    bench = pytest.importorskip("manyseg_bench.tasks")
    t = footprint_targets([_clipped()], (64, 64), tile_edges="ignore")
    y = {k: torch.as_tensor(np.asarray(v))[None] for k, v in t.items()
         if k in ("instances", "corner", "direction", "direction_mask", "corner_ignore")}
    task = bench.CornerLineTask()
    out = {"extent": torch.zeros(1, 1, 64, 64), "boundary": torch.zeros(1, 1, 64, 64),
           "distance": torch.zeros(1, 1, 64, 64), "corner": torch.full((1, 1, 64, 64), -5.0),
           "direction": torch.zeros(1, 2, 64, 64)}
    base = task.losses(out, task.targets(y))["corner"]
    # firing on the ignored pixels costs nothing; firing on plain background does
    hot = out | {"corner": out["corner"].masked_fill(y["corner_ignore"][:, None], 5.0)}
    assert torch.isclose(task.losses(hot, task.targets(y))["corner"], base)
    bg = torch.zeros_like(y["corner_ignore"])
    bg[:, 50:60, 5:15] = True
    hot = out | {"corner": out["corner"].masked_fill(bg[:, None], 5.0)}
    assert task.losses(hot, task.targets(y))["corner"] > base + 1
    # without the label the same pixels are negatives
    y2 = {k: v for k, v in y.items() if k != "corner_ignore"}
    hot = out | {"corner": out["corner"].masked_fill(y["corner_ignore"][:, None], 5.0)}
    assert task.losses(hot, task.targets(y2))["corner"] > base + 1


def test_corner_line_task_absolute_distance():
    bench = pytest.importorskip("manyseg_bench.tasks")
    t = footprint_targets([_clipped()], (64, 64), distance="absolute")
    y = torch.as_tensor(t["instances"])[None]
    d = bench.CornerLineTask(distance="absolute").targets({"instances": y})["inner"]["distance"]
    np.testing.assert_allclose(d[0, 0].numpy(), t["distance"], atol=1e-5)
    assert float(d.max()) > 5  # pixels, not normalised
