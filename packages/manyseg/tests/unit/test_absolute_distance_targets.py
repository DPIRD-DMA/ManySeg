"""Absolute distance targets (``distance="absolute"``; the default of ``field_targets``,
E052): the bench task's targets equal the library's. Measured end to end in
``tests/integration/test_vector_pipeline.py``. (``corner_targets(tile_edges="ignore")`` was
tried there too, changed nothing measurable, and was dropped.)"""

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("rasterio")

import shapely  # noqa: E402
from manyseg.vectors import field_targets, footprint_targets  # noqa: E402


def _clipped():
    """A 40 x 20 building cut by the tile's right edge at x = 64 (a 64 x 64 tile)."""
    return shapely.box(40, 20, 80, 40).intersection(shapely.box(0, 0, 64, 64))


def test_corner_line_task_absolute_distance():
    bench = pytest.importorskip("manyseg_bench.tasks")
    t = footprint_targets([_clipped()], (64, 64), distance="absolute")
    y = torch.as_tensor(t["instances"])[None]
    d = bench.CornerLineTask(distance="absolute").targets({"instances": y})["inner"]["distance"]
    np.testing.assert_allclose(d[0, 0].numpy(), t["distance"], atol=1e-5)
    assert float(d.max()) > 5  # pixels, not normalised


def test_field_targets_default_to_absolute_distance():
    t = field_targets([_clipped()], (64, 64))
    assert float(t["distance"].max()) > 5  # pixels, not normalised
    n = field_targets([_clipped()], (64, 64), distance="normalized")
    assert float(n["distance"].max()) == pytest.approx(1.0)
