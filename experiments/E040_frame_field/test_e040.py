"""Frame-field corner detection must not depend on root order (Codex E040 #1)."""
import numpy as np
import shapely
from shapely.geometry import Polygon, box

import e040 as F


def ring_of(poly):
    r = np.asarray(shapely.segmentize(poly.exterior, 1.0).coords)
    return r[:-1]


def test_rectangle_has_four_corners_and_45deg_noise_none():
    S = (200, 200)
    rect = box(40.3, 50.6, 150.2, 120.4)
    f = F.frame_targets([rect], S)
    idx = F.field_corners(F.ring_dirs(ring_of(rect), f), 9)
    assert len(idx) == 4
    diamond = Polygon([(100, 20), (180, 100), (100, 180), (20, 100)])  # 45 deg walls: c0 = 1, c2 = 0
    f = F.frame_targets([diamond], S)
    rng = np.random.default_rng(0)
    f = f + rng.normal(0, 1e-6, f.shape).astype(np.float32) * (np.hypot(f[0], f[1]) > 0)
    idx = F.field_corners(F.ring_dirs(ring_of(diamond), f), 9)
    assert len(idx) == 4  # the four real corners only, no branch-cut corners along the walls
