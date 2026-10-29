"""Point coordinates between :mod:`manyseg.points` and :mod:`manyseg.vectors` (GeoJSON, CRS).

The two use different conventions, and these two functions are the only place they meet:

* :mod:`manyseg.points` (targets, :func:`find_peaks`): ``(y, x)``, **pixel centres at
  integers** (pixel ``(row, col)`` is the point ``(row, col)``).
* :mod:`manyseg.vectors` (:func:`~manyseg.vectors.to_geojson`,
  :func:`~manyseg.vectors.to_pixels`, :func:`~manyseg.vectors.pixel_to_world`): ``(x, y)``,
  **pixel corners at integers** (the centre of pixel ``(row, col)`` is ``(col + 0.5,
  row + 0.5)``), so a rasterio / ``affine.Affine`` geotransform applies directly.

So ``xy = yx[:, ::-1] + 0.5``. Getting this wrong shifts every point by half a pixel or
mirrors it about the diagonal; both pass a self-consistent round trip, so convert only here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def points_to_xy(points) -> np.ndarray:
    """``(N, 2)`` ``(y, x)`` points (centres at integers; a tensor from :func:`find_peaks` or an
    array) -> ``(N, 2)`` float64 ``(x, y)`` with pixel corners at integers, for
    :func:`manyseg.vectors.to_geojson` / :func:`manyseg.vectors.pixel_to_world`."""
    p = points.detach().cpu().numpy() if isinstance(points, Tensor) else np.asarray(points)
    p = p.astype(np.float64).reshape(-1, 2)
    return p[:, ::-1] + 0.5


def points_from_xy(xy) -> Tensor:
    """``(N, 2)`` ``(x, y)`` with pixel corners at integers (e.g. from
    ``manyseg.vectors.to_pixels(..., kind="points")``) -> a float32 ``(N, 2)`` ``(y, x)`` tensor
    with pixel centres at integers, one image's points for :func:`gaussian_heatmap` /
    :func:`density_map` (pass a list of them, one per image)::

        hm = gaussian_heatmap([points_from_xy(xy)], (H, W), sigma=2.0)   # (1, 1, H, W)
    """
    a = np.asarray(xy, np.float64).reshape(-1, 2)
    return torch.from_numpy(np.ascontiguousarray(a[:, ::-1] - 0.5)).float()
