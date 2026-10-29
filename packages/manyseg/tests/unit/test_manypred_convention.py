"""Every ``manyseg.vectors.block_decoder`` task under ManyPred's calling convention
(``manypred.blocks``: ``postprocess_blocks`` / ``BlockPostprocess``): its own frozen ``Block``
(``index``, ``core``, ``window``, ``scene_edges``) and ``Scene`` (``height``, ``width``,
``transform``, ``crs``) classes, its grid (no offset), ``fn(window_fields, block, scene)`` with
each field a float32 ``(C, h, w)`` copy (3-D even for one channel), ``combine(results, scene)``
once, and a pickle round trip of the decoder and each result (``workers > 0``). ManyPred itself
is not needed: the driver below follows its code. The result equals ``decode_scene`` on the
same grid (whose equality with the whole-scene decode the per-task tests check)."""

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("shapely")
pytest.importorskip("skimage")
pytest.importorskip("networkx")

import shapely  # noqa: E402
from manyseg.vectors import block_decoder, decode_scene, records_to_geojson  # noqa: E402


@dataclass(frozen=True)
class MPBlock:  # manypred.blocks.Block (its ``inner`` property left out: decoders must not need it)
    index: int
    core: tuple[int, int, int, int]
    window: tuple[int, int, int, int]
    scene_edges: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class MPScene:  # manypred.blocks.Scene
    height: int
    width: int
    transform: Any = None
    crs: Any = None


def mp_block_grid(height, width, block, halo):  # manypred.blocks.block_grid
    blocks = []
    for top in range(0, height, block):
        bottom = min(top + block, height)
        for left in range(0, width, block):
            right = min(left + block, width)
            window = (max(top - halo, 0), min(bottom + halo, height),
                      max(left - halo, 0), min(right + halo, width))
            edges = (window[0] == 0, window[1] == height, window[2] == 0, window[3] == width)
            blocks.append(MPBlock(len(blocks), (top, bottom, left, right), window, edges))
    return blocks


def mp_postprocess(fields, fn, *, block, halo, combine, scene):
    """BlockPostprocess.run -> postprocess_blocks (workers > 0: fn and results pickled)."""
    fn = pickle.loads(pickle.dumps(fn))
    results = []
    for b in mp_block_grid(scene.height, scene.width, block, halo):
        t, bt, lf, rt = b.window
        win = {k: np.array(v[None, t:bt, lf:rt] if v.ndim == 2 else v[:, t:bt, lf:rt],
                           dtype=np.float32) for k, v in fields.items()}  # manypred _cut
        assert all(w.ndim == 3 and w.dtype == np.float32 for w in win.values())
        results.append(pickle.loads(pickle.dumps(fn(win, b, scene))))
    return combine(results, scene)


def _transform():
    affine = pytest.importorskip("affine")
    return affine.Affine(0.5, 0.0, 400000.0, 0.0, -0.5, 6500000.0)


def _canon(records):
    return sorted(shapely.normalize(r["geometry"]).wkb for r in records)


def _check(task, fields, block, halo, same, **params):
    H, W = fields[next(iter(fields))].shape[-2:]
    scene = MPScene(H, W, _transform(), "EPSG:28350")
    dec = block_decoder(task, **params)
    out = mp_postprocess(fields, dec, block=block, halo=halo, combine=dec.combine, scene=scene)
    ref = decode_scene(fields, block_decoder(task, **params), block, halo)
    assert same(out, ref), task
    return out, scene


@pytest.mark.parametrize("task", ["footprints", "fields", "roofs"])
def test_instance_tasks_under_manypred(task):
    from test_vector_blocks import _terraces

    f = _terraces(3)  # mixed 2-D / 3-D fields; ManyPred hands them all as (C, h, w) float32
    out, scene = _check(task, f, 150, 48, lambda a, b: _canon(a) == _canon(b))
    assert len(out) > 10
    fc = records_to_geojson(out, scene.transform, scene.crs)
    x0 = fc["features"][0]["geometry"]["coordinates"][0][0][0]
    assert 400000 <= x0 <= 400000 + 0.5 * scene.width  # world coordinates via the Affine


def test_roads_under_manypred():
    from test_vector_road_blocks import _same, _scene

    f = _scene(0)
    out, _ = _check("roads", f, 128, 40, _same, threshold=0.5, gap=50, keypoint_threshold=0.3)
    assert len(out.edges) > 5


def test_points_and_density_under_manypred():
    from test_points_blocks import scene_heatmap

    hm = scene_heatmap()  # (1, H, W)
    out, scene = _check("points", {"heatmap": hm[0]}, 24, 2,
                        lambda a, b: np.array_equal(a.pixels, b.pixels)
                        and np.array_equal(a.scores, b.scores), threshold=0.3, kernel=5)
    assert len(out.pixels) > 20 and out.transform == scene.transform and out.crs == scene.crs
    d = np.abs(hm).astype(np.float32)
    cnt, _ = _check("density", {"density": d}, 24, 0,
                    lambda a, b: a.total == pytest.approx(b.total), value_scale=2.0)
    assert cnt.total == pytest.approx(float(d.sum(dtype=np.float64)) / 2.0, rel=1e-6)
