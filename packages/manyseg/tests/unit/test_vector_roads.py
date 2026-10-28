"""Unified road heads decoder (E039 R4 / E044b): crossings stay unconnected, junctions connect."""

import math

import numpy as np
import pytest

pytest.importorskip("skimage")
from manyseg.vectors import graph_from_road_heads  # noqa: E402


def _heads(kind: str, H=200, W=200, width=6):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    mask = ((np.abs(yy - 100) <= width / 2) | (np.abs(xx - 100) <= width / 2)).astype(np.float32)
    kp = np.zeros((3, H, W), np.float32)
    ch = {"junction": 0, "crossing": 2}[kind]
    kp[ch] = np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / 8)
    d = np.zeros((4, H, W), np.float32)
    horiz = np.abs(yy - 100) <= width / 2
    vert = np.abs(xx - 100) <= width / 2
    d[0][horiz], d[1][horiz] = math.cos(0), math.sin(0)  # 2t = 0
    d[0][vert & ~horiz], d[1][vert & ~horiz] = math.cos(math.pi), math.sin(math.pi)  # 2t = pi
    both = horiz & vert
    d[2][both], d[3][both] = math.cos(math.pi), math.sin(math.pi)
    return mask, kp, d


def _degrees(vg):
    return np.bincount(vg.edges.reshape(-1), minlength=len(vg.xy))


def test_crossing_keeps_two_unconnected_roads():
    vg = graph_from_road_heads(*_heads("crossing"), gap=0)
    g = vg.to_networkx()
    import networkx as nx

    assert nx.number_connected_components(g) == 2
    assert _degrees(vg).max() <= 2


def test_junction_connects_four_arms():
    vg = graph_from_road_heads(*_heads("junction"), gap=0)
    deg = _degrees(vg)
    assert deg.max() == 4
    centre = vg.xy[deg.argmax()]
    assert np.hypot(*(centre - 100)) < 2


def test_no_keypoints_is_the_surface_skeleton():
    from manyseg.vectors import graph_from_mask

    m, kp, d = _heads("junction")
    kp[:] = 0
    a = graph_from_road_heads(m, kp, d, gap=0)
    b = graph_from_mask(m, 0.3, spur=30, gap=0, simplify=2.0, min_length=20)
    assert len(a.edges) == len(b.edges) and np.allclose(np.sort(a.xy, 0), np.sort(b.xy, 0))
