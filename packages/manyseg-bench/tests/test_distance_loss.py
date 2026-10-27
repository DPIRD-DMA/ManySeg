"""distance_weighted_bce and distance_map (review fixes: explicit geometry, pixel-mean scale,
precomputed linear-time distances)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import ndimage

from manyseg_bench.losses import distance_map, distance_weighted_bce

H = W = 64


def blob(cx, cy, sigma=1.5):
    yy, xx = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))[None]


def logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def point_distance(cx, cy):
    g = np.zeros((H, W), bool)
    g[int(cy), int(cx)] = True  # the exact vertex pixel, not a threshold of the blob
    return torch.from_numpy(distance_map(g))[None]


def test_distance_map_matches_scipy_and_caps():
    rng = np.random.default_rng(0)
    g = rng.random((40, 50)) < 0.01
    np.testing.assert_allclose(distance_map(g, 1e9), ndimage.distance_transform_edt(~g), rtol=1e-6)
    assert distance_map(np.zeros((8, 8), bool), 7.0).max() == 7.0
    assert distance_map(g, 3.0).max() <= 3.0


def test_far_miss_costs_more_than_near_miss_points():
    target, d = blob(32.0, 32.0), point_distance(32.0, 32.0)
    exact, near, mid, far = (distance_weighted_bce(logit(blob(c, 32.0)), target, d) for c in (32.0, 36.0, 40.0, 48.0))
    assert exact < near < mid < far
    # plain BCE cannot tell 8 px from 16 px apart; this loss can, more so with a larger far_weight
    b_mid = F.binary_cross_entropy_with_logits(logit(blob(40.0, 32.0)), target)
    b_far = F.binary_cross_entropy_with_logits(logit(blob(48.0, 32.0)), target)
    assert abs(b_mid - b_far) < 0.05 * b_far
    gaps = [(distance_weighted_bce(logit(blob(48.0, 32.0)), target, d, far_weight=w)
             - distance_weighted_bce(logit(blob(40.0, 32.0)), target, d, far_weight=w)).item() for w in (0.0, 1.0, 4.0)]
    assert gaps[0] < gaps[1] < gaps[2] and gaps[1] > 0


def test_lines():
    g = np.zeros((H, W), bool); g[30, 5:60] = True
    target, d = torch.from_numpy(g[None].astype(np.float32)), torch.from_numpy(distance_map(g))[None]
    near = torch.zeros(1, H, W); near[:, 33, 5:60] = 1
    far = torch.zeros(1, H, W); far[:, 50, 5:60] = 1
    lo = lambda m: logit(m * 0.98 + 0.01)
    assert distance_weighted_bce(lo(near), target, d) < distance_weighted_bce(lo(far), target, d)


def test_scale_independent_of_image_size_and_empty_images():
    """Pixel means: an empty image costs about plain BCE, not thousands (review issue 1)."""
    z = torch.zeros(1, 512, 512)  # p = 0.5 everywhere
    empty = distance_weighted_bce(z, torch.zeros(1, 512, 512), torch.full((1, 512, 512), 32.0))
    assert abs(empty.item() - (math.log(2) + 0.5)) < 1e-4  # BCE at p = 0.5 + capped penalty, far_weight 1
    small = distance_weighted_bce(torch.zeros(1, 64, 64), torch.zeros(1, 64, 64), torch.full((1, 64, 64), 32.0))
    assert abs(empty.item() - small.item()) < 1e-5


def test_far_pixels_pushed_down_harder_and_soft_optimum():
    target, d = blob(32.0, 32.0), point_distance(32.0, 32.0)
    pred = torch.full((1, H, W), -6.0)
    pred[0, 32, 35] = pred[0, 32, 55] = 2.0
    pred.requires_grad_()
    distance_weighted_bce(pred, target, d).backward()
    assert pred.grad[0, 32, 55] > pred.grad[0, 32, 35] > 0
    # free logits converge to the soft target near the point (no flat-topped peak)
    z = torch.zeros(1, H, W, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.1)
    for _ in range(400):
        opt.zero_grad(); distance_weighted_bce(z, target, d).backward(); opt.step()
    p = torch.sigmoid(z)[0]
    assert abs(p[32, 32].item() - target[0, 32, 32].item()) < 0.05
    assert abs(p[32, 34].item() - target[0, 32, 34].item()) < 0.05


def test_validation():
    t = torch.zeros(1, H, W)
    with pytest.raises(ValueError):
        distance_weighted_bce(t, t, t, tolerance=0)
    with pytest.raises(ValueError):
        distance_weighted_bce(t, t, torch.zeros(1, 1, H + 1, W))
    out = distance_weighted_bce(t[:, None], t[:, None], torch.full((1, 1, H, W), 3.0))  # (B, 1, H, W) accepted
    assert math.isfinite(out.item())
