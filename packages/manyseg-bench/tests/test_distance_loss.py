"""distance_aware_bce: near misses cost less than far ones, for points and lines; gradients point
towards the truth; plain BCE cannot tell them apart."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("manyseg")

from manyseg_bench.losses import distance_aware_bce  # noqa: E402

H = W = 64


def blob(cx, cy, sigma=1.5):
    yy, xx = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))[None]


def logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def test_point_near_miss_costs_less_than_far_miss():
    target = blob(32.0, 32.0)
    near, far = logit(blob(36.0, 32.0)), logit(blob(48.0, 32.0))  # 4 px and 16 px off
    assert distance_aware_bce(near, target) < distance_aware_bce(far, target)
    # plain BCE: once both are clear of the target blob (8 vs 16 px) it cannot tell them apart
    mid = logit(blob(40.0, 32.0))
    b_mid = F.binary_cross_entropy_with_logits(mid, target)
    b_far = F.binary_cross_entropy_with_logits(far, target)
    assert abs(b_mid - b_far) < 0.05 * b_far
    assert distance_aware_bce(mid, target) < 0.95 * distance_aware_bce(far, target)
    exact = distance_aware_bce(logit(target), target)
    assert exact < distance_aware_bce(near, target)


def test_line_near_miss_costs_less():
    target = torch.zeros(1, H, W)
    target[:, 30, 5:60] = 1  # a horizontal line (e.g. a boundary)
    near = torch.zeros_like(target); near[:, 33, 5:60] = 1
    far = torch.zeros_like(target); far[:, 50, 5:60] = 1
    lo = lambda m: logit(m * 0.98 + 0.01)
    assert distance_aware_bce(lo(near), target) < distance_aware_bce(lo(far), target)


def test_gradient_pushes_far_mass_down_more():
    target = blob(32.0, 32.0)
    pred = torch.full((1, H, W), -6.0)
    pred[0, 32, 35] = pred[0, 32, 55] = 2.0  # a near (3 px) and a far (23 px) false positive
    pred.requires_grad_()
    distance_aware_bce(pred, target).backward()
    g = pred.grad[0]
    assert g[32, 55] > g[32, 35] > 0  # both pushed down, the far one harder


def test_empty_geometry_and_shapes():
    target = torch.zeros(2, 1, H, W)
    out = distance_aware_bce(torch.zeros(2, 1, H, W), target)
    assert math.isfinite(out.item())
    d = torch.full((2, H, W), 5.0)
    assert math.isfinite(distance_aware_bce(torch.zeros(2, H, W), target[:, 0], distance=d).item())
