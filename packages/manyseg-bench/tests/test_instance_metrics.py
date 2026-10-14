import math

import pytest
import torch
from manyseg_bench.evaluation.objects import (
    CountMetrics,
    ObjectMetrics,
    PointMetrics,
    match_points,
    panoptic_quality,
)


def lab(*rects, shape=(20, 20)):
    m = torch.zeros(shape, dtype=torch.long)
    for k, (y0, y1, x0, x1) in enumerate(rects, 1):
        m[y0:y1, x0:x1] = k
    return m


def test_perfect_prediction():
    gt = lab((0, 5, 0, 5), (10, 15, 10, 15))
    om = ObjectMetrics()
    om.update(gt.clone(), gt)
    r = om.compute()
    assert r["precision"] == r["recall"] == r["f1"] == 1.0
    assert r["pq"] == pytest.approx(1.0) and r["over_seg"] == 0 and r["under_seg"] == 0


def test_split_object_is_over_segmented():
    gt = lab((0, 10, 0, 10))
    pred = lab((0, 10, 0, 5), (0, 10, 5, 10))
    om = ObjectMetrics()
    om.update(pred, gt)
    r = om.compute()
    assert r["tp"] == 0 and r["fp"] == 2 and r["fn"] == 1   # IoU 0.5 is not > 0.5
    assert r["over_seg"] == pytest.approx(0.5)              # best partner covers half the GT
    assert r["under_seg"] == pytest.approx(0.0)


def test_merged_objects_are_under_segmented():
    gt = lab((0, 10, 0, 5), (0, 10, 5, 10))
    pred = lab((0, 10, 0, 10))
    om = ObjectMetrics()
    om.update(pred, gt)
    assert om.compute()["under_seg"] == pytest.approx(0.5)


def test_pq_hand_computed():
    gt = lab((0, 10, 0, 10), (12, 16, 12, 16))
    pred = lab((0, 10, 0, 8))                    # IoU 0.8 with gt1; gt2 missed
    r = panoptic_quality(pred, gt)
    # SQ = 0.8, RQ = 1 / (1 + 0.5*0 + 0.5*1) = 2/3
    assert r["sq"] == pytest.approx(0.8) and r["rq"] == pytest.approx(2 / 3)
    assert r["pq"] == pytest.approx(0.8 * 2 / 3)


def test_metrics_accumulate_over_images():
    gt = lab((0, 5, 0, 5))
    om = ObjectMetrics()
    om.update(gt, gt)
    om.update(torch.zeros_like(gt), gt)
    r = om.compute()
    assert r["tp"] == 1 and r["fn"] == 1 and r["recall"] == 0.5


def test_point_matching_within_radius():
    gt = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    pred = torch.tensor([[1.0, 1.0], [10.0, 13.5], [50.0, 50.0]])
    tp, fp, fn = match_points(pred, gt, radius=3.0)
    assert (tp, fp, fn) == (1, 2, 2)
    pm = PointMetrics(radius=4.0)
    pm.update(pred, gt)
    r = pm.compute()
    assert r["tp"] == 2 and r["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_point_matching_is_one_to_one():
    gt = torch.tensor([[0.0, 0.0]])
    pred = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert match_points(pred, gt, radius=2) == (1, 1, 0)


def test_count_metrics():
    cm = CountMetrics()
    cm.update(torch.tensor([10.0, 4.0]), torch.tensor([8.0, 4.0]))
    r = cm.compute()
    assert r["mae"] == pytest.approx(1.0) and r["rmse"] == pytest.approx(math.sqrt(2))
