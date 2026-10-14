import math

import torch
from manyseg_bench.evaluation import ConfusionMatrix, boundary_f1
from manyseg_bench.reporting import pareto_front


def test_confusion_metrics_hand_computed():
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    pred = torch.tensor([0, 1, 1, 1, 2, 0])
    cm = ConfusionMatrix(3)
    cm.update(pred, target)
    m = cm.compute()
    # class0: tp1 fp1 fn1 -> 1/3 ; class1: tp2 fp1 fn0 -> 2/3 ; class2: tp1 fp0 fn1 -> 1/2
    assert math.isclose(m["iou"][0], 1 / 3)
    assert math.isclose(m["iou"][1], 2 / 3)
    assert math.isclose(m["iou"][2], 1 / 2)
    assert math.isclose(m["miou"], (1 / 3 + 2 / 3 + 1 / 2) / 3)
    assert math.isclose(m["pixel_acc"], 4 / 6)
    assert math.isclose(m["dice"][1], 2 * 2 / (2 + 3))


def test_ignore_index_excluded():
    cm = ConfusionMatrix(2, ignore_index=255)
    cm.update(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 255]))
    assert cm.compute()["miou"] == 1.0


def test_absent_class_not_averaged():
    cm = ConfusionMatrix(3)
    cm.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert cm.compute()["miou"] == 1.0


def test_accumulation_equals_single_update():
    torch.manual_seed(0)
    p, t = torch.randint(0, 4, (2, 50)), torch.randint(0, 4, (2, 50))
    a, b = ConfusionMatrix(4), ConfusionMatrix(4)
    a.update(p, t)
    b.update(p[0], t[0])
    b.update(p[1], t[1])
    assert a.compute() == b.compute()


def test_boundary_f1_perfect_and_shifted():
    t = torch.zeros(1, 32, 32, dtype=torch.long)
    t[:, 8:24, 8:24] = 1
    assert boundary_f1(t, t).item() == 1.0
    shifted = torch.roll(t, 1, dims=2)
    assert boundary_f1(shifted, t, tolerance=2).item() == 1.0
    far = torch.roll(t, 6, dims=2)
    assert boundary_f1(far, t, tolerance=2).item() < 0.7


def test_pareto():
    pts = [(1, 0.5, "a"), (2, 0.6, "b"), (3, 0.55, "c"), (4, 0.7, "d")]
    assert pareto_front(pts) == ["a", "b", "d"]
