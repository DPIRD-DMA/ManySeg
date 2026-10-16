"""Change detection in the bench: binary change task (change-class F1), semantic change task
(SECOND metrics: mIoU of change/no-change, SeK, Score), multi-channel pixel labels in the
loader."""

import math

import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader
from manyseg_bench.tasks import build_task


def test_binary_change_metrics():
    t = build_task("change", 2)
    assert t.primary_metric == "f1_change"
    y = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 1, 0, 1] = logits[0, 1, 1, 1] = 5.0  # predict change at (0,1) and (1,1)
    ev = t.evaluator()
    ev.update(logits, y)
    m = ev.compute()
    # tp=1 (0,1), fp=1 (1,1), fn=1 (1,0)
    assert m["precision_change"] == pytest.approx(0.5) and m["recall_change"] == 0.5
    assert m["f1_change"] == pytest.approx(0.5) and m["iou_change"] == pytest.approx(1 / 3)


def scd_ref(gt1, gt2, p1, p2, k):
    """Reference SECOND metrics from the definitions (Yang et al. 2021)."""
    gt, pr = np.concatenate([gt1.ravel(), gt2.ravel()]), np.concatenate([p1.ravel(), p2.ravel()])
    cm = np.zeros((k + 1, k + 1))
    np.add.at(cm, (gt, pr), 1)
    iou_nc = cm[0, 0] / (cm[0].sum() + cm[:, 0].sum() - cm[0, 0])
    iou_c = cm[1:, 1:].sum() / (cm.sum() - cm[0, 0])
    q = cm.copy()
    q[0, 0] = 0
    po = np.trace(q) / q.sum()
    pe = (q.sum(0) * q.sum(1)).sum() / q.sum() ** 2
    sek = (po - pe) / (1 - pe) * math.exp(iou_c - 1)
    miou = (iou_nc + iou_c) / 2
    return miou, sek, 0.3 * miou + 0.7 * sek


def test_semantic_change_metrics_match_reference():
    k = 3
    t = build_task("semantic_change", k)
    rng = np.random.default_rng(0)
    gt_change = rng.random((2, 8, 8)) < 0.4
    gt1 = np.where(gt_change, rng.integers(1, k + 1, (2, 8, 8)), 0)
    gt2 = np.where(gt_change, rng.integers(1, k + 1, (2, 8, 8)), 0)
    y = torch.from_numpy(np.stack([gt1, gt2, gt_change.astype(int)], 1)).long()
    out = {"change": torch.randn(2, 2, 8, 8), "lc1": torch.randn(2, k, 8, 8),
           "lc2": torch.randn(2, k, 8, 8)}
    ev = t.evaluator()
    ev.update(out, y)
    m = ev.compute()
    ch = out["change"].argmax(1).numpy() == 1
    p1 = np.where(ch, out["lc1"].argmax(1).numpy() + 1, 0)
    p2 = np.where(ch, out["lc2"].argmax(1).numpy() + 1, 0)
    miou, sek, score = scd_ref(gt1, gt2, p1, p2, k)
    assert m["miou_scd"] == pytest.approx(miou) and m["sek"] == pytest.approx(sek)
    assert m["score"] == pytest.approx(score) and t.primary_metric == "score"
    loss = t.loss(out, t.targets(y))
    assert torch.isfinite(loss) and loss.ndim == 0


def test_semantic_change_heads_per_input_or_fused():
    per = build_task("semantic_change", 6, per_input=True, share=True).heads()
    assert per["lc1"]["input"] == "t1" and per["lc2"] == {**per["lc1"], "input": "t2",
                                                          "share": "lc1"}
    fused = build_task("semantic_change", 6, per_input=False).heads()
    assert "input" not in fused["lc1"] and "share" not in fused["lc2"]


def test_loader_multichannel_pixel_labels_follow_augmentation():
    imgs = {"t1": np.zeros((2, 32, 32, 3), np.float16), "t2": np.zeros((2, 32, 32, 3),
                                                                        np.float16)}
    imgs["t1"][:, 4:12, 20:30] = 1
    y = np.zeros((2, 3, 32, 32), np.uint8)
    y[:, :, 4:12, 20:30] = [[[1]], [[2]], [[1]]]
    ld = GPULoader(imgs, y, 2, {"t1": [0.0] * 3, "t2": [0.0] * 3}, {"t1": [1.0] * 3,
                                                                   "t2": [1.0] * 3},
                   torch.device("cpu"), train=True, stack=False, out_size=(24, 24), seed=1)
    (x, yy), = list(ld)
    assert yy.shape == (2, 3, 24, 24) and yy.dtype == torch.long
    marker = x["t1"][:, 0] > 0.5
    assert ((yy[:, 0] == 1) == marker).float().mean() > 0.97
    assert torch.equal(yy[:, 1] == 2, yy[:, 0] == 1)


def test_multilabel_segmentation_task():
    """Overlapping classes per pixel (e.g. cloud and shadow): (B, K, H, W) multi-hot labels,
    255 = ignored; sigmoid + BCE; per-class IoU / F1 at a threshold."""
    t = build_task("multilabel_seg", 2, threshold=0.5)
    assert t.primary_metric == "miou"
    y = torch.tensor([[[[1, 1], [0, 255]], [[1, 0], [0, 0]]]])  # class0, class1 on a 2x2 tile
    logits = torch.full((1, 2, 2, 2), -5.0)
    logits[0, 0, 0, :] = 5.0           # class 0 predicted on the top row: matches its labels
    logits[0, 1, 0, 0] = logits[0, 1, 1, 1] = 5.0  # class 1: one hit, one false positive
    ev = t.evaluator()
    ev.update(logits, y)
    m = ev.compute()
    assert m["iou"] == pytest.approx([1.0, 0.5]) and m["miou"] == pytest.approx(0.75)
    assert m["f1"][1] == pytest.approx(2 / 3)
    loss = t.loss(logits, t.targets(y))
    assert torch.isfinite(loss)
    y_all_ignored = torch.full_like(y, 255)
    assert float(t.loss(logits, t.targets(y_all_ignored))) == 0.0


def test_multilabel_seg_loss_is_per_valid_pixel_and_empty_evaluator():
    t = build_task("multilabel_seg", 1)
    a = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    b = torch.full((1, 1, 2, 2), 255, dtype=torch.long)
    b[0, 0, 0, 0] = 1                                   # one valid pixel
    la, lb = torch.full((1, 1, 2, 2), 2.0), torch.full((1, 1, 2, 2), -2.0)
    ev = t.evaluator()
    ev.update(la, a)
    ev.update(lb, b)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    ref = (bce(la, a.float(), reduction="sum") + bce(lb[0, 0, 0, 0], torch.tensor(1.0))) / 5
    assert ev.compute()["loss"] == pytest.approx(float(ref))
    assert t.evaluator().compute()["miou"] == 0.0
