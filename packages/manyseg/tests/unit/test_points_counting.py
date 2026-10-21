"""Counting helpers: density loss and counts, peak smoothing / masking, points from boxes,
threshold calibration."""

import math

import pytest
import torch
from manyseg import create_model
from manyseg.features import FeatureSpec
from manyseg.heads import PixelHead
from manyseg.points import (
    calibrate_threshold,
    count_from_density,
    density_head,
    density_loss,
    density_map,
    find_peaks,
    gaussian_heatmap,
    points_from_boxes,
)

# --- density head and loss ----------------------------------------------------------------


def test_density_head_activation():
    assert "activation" not in density_head()["density"]  # default unchanged (E011 setup)
    assert density_head(activation="softplus")["density"]["activation"] == "softplus"
    m = create_model(backbone="resnet18", pretrained=False,
                     heads=density_head("unet", activation="softplus"))
    assert m(torch.randn(1, 3, 37, 45))["density"].min() >= 0


def test_density_loss_target_integrates_to_count():
    pts = [torch.tensor([[10.0, 12.0], [30.0, 31.0], [20.0, 5.5]]), torch.zeros(0, 2)]
    pred = torch.zeros(2, 1, 40, 44, requires_grad=True)
    loss = density_loss(pred, pts, sigma=3.0)
    s = 2 * math.pi * 9
    # pred = 0: the loss is the target's mean square, target = density_map * 2 pi sigma^2
    tgt = density_map(pts, (40, 44), 3.0) * s
    torch.testing.assert_close(loss, (tgt ** 2).mean())
    assert abs(tgt[0].sum() / s - 3) < 1e-4 and tgt[1].sum() == 0
    # an isolated point on a pixel centre has a peak of about 1
    assert abs(tgt[0, 0, 10, 12] - 1) < 0.01
    loss.backward()
    assert pred.grad is not None
    # the count term: |0 - 3| / 2 images on average
    both = density_loss(pred.detach(), pts, sigma=3.0, count_weight=1.0)
    torch.testing.assert_close(both - loss.detach(), torch.tensor(1.5), atol=1e-4, rtol=0)


def test_density_loss_mask_and_errors():
    pts = [torch.tensor([[5.0, 5.0]])]
    pred = torch.full((1, 1, 20, 20), 0.5)
    mask = torch.zeros(1, 20, 20, dtype=torch.bool)
    mask[:, :10, :10] = True
    full = density_loss(pred, pts, sigma=2.0, kind="l1")
    part = density_loss(pred, pts, sigma=2.0, kind="l1", mask=mask)
    assert part != full
    with pytest.raises(ValueError, match="give value_scale"):
        density_loss(pred, pts, sigma=[torch.tensor([2.0])])
    density_loss(pred, pts, sigma=[torch.tensor([2.0])], value_scale=25.0)
    with pytest.raises(ValueError, match="point lists for a batch"):
        density_loss(pred, pts * 2)


def test_count_from_density_scale_and_mask():
    d = torch.ones(2, 1, 4, 5)
    assert count_from_density(d).tolist() == [20.0, 20.0]
    assert count_from_density(d, value_scale=4.0).tolist() == [5.0, 5.0]
    m = torch.zeros(2, 4, 5)
    m[0, :2] = 1
    assert count_from_density(d, mask=m).tolist() == [10.0, 0.0]


def blob_images(n, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs, pts = [], []
    for _ in range(n):
        k = int(torch.randint(1, 6, (1,), generator=g))
        p = torch.randint(4, size - 4, (k, 2), generator=g).float()
        imgs.append(gaussian_heatmap([p], (size, size), sigma=1.5)[0].expand(3, -1, -1)
                    + 0.05 * torch.rand(3, size, size, generator=g))
        pts.append(p)
    return torch.stack(imgs), pts


def test_trained_toy_density_counts():
    torch.manual_seed(0)
    x, pts = blob_images(16)
    net = PixelHead(FeatureSpec(3, 1), 1, hidden=[16, 16], dilation=[1, 2],
                    activation="softplus", bias_init=-4.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(150):
        loss = density_loss(net(x), pts, sigma=2.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    xv, pv = blob_images(8, seed=1)
    with torch.no_grad():
        pred = net(xv)
    assert pred.min() >= 0
    counts = count_from_density(pred, value_scale=2 * math.pi * 4)
    truth = torch.tensor([float(len(p)) for p in pv])
    assert (counts - truth).abs().mean() < 0.6, (counts, truth)


# --- find_peaks: smoothing and masks ------------------------------------------------------

def test_find_peaks_defaults_unchanged():
    g = torch.Generator().manual_seed(0)
    hm = torch.rand(3, 1, 30, 40, generator=g)
    got = find_peaks(hm, 0.5, kernel=5, max_points=7)
    # the pre-change implementation, inline
    pooled = torch.nn.functional.max_pool2d(hm[:, 0][:, None], 5, 1, 2)[:, 0]
    keep = (hm[:, 0] == pooled) & (hm[:, 0] > 0.5)
    for i, (p, s) in enumerate(got):
        ref_s = hm[i, 0][keep[i]]
        order = torch.argsort(ref_s, descending=True)[:7]
        assert torch.equal(p, torch.nonzero(keep[i])[order])
        assert torch.equal(s, ref_s[order])
        assert s.dtype == hm.dtype


def noisy_heatmap(seed=0):
    pts = torch.tensor([[12.0, 14.0], [40.0, 45.0], [20.0, 50.0], [50.0, 10.0]])
    g = torch.Generator().manual_seed(seed)
    hm = gaussian_heatmap([pts], (64, 64), sigma=5.0) + 0.25 * torch.rand(1, 1, 64, 64,
                                                                         generator=g)
    return hm, pts


def test_find_peaks_smoothing():
    hm, pts = noisy_heatmap()
    raw = find_peaks(hm, 0.6, kernel=5)[0][0]
    smooth = find_peaks(hm, 0.6, kernel=5, smooth_sigma=3)[0][0]
    assert len(raw) > len(pts)
    assert len(smooth) == len(pts)
    d = torch.cdist(smooth.float(), pts).min(1).values
    assert d.max() <= 2
    bf = find_peaks(hm.bfloat16(), 0.6, kernel=5, smooth_sigma=3)[0][0]  # upcast internally
    assert len(bf) == len(pts)
    with pytest.raises(ValueError, match="kernel must be odd"):
        find_peaks(hm, kernel=4, smooth_sigma=3)
    with pytest.raises(ValueError, match="smooth_sigma"):
        find_peaks(hm, smooth_sigma=-1)


def test_find_peaks_tiny_image_smoothing():
    hm = torch.zeros(1, 3, 3)
    hm[0, 1, 1] = 1
    (p, _), = find_peaks(hm, 0.01, smooth_sigma=3)  # radius 9 > image: replicate padding
    assert p.tolist() == [[1, 1]]


@pytest.mark.parametrize("as_float", [False, True])
def test_find_peaks_mask(as_float):
    hm, pts = noisy_heatmap()
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    mask[..., :32, :32] = True  # only the first point's quadrant
    mask = mask.float() if as_float else mask
    p, _ = find_peaks(hm, 0.6, kernel=5, smooth_sigma=3, mask=mask)[0]
    assert len(p) == 1 and (p < 32).all()
    p, _ = find_peaks(hm, 0.6, kernel=5, smooth_sigma=3, mask=mask[:, 0])[0]
    assert len(p) == 1
    with pytest.raises(ValueError, match="does not match"):
        find_peaks(hm, mask=mask[..., :10])


# --- points from boxes --------------------------------------------------------------------

def test_points_from_boxes_formats():
    h, w = 40, 60
    # a box over pixels x 10..19, y 4..9 (edges 10..20, 4..10): centre pixel (y, x) = (6.5, 14.5)
    xyxy = torch.tensor([[10.0, 4.0, 20.0, 10.0], [0.0, 0.0, 3.0, 3.0]])
    xywh = torch.tensor([[10.0, 4.0, 10.0, 6.0], [0.0, 0.0, 3.0, 3.0]])
    yolo = torch.tensor([[15.0 / w, 7.0 / h, 10.0 / w, 6.0 / h],
                         [1.5 / w, 1.5 / h, 3.0 / w, 3.0 / h]])
    want = torch.tensor([[6.5, 14.5], [1.0, 1.0]])
    for fmt, b in (("xyxy", xyxy), ("xywh", xywh), ("yolo", yolo)):
        (p,) = points_from_boxes([b], (h, w), fmt=fmt)
        torch.testing.assert_close(p, want)
        hm = gaussian_heatmap([torch.tensor([[1.0, 1.0]])], (h, w), 2.0)
        assert hm[0, 0].argmax() == 1 * w + 1  # y first: row 1, column 1


def test_points_from_boxes_axis_order_and_sigma():
    # a wide box off the diagonal: x = 50, y = 10, so (y, x) must not be swapped
    b = torch.tensor([[45.0, 8.0, 55.0, 12.0]])
    pts, sig = points_from_boxes([b, torch.zeros(0, 4)], fmt="xyxy", sigma_from_size=0.5)
    torch.testing.assert_close(pts[0], torch.tensor([[9.5, 49.5]]))
    torch.testing.assert_close(sig[0], torch.tensor([0.5 * math.sqrt(40.0)]))
    assert pts[1].shape == (0, 2) and sig[1].shape == (0,)
    hm = gaussian_heatmap([torch.floor(p + 0.5) for p in pts], (20, 64), sig)
    assert divmod(int(hm[0, 0].argmax()), 64) == (10, 50)
    assert hm[1].max() == 0
    d = density_map(pts, (20, 64), sig)
    assert abs(d[0].sum() - 1) < 1e-4


def test_points_from_boxes_errors():
    with pytest.raises(ValueError, match="size"):
        points_from_boxes([torch.zeros(1, 4)])
    with pytest.raises(ValueError, match="YOLO class column"):
        points_from_boxes([torch.zeros(1, 5)], (10, 10))
    with pytest.raises(ValueError, match="negative width"):
        points_from_boxes([torch.tensor([[5.0, 5.0, 1.0, 1.0]])], fmt="xyxy")
    with pytest.raises(ValueError, match="fmt"):
        points_from_boxes([torch.zeros(1, 4)], fmt="cxcywh")


def test_per_image_sigma_lists():
    pts = [torch.tensor([[5.0, 5.0]]), torch.tensor([[5.0, 5.0], [15.0, 15.0]])]
    hm = gaussian_heatmap(pts, (20, 20), [torch.tensor([1.0]), torch.tensor([1.0, 3.0])])
    assert hm[1, 0, 15, 16] > hm[0, 0, 5, 6]  # wider sigma on image 1's second point
    with pytest.raises(ValueError, match="per-image sigma"):
        gaussian_heatmap(pts, (20, 20), [torch.tensor([1.0])])
    with pytest.raises(ValueError, match="values for 2 points"):
        gaussian_heatmap(pts, (20, 20), torch.tensor([1.0]))


# --- threshold calibration ----------------------------------------------------------------

def calib_data():
    """Each image has true peaks of 0.6 and spurious ones of 0.3 (and one image has no
    objects): any threshold in [0.3, 0.6) keeps exactly the true ones."""
    imgs, gts = [], []
    g = torch.Generator().manual_seed(0)
    for i in range(6):
        k = i % 3  # 0, 1 or 2 objects
        pts = torch.randint(4, 60, (k, 2), generator=g).float()
        spur = torch.randint(4, 60, (2, 2), generator=g).float()
        hm = 0.6 * gaussian_heatmap([torch.floor(pts + 0.5)], (64, 64), 1.0) if k else \
            torch.zeros(1, 1, 64, 64)
        hm = torch.maximum(hm, 0.3 * gaussian_heatmap([spur], (64, 64), 1.0))
        imgs.append(hm)
        gts.append(torch.floor(pts + 0.5))
    return torch.cat(imgs), gts


@pytest.mark.parametrize("metric", ["count_mae", "f1"])
def test_calibrate_threshold_recovers_best(metric):
    hm, gts = calib_data()
    best, curve = calibrate_threshold(hm, gts, thresholds=[0.1, 0.2, 0.4, 0.5, 0.7],
                                      metric=metric, kernel=3)
    assert best == 0.4  # ties go to the lower threshold
    perfect = 0.0 if metric == "count_mae" else 1.0
    assert curve[0.4] == curve[0.5] == perfect
    assert curve[0.2] != perfect and curve[0.7] != perfect


def test_calibrate_threshold_counts_and_batches():
    hm, gts = calib_data()
    counts = torch.tensor([float(len(g)) for g in gts])
    best, _ = calibrate_threshold([hm[:3], hm[3:]], counts, thresholds=[0.2, 0.4])
    assert best == 0.4
    best, _ = calibrate_threshold(hm, [len(g) for g in gts], thresholds=[0.2, 0.4])
    assert best == 0.4
    with pytest.raises(ValueError, match="needs ground-truth points"):
        calibrate_threshold(hm, counts, metric="f1")
    with pytest.raises(ValueError, match="targets for"):
        calibrate_threshold(hm, counts[:2])
    with pytest.raises(ValueError, match="'thresholds'"):
        calibrate_threshold(hm, counts, threshold=0.3)


def test_calibrate_threshold_no_objects():
    hm = torch.zeros(2, 1, 16, 16)
    best, curve = calibrate_threshold(hm, [torch.zeros(0, 2)] * 2, metric="f1")
    assert all(v == 1.0 for v in curve.values())
    best, curve = calibrate_threshold(hm, torch.zeros(2))
    assert all(v == 0.0 for v in curve.values())


# --- regressions from review ------------------------------------------------------------

def test_f1_matching_is_maximum_not_greedy():
    from manyseg.points.postprocess import _match_points

    # greedy takes the closest pair (0 -> 0.5) and finds 1 match; the maximum matching is 2
    pred = torch.tensor([[0.0, 0.0], [0.0, 1.4]])
    gt = torch.tensor([[0.0, 0.5], [0.0, -0.9]])
    assert _match_points(pred, gt, 1.0) == 2
    assert _match_points(pred, gt[:0], 1.0) == 0


def test_zero_size_boxes_get_min_sigma():
    pts, sig = points_from_boxes([torch.tensor([[5.0, 5.0, 5.0, 9.0]])], fmt="xyxy",
                                 sigma_from_size=0.1)
    assert sig[0].tolist() == [1.0]
    assert torch.isfinite(gaussian_heatmap(pts, (16, 16), sig)).all()
    with pytest.raises(ValueError, match="sigma must be > 0"):
        gaussian_heatmap(pts, (16, 16), [torch.tensor([0.0])])


def test_density_masks_must_match_batch():
    d = torch.ones(2, 1, 4, 5)
    with pytest.raises(ValueError, match="does not match"):
        count_from_density(d, mask=torch.ones(1, 4, 5))
    with pytest.raises(ValueError, match="does not match"):
        density_loss(d, [torch.zeros(0, 2)] * 2, mask=torch.ones(1, 1, 4, 5))


def test_calibrate_threshold_matches_find_peaks_at_the_threshold():
    # Gaussian targets peak at exactly the threshold: find_peaks keeps score > thr, so
    # calibration must too
    hm = 0.5 * gaussian_heatmap([torch.tensor([[8.0, 8.0], [20.0, 20.0]])], (32, 32), 2.0)
    _, curve = calibrate_threshold(hm, torch.tensor([2.0]), thresholds=[0.4, 0.5])
    assert len(find_peaks(hm, 0.5)[0][0]) == 0
    assert curve[0.5] == 2.0 and curve[0.4] == 0.0


@pytest.mark.parametrize("metric", ["count_mae", "f1"])
def test_calibrated_count_equals_find_peaks_at_best(metric):
    # peaks of exactly 0.5 and 0.8, with 0.5 a candidate threshold
    pts = torch.tensor([[8.0, 8.0], [24.0, 24.0]])
    hm = torch.maximum(0.8 * gaussian_heatmap([pts[:1]], (32, 32), 2.0),
                       0.5 * gaussian_heatmap([pts[1:]], (32, 32), 2.0))
    best, curve = calibrate_threshold(hm, [pts[:1]], thresholds=[0.5, 0.6], metric=metric)
    # at 0.5 only the 0.8 peak survives in both, so 0.5 and 0.6 tie and 0.5 wins
    assert best == 0.5 and curve[0.5] == curve[0.6]
    n = len(find_peaks(hm, threshold=best)[0][0])
    assert n == 1
    if metric == "count_mae":
        assert curve[best] == abs(n - 1)
