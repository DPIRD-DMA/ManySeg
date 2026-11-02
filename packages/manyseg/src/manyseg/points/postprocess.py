"""Post-processing for point/density heads (runs outside the exported model graph).

Scaled heads (head ``scale`` < 1): everything here works on the grid it is given. Pass
``scale=`` to :func:`find_peaks` to get points back in input-pixel coordinates (float, pixel
centres at integers; see :mod:`manyseg.utils.resolution`). :func:`group_by_offsets` works on the
head grid (centres from ``find_peaks`` without ``scale``, offsets in head pixels as produced by
targets made at head scale); map its labels to the input grid with
``manyseg.utils.upsample_output(labels, (H, W), scale, mode="nearest")``. Density maps are
sums, so :func:`count_from_density` is scale-independent if the density target was made with
``manyseg.utils.downsample_sum``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils._peaks import maxima_mask
from ..utils.resolution import scale_factor, to_input_coords


def blur_radius(sigma: float) -> int:
    """Radius in pixels of :func:`find_peaks`' ``smooth_sigma`` blur: ``max(1, ceil(3 sigma))``
    (0 without a blur)."""
    return max(1, math.ceil(3 * sigma)) if sigma > 0 else 0


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur of ``(B, H, W)`` (radius :func:`blur_radius`, replicate padding,
    float32 internally).

    A sum over the kernel taps in a fixed order (not ``conv2d``): every output pixel is the same
    sequence of float operations on its neighbourhood whatever the image size, so a crop gives
    bit-identical values away from its edges. Block decoding (:mod:`manyseg.points.blocks`)
    relies on this to equal a whole-scene :func:`find_peaks` exactly, ties included."""
    r = blur_radius(sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).tolist()
    y = x.float()
    h, w = y.shape[-2:]
    p = F.pad(y[:, None], (r, r, 0, 0), mode="replicate")[:, 0]
    y = sum((k[i] * p[..., i:i + w] for i in range(1, 2 * r + 1)), k[0] * p[..., 0:w])
    p = F.pad(y[:, None], (0, 0, r, r), mode="replicate")[:, 0]
    return sum((k[i] * p[:, i:i + h] for i in range(1, 2 * r + 1)), k[0] * p[:, 0:h])


def _squeeze(t: Tensor, name: str, like: Tensor | None = None) -> Tensor:
    """``(B, H, W)`` from ``(B, [1,] H, W)``; with ``like``, its batch and spatial size must
    match (no silent broadcasting of one mask over a batch)."""
    if t.dim() == 4:
        if t.shape[1] != 1:
            raise ValueError(f"{name} must be (B, H, W) or (B, 1, H, W), got {tuple(t.shape)}")
        t = t[:, 0]
    elif t.dim() != 3:
        raise ValueError(f"{name} must be (B, H, W) or (B, 1, H, W), got {tuple(t.shape)}")
    if like is not None and (t.shape[0], *t.shape[-2:]) != (like.shape[0], *like.shape[-2:]):
        raise ValueError(f"{name} {tuple(t.shape)} does not match the map {tuple(like.shape)}")
    return t


@torch.no_grad()
def find_peaks(heatmap: Tensor, threshold: float = 0.5, *, kernel: int = 3,
               max_points: int | None = None, scale: float = 1, smooth_sigma: float = 0,
               mask: Tensor | None = None) -> list[tuple[Tensor, Tensor]]:
    """Local maxima via max-pool NMS. Returns per image ``(points (K, 2) (y, x), scores)``,
    sorted by score descending. Pass probabilities (apply ``sigmoid`` to logits first).

    Points are long heatmap-pixel indices, or, for a heatmap from a head with ``scale`` < 1,
    float input-pixel coordinates ``(p + 0.5) / scale - 0.5`` (the centre of the input area
    the output pixel covers). ``kernel`` is in heatmap pixels; for large objects, about their
    diameter.

    * ``smooth_sigma`` > 0: Gaussian-blur the heatmap first (separable, replicate padding, in
      float32), so a noisy blob gives one peak, not a cluster. Scores are then the smoothed
      values. On drone tiles with large plants, sigma 3 halved count error (MAPE 0.79 -> 0.41).
    * ``mask`` ``(B, H, W)`` or ``(B, 1, H, W)``, bool or float (> 0 is inside): no peaks
      outside it, e.g. ``sigmoid(out["fg"]) > 0.5`` from a foreground head, which removes
      background false positives. Applied after smoothing.
    """
    if kernel < 1 or kernel % 2 != 1:
        raise ValueError(f"kernel must be odd and positive, got {kernel}")
    if max_points is not None and max_points < 0:
        raise ValueError(f"max_points must be >= 0 or None, got {max_points}")
    if smooth_sigma < 0:
        raise ValueError(f"smooth_sigma must be >= 0, got {smooth_sigma}")
    scale_factor(scale)  # validate before any work
    hm = heatmap[:, 0] if heatmap.dim() == 4 else heatmap
    if smooth_sigma > 0:
        hm = _gaussian_blur(hm, smooth_sigma)
    if mask is not None:
        m = _squeeze(mask, "mask")
        if m.shape != hm.shape:
            raise ValueError(f"mask {tuple(m.shape)} does not match the heatmap "
                             f"{tuple(hm.shape)}")
        hm = hm.masked_fill(~(m.bool() if m.dtype == torch.bool else m > 0), float("-inf"))
    keep = maxima_mask(hm, threshold, kernel)
    out = []
    for i in range(hm.shape[0]):
        pts = torch.nonzero(keep[i])
        scores = hm[i][keep[i]]
        order = torch.argsort(scores, descending=True)
        if max_points is not None:
            order = order[:max_points]
        out.append((to_input_coords(pts[order], scale), scores[order]))
    return out


def count_from_density(density: Tensor, *, value_scale: float = 1.0,
                       mask: Tensor | None = None) -> Tensor:
    """Predicted count per image: the integral of the density map, divided by ``value_scale``
    (the factor the training target was multiplied by; see :func:`density_loss`), over
    ``mask`` (``(B, H, W)`` or ``(B, 1, H, W)``, > 0 inside) if given.

    The sum runs over every pixel, so a small background bias adds up: 0.005 per pixel over
    a 792 x 792 tile is 3 phantom objects. Check counts on whole validation images (not
    training crops), and prefer peak counting for a few large objects per image."""
    d = density.float()
    if mask is not None:
        m = _squeeze(mask, "mask", like=d).float()
        d = d * (m[:, None] if d.dim() == 4 else m)
    return d.flatten(1).sum(1) / value_scale


DEFAULT_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7)


def _match_points(pred: Tensor, gt: Tensor, radius: float) -> int:
    """Maximum one-to-one matching of ``(N, 2)`` point sets within ``radius`` (augmenting
    paths, Kuhn's algorithm): the number of matches (true positives)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    near = torch.cdist(pred.double().cpu(), gt.double().cpu()) <= radius
    cand = [torch.nonzero(row)[:, 0].tolist() for row in near]
    owner: dict[int, int] = {}  # gt index -> pred index

    def augment(i: int, seen: set) -> bool:
        for j in cand[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(cand)) if cand[i])


@torch.no_grad()
def calibrate_threshold(heatmaps: Tensor | Sequence[Tensor], targets: Tensor | Sequence,
                        thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                        metric: str = "count_mae", *, radius: float = 3.0,
                        **find_peaks_kwargs) -> tuple[float, dict[float, float]]:
    """Choose the :func:`find_peaks` threshold on validation data.

    ``heatmaps``: probabilities, a ``(B, [1,] H, W)`` tensor or a sequence of them (batches,
    sizes may differ). ``targets``: one per image, either counts (a ``(B,)`` tensor or ints;
    ``metric="count_mae"`` only) or ground-truth points ``(N_i, 2)`` ``(y, x)`` in the same
    coordinates :func:`find_peaks` returns. ``metric``: ``"count_mae"`` (lower is better) or
    ``"f1"`` (point F1, a maximum one-to-one matching within ``radius``; higher is better),
    pooled over all images. Other keywords go to :func:`find_peaks` (``kernel``, ``smooth_sigma``,
    ``scale``; a ``mask`` must cover the same images as ``heatmaps`` when it is one tensor).
    Returns ``(best threshold, {threshold: score})``; ties go to the lower threshold.

    .. warning:: With a small validation set the chosen threshold is noisy. On 45 images,
       two-fold cross-calibration picked thresholds that differed between folds (0.2 vs 0.35)
       and was no better than a fixed 0.3, once worse (count MAE 1.60 -> 2.02). On a large
       one (OAM-TCD trees, E011b) it raised test F1 from 0.40 to 0.59. Calibrate on separate
       folds and use it only if the threshold is stable across them.
    """
    if metric not in ("count_mae", "f1"):
        raise ValueError(f"metric must be 'count_mae' or 'f1', got {metric!r}")
    if "threshold" in find_peaks_kwargs:
        raise ValueError("pass candidate values as 'thresholds', not 'threshold'")
    thresholds = sorted(float(t) for t in thresholds)
    if not thresholds:
        raise ValueError("thresholds is empty")
    batches = [heatmaps] if isinstance(heatmaps, Tensor) else list(heatmaps)
    scores = []
    for hm in batches:
        hm = hm[None] if hm.dim() == 2 else hm
        scores += [(p, s) for p, s in find_peaks(hm, thresholds[0], **find_peaks_kwargs)]
    gts = list(targets.unbind(0)) if isinstance(targets, Tensor) and targets.dim() <= 1 \
        else list(targets)
    if len(gts) != len(scores):
        raise ValueError(f"{len(gts)} targets for {len(scores)} heatmaps")
    as_points = [isinstance(g, Tensor) and g.dim() == 2 for g in gts]
    if any(as_points) and not all(as_points):
        raise ValueError("targets must be all counts or all (N, 2) point tensors")
    if metric == "f1" and not all(as_points):
        raise ValueError("metric='f1' needs ground-truth points, not counts")
    counts = [len(g) if pts else float(g) for g, pts in zip(gts, as_points)]
    curve = {}
    for thr in thresholds:
        kept = [p[s > thr] for p, s in scores]  # strict, as find_peaks
        if metric == "count_mae":
            curve[thr] = sum(abs(len(k) - c) for k, c in zip(kept, counts)) / max(1, len(kept))
        else:
            tp = sum(_match_points(k.float(), g.float(), radius) for k, g in zip(kept, gts))
            n_pred, n_gt = sum(len(k) for k in kept), sum(counts)
            curve[thr] = 2 * tp / (n_pred + n_gt) if n_pred + n_gt else 1.0
    best = (min if metric == "count_mae" else max)(
        thresholds, key=lambda t: (curve[t], t) if metric == "count_mae" else (curve[t], -t))
    return best, curve


@torch.no_grad()
def group_by_offsets(centers: Sequence[Tensor], offsets: Tensor, foreground: Tensor,
                     chunk: int = 65536) -> Tensor:
    """Panoptic-DeepLab grouping: each foreground pixel votes ``pixel + offset`` and joins the
    nearest centre. Returns instance labels ``(B, H, W)`` (centre ``k`` -> label ``k + 1``)."""
    b, _, h, w = offsets.shape
    ys = torch.arange(h, device=offsets.device, dtype=torch.float32)
    xs = torch.arange(w, device=offsets.device, dtype=torch.float32)
    vy = (offsets[:, 0] + ys[:, None]).reshape(b, -1)
    vx = (offsets[:, 1] + xs[None, :]).reshape(b, -1)
    labels = torch.zeros(b, h * w, dtype=torch.long, device=offsets.device)
    for i, c in enumerate(centers):
        if len(c) == 0:
            continue
        c = c.float().to(offsets.device)
        for s in range(0, h * w, chunk):
            d = (vy[i, s:s + chunk, None] - c[None, :, 0]) ** 2 + \
                (vx[i, s:s + chunk, None] - c[None, :, 1]) ** 2
            labels[i, s:s + chunk] = d.argmin(1) + 1
    return labels.view(b, h, w) * foreground.long()


def panoptic_labels(semantic: Tensor, instances: Tensor, thing_classes: Sequence[int],
                    label_divisor: int = 1000) -> Tensor:
    """``class * label_divisor + instance_id`` for thing classes, ``class * label_divisor`` for
    stuff (Cityscapes/COCO panoptic convention); background class 0 stays 0."""
    things = torch.isin(semantic, torch.as_tensor(list(thing_classes), device=semantic.device))
    return semantic * label_divisor + torch.where(things, instances, 0)
