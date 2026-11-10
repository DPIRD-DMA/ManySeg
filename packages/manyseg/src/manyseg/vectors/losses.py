"""Losses of the trained vector models, as plain functions (pure torch).

Each returns its unweighted terms; the trained models summed them with weight 1. Targets come from
a stacked batch of the target encoders in :mod:`manyseg.vectors`, as tensors on the model's device
and at full resolution. Augment with :func:`manyseg.vectors.dihedral`.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from ..instances.losses import as_channel, instance_losses
from ..losses import dice_bce, heatmap_focal, heatmap_peaks, masked_regression


def corner_line_losses(out: Mapping[str, Tensor], targets: Mapping[str, Tensor], *,
                       boundary_pos_weight: float | None = None) -> dict[str, Tensor]:
    """Terms for :func:`corner_line_heads` (footprints, roof faces): ``instances``
    (the sum of :func:`manyseg.instances.instance_losses`), ``corner`` (CenterNet focal loss,
    positives at the heatmap's 3 x 3 peaks above 0.5) and ``direction`` (L1 on the outlines).

    ``targets``: as :func:`footprint_targets` or :func:`roof_face_targets` return them, stacked:
    ``extent``, ``boundary``, ``distance``, ``corner``, ``direction``, ``direction_mask``.
    """
    corner = as_channel(targets["corner"]).float()
    mask = as_channel(targets["direction_mask"]).bool()
    parts = instance_losses(out, targets, boundary_pos_weight=boundary_pos_weight)
    return {"instances": parts["extent"] + parts["boundary"] + parts["distance"],
            "corner": heatmap_focal(out["corner"], corner, pos=heatmap_peaks(corner)),
            "direction": masked_regression(out["direction"], targets["direction"], mask)}


def road_losses(out: Mapping[str, Tensor], targets: Mapping[str, Tensor], *,
                min_keypoints: int = 16) -> dict[str, Tensor]:
    """Terms for :func:`road_heads`: ``mask`` and ``line`` (Dice + BCE), ``keypoints``
    (one focal loss over the three heatmaps) and ``direction`` (L1 of mode 1 on the road surface,
    plus L1 of mode 2 where its target is set).

    ``targets``: as :func:`road_targets` returns them, stacked: ``mask``, ``line``, ``keypoints``,
    ``direction``. ``min_keypoints`` floors the focal loss's normalisation: with a floor of 1, a
    batch without crossings let the summed negatives swamp the decoder and the mask collapsed.
    """
    mask, keypoints = as_channel(targets["mask"]), targets["keypoints"].float()
    direction, pred = targets["direction"].float(), out["direction"]
    crossing = direction[:, 2:].abs().sum(1, keepdim=True) > 0
    return {"mask": dice_bce(out["mask"], mask.float()),
            "line": dice_bce(out["line"], as_channel(targets["line"]).float()),
            "keypoints": heatmap_focal(out["keypoints"], keypoints,
                                       pos=heatmap_peaks(keypoints), min_pos=min_keypoints),
            "direction": (masked_regression(pred[:, :2], direction[:, :2], mask > 0.5)
                          + masked_regression(pred[:, 2:], direction[:, 2:], crossing))}
