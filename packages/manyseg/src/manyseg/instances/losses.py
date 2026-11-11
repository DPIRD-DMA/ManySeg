"""Losses for the instance heads (pure torch)."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn.functional as F
from torch import Tensor

from ..losses import dice_bce, masked_regression, multiclass_dice


def as_channel(t: Tensor) -> Tensor:
    """A one-channel target as ``(B, 1, H, W)``, from ``(B, H, W)`` or ``(B, 1, H, W)``."""
    return t[:, None] if t.ndim == 3 else t


def _extent_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Binary Dice + BCE for a 1-channel extent; cross-entropy + multiclass Dice for more."""
    t = as_channel(target)
    if logits.shape[1] == 1:
        if t.numel() and not t.is_floating_point() and int(t.max()) > 1:
            raise ValueError(
                f"extent target holds class ids up to {int(t.max())} but the extent head has 1 "
                "channel; use instance_heads(num_classes=K) (K + 1 channels) for class-id "
                "targets, or pass a binary extent (instance_mask > 0)")
        if t.numel() and t.is_floating_point() and float(t.max()) > 1:
            raise ValueError(
                f"extent target has values up to {float(t.max())} but the extent head has 1 "
                "channel (binary, values in [0, 1]); use instance_heads(num_classes=K) for "
                "class-id targets")
        return dice_bce(logits, t.float())
    if t.shape[1] != 1:
        raise ValueError(f"a {logits.shape[1]}-channel extent head needs class-id targets "
                         f"(B, H, W) or (B, 1, H, W), got {tuple(target.shape)}")
    ids = t[:, 0].long()
    return F.cross_entropy(logits.float(), ids) + multiclass_dice(logits, ids)


def instance_losses(out: Mapping[str, Tensor], targets: Mapping[str, Tensor], *,
                    boundary_pos_weight: float | None = None) -> dict[str, Tensor]:
    """The unweighted loss terms of :func:`instance_heads`: ``extent``, ``boundary`` (Dice +
    BCE on logits) and ``distance`` (L1 inside the objects). Sum them for the trained models'
    loss.

    ``extent`` follows the head: a 1-channel head (``num_classes=1``) gets binary Dice + BCE
    against a 0/1 target; a ``num_classes + 1``-channel head gets cross-entropy + multiclass
    Dice (:func:`manyseg.losses.multiclass_dice`) against class ids (background 0), e.g. the
    ``extent`` of :func:`instance_targets` with ``semantic=``. A 1-channel head with class ids
    above 1 in the target raises.

    ``targets``: ``extent``, ``boundary`` and ``distance`` as ``(B, H, W)`` or ``(B, 1, H, W)``,
    from :func:`instance_targets` or a stacked batch of the target encoders in
    :mod:`manyseg.vectors`, on the outputs' grid. ``distance_mask`` is optional (default: where
    ``extent`` is set, i.e. ``extent > 0``, any class).
    """
    extent = as_channel(targets["extent"])
    boundary = as_channel(targets["boundary"]).float()
    mask = as_channel(targets["distance_mask"]) if "distance_mask" in targets else extent > 0
    return {"extent": _extent_loss(out["extent"], targets["extent"]),
            "boundary": dice_bce(out["boundary"], boundary, pos_weight=boundary_pos_weight),
            "distance": masked_regression(out["distance"], as_channel(targets["distance"]), mask)}
