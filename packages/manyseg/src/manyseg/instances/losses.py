"""Losses for the instance heads (pure torch)."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from ..losses import dice_bce, masked_regression


def as_channel(t: Tensor) -> Tensor:
    """A one-channel target as ``(B, 1, H, W)``, from ``(B, H, W)`` or ``(B, 1, H, W)``."""
    return t[:, None] if t.ndim == 3 else t


def instance_losses(out: Mapping[str, Tensor], targets: Mapping[str, Tensor], *,
                    boundary_pos_weight: float | None = None) -> dict[str, Tensor]:
    """The unweighted loss terms of :func:`instance_heads`: ``extent`` and ``boundary`` (Dice +
    BCE on logits), ``distance`` (L1 inside the objects). Sum them for the trained models' loss.

    ``targets``: ``extent``, ``boundary`` and ``distance`` as ``(B, H, W)`` or ``(B, 1, H, W)``,
    from :func:`instance_targets` or a stacked batch of the target encoders in
    :mod:`manyseg.vectors`, on the outputs' grid. ``distance_mask`` is optional (default: where
    ``extent`` is set).
    """
    extent = as_channel(targets["extent"]).float()
    boundary = as_channel(targets["boundary"]).float()
    mask = as_channel(targets["distance_mask"]) if "distance_mask" in targets else extent > 0
    return {"extent": dice_bce(out["extent"], extent),
            "boundary": dice_bce(out["boundary"], boundary, pos_weight=boundary_pos_weight),
            "distance": masked_regression(out["distance"], as_channel(targets["distance"]), mask)}
