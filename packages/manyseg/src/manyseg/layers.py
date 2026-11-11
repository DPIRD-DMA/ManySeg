"""Small building blocks shared by decoders and heads."""

from __future__ import annotations

from torch import Tensor, nn


class GroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` whose output keeps the input's dtype: CUDA autocast runs ``group_norm``
    in float32, which would otherwise carry float32 activations (and the ReLU after them) into
    the next layer, which casts back to bf16 anyway (-19% decoder training memory in bf16)."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).to(x.dtype)


def group_norm_groups(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of ``channels`` that is at most ``max_groups``: ``min(32, channels)``
    whenever that divides (the usual widths: unchanged), and a valid group count otherwise
    (e.g. 24 for 48 channels) instead of GroupNorm's divisibility error."""
    return next(g for g in range(min(max_groups, channels), 0, -1) if channels % g == 0)


NORMS = ("batch", "group", None)
"""Normalisation choices for decoder blocks and pixel-head hidden layers."""
NORM_ALIASES = {"bn": "batch", "gn": "group", "none": None}
"""Other accepted spellings (saved configs use ``"bn"`` / ``"gn"``)."""


def normalize_norm(norm: str | None, what: str = "norm") -> str | None:
    """``norm`` as one of :data:`NORMS`: ``"batch"`` (alias ``"bn"``), ``"group"`` (alias
    ``"gn"``) or None (alias ``"none"``); anything else raises ValueError."""
    if isinstance(norm, str):
        norm = NORM_ALIASES.get(norm, norm)
    if norm not in NORMS:
        raise ValueError(f"{what} must be one of 'batch', 'group' or None ('bn', 'gn' and "
                         f"'none' are accepted aliases), got {norm!r}")
    return norm


def make_norm(norm: str | None, channels: int) -> nn.Module:
    """The norm layer for ``norm`` (see :func:`normalize_norm`); ``nn.Identity`` for None."""
    norm = normalize_norm(norm)
    if norm is None:
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    return GroupNorm(group_norm_groups(channels), channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, norm: str | None = "batch") -> None:
        norm = normalize_norm(norm)  # before the bias: "none" is no norm, so the conv has one
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=norm is None),
            make_norm(norm, cout),
            nn.ReLU(inplace=True),
        )


def init_decoder_(module: nn.Module) -> None:
    """Kaiming-uniform (fan_in, ReLU) convs, unit/zero norms: matches SMP's decoder init."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
