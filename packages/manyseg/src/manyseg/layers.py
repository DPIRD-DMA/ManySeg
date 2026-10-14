"""Small building blocks shared by decoders and heads."""

from __future__ import annotations

from torch import nn


def make_norm(norm: str | None, channels: int) -> nn.Module:
    if norm is None or norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        return nn.GroupNorm(min(32, channels), channels)
    raise ValueError(f"unknown norm {norm!r}; choose from 'batch', 'group', None")


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, norm: str | None = "batch") -> None:
        has_norm = norm not in (None, "none")
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=not has_norm),
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
