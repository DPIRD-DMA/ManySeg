"""Adapting pretrained stem convolutions to an arbitrary number of input channels."""

from __future__ import annotations

import torch
from torch import Tensor, nn

WEIGHT_ADAPTERS = ("repeat", "mean", "rgb_plus_random", "rgb_plus_zero")
CHANNEL_ADAPTERS = (*WEIGHT_ADAPTERS, "projection")


def adapt_conv_weight(
    weight: Tensor,
    in_channels: int,
    method: str = "repeat",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a new conv weight of shape ``(O, in_channels, kH, kW)`` derived from ``weight``.

    Methods:
        ``repeat``: timm's rule. ``C == 1`` sums the filters; otherwise filters are cycled and
            scaled by ``C0 / C``.
        ``mean``: every channel gets ``mean(W) * C0 / C``, which keeps the response to a
            channel-constant input for any ``C``.
        ``rgb_plus_random``: original filters kept exactly; extra channels ~ N(0, (std(W)/2)^2).
        ``rgb_plus_zero``: original filters kept exactly; extra channels zero, so the model is
            initially identical to the RGB model applied to the first ``C0`` channels.
    """
    if method not in WEIGHT_ADAPTERS:
        raise ValueError(f"unknown channel adapter {method!r}; choose from {WEIGHT_ADAPTERS}")
    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")
    out_ch, c0, kh, kw = weight.shape
    w = weight.detach()
    if in_channels == c0:
        return w.clone()

    if method == "repeat":
        if in_channels == 1:
            return w.sum(1, keepdim=True)
        idx = torch.arange(in_channels, device=w.device) % c0
        return w[:, idx] * (c0 / in_channels)

    if method == "mean":
        return w.mean(1, keepdim=True).expand(out_ch, in_channels, kh, kw) * (c0 / in_channels)

    if in_channels < c0:
        raise ValueError(f"{method!r} needs at least {c0} input channels, got {in_channels}")
    extra_shape = (out_ch, in_channels - c0, kh, kw)
    if method == "rgb_plus_zero":
        extra = w.new_zeros(extra_shape)
    else:
        extra = torch.randn(extra_shape, generator=generator, dtype=w.dtype, device=w.device)
        extra = extra * (w.float().std() * 0.5).to(w.dtype)
    return torch.cat([w, extra], dim=1)


class ProjectionAdapter(nn.Module):
    """Learned 1x1 projection from ``in_channels`` to the ``out_channels`` a stem expects.

    Initialised so that output channel ``o`` copies input channel ``o % in_channels``: with
    ``in_channels >= out_channels`` the pretrained model initially sees the first channels
    unchanged.
    """

    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for o in range(out_channels):
                self.proj.weight[o, o % in_channels] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
