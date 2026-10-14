"""Semantic FPN decoder (Kirillov et al. 2019), following SMP's FPN defaults."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import ConvNormAct, init_decoder_
from .base import DECODERS, check_out_stride, pyramid_levels


class _SegBlock(nn.Module):
    """conv-GN-ReLU, then upsample to each successively shallower level's size."""

    def __init__(self, cin: int, cout: int, n_up: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            ConvNormAct(cin if i == 0 else cout, cout, norm="group") for i in range(max(1, n_up))
        )
        self.n_up = n_up

    def forward(self, x: Tensor, sizes: list) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < self.n_up:
                x = F.interpolate(x, size=sizes[i], mode="bilinear", align_corners=False)
        return x


@DECODERS.register("fpn")
class FPNDecoder(nn.Module):
    """Laterals + top-down pathway on strides >= 4, per-level segmentation branches upsampled to
    stride 4 and summed. Output: ``segmentation_channels`` at stride 4 (``out_stride`` may only
    restate it)."""

    def __init__(self, in_specs: Sequence[FeatureSpec], pyramid_channels: int = 256,
                 segmentation_channels: int = 128, dropout: float = 0.2,
                 out_stride: int | None = None) -> None:
        super().__init__()
        self.levels = pyramid_levels(in_specs)
        check_out_stride("fpn", in_specs[self.levels[0]].stride, out_stride)
        specs = [in_specs[i] for i in self.levels]
        self.lateral = nn.ModuleList(nn.Conv2d(s.channels, pyramid_channels, 1) for s in specs)
        self.seg = nn.ModuleList(
            _SegBlock(pyramid_channels, segmentation_channels, k) for k in range(len(specs))
        )
        self.dropout = nn.Dropout2d(dropout)
        self.out_spec = FeatureSpec(segmentation_channels, specs[0].stride, "fpn")
        init_decoder_(self)

    def forward(self, features: list[Tensor]) -> Tensor:
        cs = [features[i] for i in self.levels]
        p = [None] * len(cs)
        p[-1] = self.lateral[-1](cs[-1])
        for i in range(len(cs) - 2, -1, -1):
            up = F.interpolate(p[i + 1], size=cs[i].shape[-2:], mode="nearest")
            p[i] = self.lateral[i](cs[i]) + up
        sizes = [c.shape[-2:] for c in cs]
        out = self.seg[0](p[0], [])
        for k in range(1, len(p)):
            # level k upsamples through sizes of levels k-1, ..., 0
            out = out + self.seg[k](p[k], [sizes[j] for j in range(k - 1, -1, -1)])
        return self.dropout(out)
