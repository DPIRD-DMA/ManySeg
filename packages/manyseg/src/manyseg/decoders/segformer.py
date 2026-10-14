"""SegFormer all-MLP decoder (Xie et al. 2021): lightweight, transformer-friendly."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import ConvNormAct, init_decoder_
from .base import DECODERS, check_out_stride, pyramid_levels


@DECODERS.register("segformer")
class SegFormerDecoder(nn.Module):
    """Per-level linear (1x1) to ``channels``, upsample to the shallowest level, concat, fuse.
    Output: ``channels`` at stride 4 (``out_stride`` may only restate it)."""

    def __init__(self, in_specs: Sequence[FeatureSpec], channels: int = 256,
                 dropout: float = 0.1, out_stride: int | None = None) -> None:
        super().__init__()
        self.levels = pyramid_levels(in_specs)
        check_out_stride("segformer", in_specs[self.levels[0]].stride, out_stride)
        self.linear = nn.ModuleList(nn.Conv2d(in_specs[i].channels, channels, 1)
                                    for i in self.levels)
        self.fuse = ConvNormAct(len(self.levels) * channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.out_spec = FeatureSpec(channels, in_specs[self.levels[0]].stride, "segformer")
        init_decoder_(self)

    def forward(self, features: list[Tensor]) -> Tensor:
        size = features[self.levels[0]].shape[-2:]
        outs = []
        for i, lin in zip(self.levels, self.linear):
            y = lin(features[i])
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            outs.append(y)
        return self.dropout(self.fuse(torch.cat(outs, dim=1)))
