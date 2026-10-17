"""Necks that turn single-scale transformer features into a multi-scale pyramid."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec


class ViTPyramidNeck(nn.Module):
    """Resample features from several ViT blocks onto a stride pyramid.

    Level ``i`` takes block feature ``i`` (stride = patch size), projects it with a 1x1 conv to
    ``channels[i]``, resizes it to ``ceil(H / strides[i])`` (bilinear; works for any patch size,
    including 14), then refines with a 3x3 conv-BN-ReLU. Similar in spirit to DPT reassembly
    and ViTDet's simple feature pyramid, but defined by target sizes rather than fixed factors.
    """

    def __init__(self, in_channels: int, strides: Sequence[int], channels: Sequence[int],
                 patch_stride: int) -> None:
        super().__init__()
        if len(strides) != len(channels):
            raise ValueError("strides and channels must have equal length")
        self.strides = tuple(strides)
        self.patch_stride = patch_stride
        self.proj = nn.ModuleList(nn.Conv2d(in_channels, c, 1) for c in channels)
        self.refine = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                          nn.ReLU(inplace=True))
            for c in channels
        )
        self.active = (True,) * len(self.strides)
        self.feature_specs = tuple(FeatureSpec(c, s, f"neck.{i}")
                                   for i, (c, s) in enumerate(zip(channels, strides)))

    def _restrict(self, levels) -> None:
        """Drop the projections of levels nothing reads; those come out as ``None``."""
        self.active = tuple(i in set(levels) for i in range(len(self.strides)))
        for i, on in enumerate(self.active):
            if not on:
                self.proj[i], self.refine[i] = nn.Identity(), nn.Identity()

    def forward(self, feats: list[Tensor], h: int, w: int) -> list[Tensor | None]:
        out = []
        for i, (f, s, proj, refine) in enumerate(zip(feats, self.strides, self.proj,
                                                     self.refine)):
            if not self.active[i]:
                out.append(None)
                continue
            f = proj(f)
            size = ((h + s - 1) // s, (w + s - 1) // s)
            if s != self.patch_stride:
                f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            out.append(refine(f))
        return out


def default_pyramid_channels(embed_dim: int, n: int) -> tuple[int, ...]:
    """embed_dim / 2^(n-1-i), floored at 32: deepest level keeps the full width."""
    return tuple(max(32, embed_dim >> (n - 1 - i)) for i in range(n))
