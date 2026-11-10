"""UPerNet decoder (Xiao et al. 2018): PPM on the deepest level + FPN, fused at the shallowest
level read (stride 4 on a standard pyramid)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import ConvNormAct, init_decoder_
from ..utils.spatial import resize_bilinear
from .base import DECODERS, check_out_stride, pyramid_levels


class PPM(nn.Module):
    """Pyramid pooling. GroupNorm (not BN) so 1x1 pooled maps work at batch size 1."""

    def __init__(self, cin: int, cout: int, scales: Sequence[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.convs = nn.ModuleList(ConvNormAct(cin, cout, 1, norm="group") for _ in scales)

    def forward(self, x: Tensor) -> list[Tensor]:
        size = x.shape[-2:]
        return [resize_bilinear(conv(F.adaptive_avg_pool2d(x, s)), size)
                for s, conv in zip(self.scales, self.convs)]


@DECODERS.register("upernet")
class UPerNetDecoder(nn.Module):
    """UPerNet decoder (``decoder="upernet"``): a pyramid pooling module on the deepest level
    and an FPN over the (at most 4) deepest levels with stride >= 4, fused at the shallowest
    of them: stride 4 on a standard stride 4..32 pyramid, coarser when the encoder has no
    stride-4 level or more than 4 such levels.

    Args:
        in_specs: The encoder features' specs.
        channels: Width of the PPM, FPN and fusion convolutions.
        pool_scales: Output sizes of the pyramid pooling branches.
        out_stride: May only restate the native output stride (the fusion's, see above;
            any other value is an error, see ``check_out_stride``).
    """

    def __init__(self, in_specs: Sequence[FeatureSpec], channels: int = 256,
                 pool_scales: Sequence[int] = (1, 2, 3, 6), out_stride: int | None = None) -> None:
        super().__init__()
        self.levels = pyramid_levels(in_specs)
        check_out_stride("upernet", in_specs[self.levels[0]].stride, out_stride)
        specs = [in_specs[i] for i in self.levels]
        self.ppm = PPM(specs[-1].channels, channels, pool_scales)
        self.bottleneck = ConvNormAct(specs[-1].channels + len(pool_scales) * channels, channels)
        self.lateral = nn.ModuleList(ConvNormAct(s.channels, channels, 1) for s in specs[:-1])
        self.fpn = nn.ModuleList(ConvNormAct(channels, channels) for _ in specs[:-1])
        self.fuse = ConvNormAct(len(specs) * channels, channels)
        self.out_spec = FeatureSpec(channels, specs[0].stride, "upernet")
        init_decoder_(self)

    def forward(self, features: list[Tensor]) -> Tensor:
        cs = [features[i] for i in self.levels]
        top = self.bottleneck(torch.cat([cs[-1], *self.ppm(cs[-1])], dim=1))
        lats = [lat(c) for lat, c in zip(self.lateral, cs[:-1])] + [top]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + resize_bilinear(lats[i + 1], lats[i].shape[-2:])
        outs = [conv(lat) for conv, lat in zip(self.fpn, lats[:-1])] + [lats[-1]]
        size = outs[0].shape[-2:]
        outs = [outs[0]] + [resize_bilinear(o, size) for o in outs[1:]]
        return self.fuse(torch.cat(outs, dim=1))
