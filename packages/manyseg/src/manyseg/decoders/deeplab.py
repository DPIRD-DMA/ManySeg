"""DeepLabV3+ decoder (Chen et al. 2018), following SMP's defaults (separable convs)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import init_decoder_
from ..utils.spatial import resize_bilinear
from .base import DECODERS, check_out_stride, pyramid_levels


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, dilation: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=dilation, dilation=dilation, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, cin: int, cout: int, rates: Sequence[int], dropout: float = 0.5) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout),
                           nn.ReLU(inplace=True))]
            + [SeparableConvBNReLU(cin, cout, r) for r in rates]
        )
        # Image-pooling branch without BatchNorm: BN on a 1x1 map fails at batch size 1.
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1),
                                  nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(cout * (len(rates) + 2), cout, 1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        outs = [b(x) for b in self.branches]
        outs.append(self.pool(x).expand(-1, -1, x.shape[-2], x.shape[-1]))
        return self.project(torch.cat(outs, dim=1))


@DECODERS.register("deeplabv3plus")
class DeepLabV3PlusDecoder(nn.Module):
    """ASPP on the deepest feature + low-level (stride-4) skip. Output: 256 ch at stride 4.

    Atrous rates default to (12, 24, 36) at output stride 16 and are scaled for other deepest
    strides (e.g. (6, 12, 18) at 32). SMP dilates the encoder to output stride 16 instead;
    pass ``backbone_kwargs={"output_stride": 16}`` (timm) to reproduce that.
    """

    def __init__(self, in_specs: Sequence[FeatureSpec], channels: int = 256,
                 low_level_channels: int = 48, atrous_rates: Sequence[int] | None = None,
                 out_stride: int | None = None) -> None:
        super().__init__()
        levels = pyramid_levels(in_specs)
        check_out_stride("deeplabv3plus", in_specs[levels[0]].stride, out_stride)
        self.low, self.high = levels[0], max(range(len(in_specs)),
                                            key=lambda i: in_specs[i].stride)
        deep = in_specs[self.high]
        if atrous_rates is None:
            atrous_rates = tuple(max(1, r * 16 // deep.stride) for r in (12, 24, 36))
        self.atrous_rates = tuple(atrous_rates)
        self.aspp = nn.Sequential(ASPP(deep.channels, channels, self.atrous_rates),
                                  SeparableConvBNReLU(channels, channels))
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_specs[self.low].channels, low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels), nn.ReLU(inplace=True))
        self.fuse = SeparableConvBNReLU(channels + low_level_channels, channels)
        self.out_spec = FeatureSpec(channels, in_specs[self.low].stride, "deeplabv3plus")
        init_decoder_(self)

    @property
    def read_levels(self) -> list[int]:
        """Indices of the encoder levels this decoder reads."""
        return sorted({self.low, self.high})

    def forward(self, features: list[Tensor]) -> Tensor:
        low = features[self.low]
        x = self.aspp(features[self.high])
        x = resize_bilinear(x, low.shape[-2:])
        return self.fuse(torch.cat([x, self.low_proj(low)], dim=1))
