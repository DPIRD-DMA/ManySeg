"""fastai-style Dynamic U-Net decoder.

Reimplements the decoder of fastai's ``DynamicUnet`` (Apache-2.0, fastai/vision/models/unet.py)
on ManySeg's feature contract, so its design can be compared against other decoders with an
identical encoder and training recipe. Differences from fastai: works from ``FeatureSpec``s
rather than hooks, and the final 1x1 classifier is ManySeg's head.

Characteristics (see docs/architecture_review.md section 1): BN+ReLU on the encoder output and a
2x-wide "middle" block; ICNR-initialised pixel-shuffle upsampling; BatchNorm on each skip before
concatenation and no norm inside the decoder convs; widths ``up//2 + skip`` (halved only in
the last block); a final shuffle to full resolution, concatenation of the input image and a
residual block.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from .base import DECODERS, check_input_spec


def _conv(cin: int, cout: int, k: int = 3, act: bool = True) -> nn.Sequential:
    conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=True)
    nn.init.kaiming_normal_(conv.weight)
    nn.init.zeros_(conv.bias)
    return nn.Sequential(conv, nn.ReLU(inplace=True)) if act else nn.Sequential(conv)


def _bn(c: int) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(c)
    nn.init.ones_(bn.weight)
    nn.init.constant_(bn.bias, 1e-3)  # fastai BatchNorm init
    return bn


class PixelShuffleICNR(nn.Module):
    """1x1 conv to 4*cout, ReLU, PixelShuffle(2). ICNR init: starts as nearest upsampling."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout * 4, 1)
        k = torch.empty(cout, cin, 1, 1)
        nn.init.kaiming_normal_(k)
        with torch.no_grad():
            self.conv.weight.copy_(k.repeat_interleave(4, dim=0))
            self.conv.bias.zero_()
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(F.relu(self.conv(x)))


class _Block(nn.Module):
    def __init__(self, up_c: int, skip_c: int, final_div: bool) -> None:
        super().__init__()
        self.shuf = PixelShuffleICNR(up_c, up_c // 2)
        self.bn = _bn(skip_c)
        ni = up_c // 2 + skip_c
        nf = ni if final_div else ni // 2
        self.conv1, self.conv2 = _conv(ni, nf), _conv(nf, nf)
        self.out_channels = nf

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        up = self.shuf(x)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="nearest")
        return self.conv2(self.conv1(F.relu(torch.cat([up, self.bn(skip)], dim=1))))


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(_conv(c, c), _conv(c, c, act=False))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.convs(x) + x)


@DECODERS.register("fastai_unet")
class FastaiUNetDecoder(nn.Module):
    wants_input = True

    def __init__(self, in_specs: Sequence[FeatureSpec], last_cross: bool = True) -> None:
        super().__init__()
        check_input_spec(in_specs)
        self.in_channels = in_specs[0].channels
        enc = list(in_specs[1:])
        order = sorted(range(len(enc)), key=lambda i: enc[i].stride)
        self.deep = order[-1] + 1  # +1: index into [input, *features]
        self.skips = [i + 1 for i in reversed(order[:-1])]  # deep -> shallow
        ni = enc[order[-1]].channels
        self.head_bn = _bn(ni)
        self.middle = nn.Sequential(_conv(ni, ni * 2), _conv(ni * 2, ni))
        blocks, c = [], ni
        for k, i in enumerate(self.skips):
            blk = _Block(c, in_specs[i].channels, final_div=k != len(self.skips) - 1)
            blocks.append(blk)
            c = blk.out_channels
        self.blocks = nn.ModuleList(blocks)
        shallowest = in_specs[self.skips[-1]].stride if self.skips else enc[order[-1]].stride
        self.final_shuffle = PixelShuffleICNR(c, c) if shallowest > 1 else None
        self.last_cross = last_cross
        if last_cross:
            c = c + self.in_channels
            self.res = _ResBlock(c)
        self.out_spec = FeatureSpec(c, 1, "fastai_unet")

    def forward(self, features: list[Tensor]) -> Tensor:
        inp = features[0]
        x = self.middle(F.relu(self.head_bn(features[self.deep])))
        for blk, i in zip(self.blocks, self.skips):
            x = blk(x, features[i])
        if self.final_shuffle is not None:
            x = self.final_shuffle(x)
        if x.shape[-2:] != inp.shape[-2:]:
            x = F.interpolate(x, size=inp.shape[-2:], mode="nearest")
        if self.last_cross:
            x = self.res(torch.cat([x, inp], dim=1))
        return x
