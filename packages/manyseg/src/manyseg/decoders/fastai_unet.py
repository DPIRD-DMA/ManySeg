"""fastai-style Dynamic U-Net decoder.

Reimplements the decoder of fastai's ``DynamicUnet`` (Apache-2.0, fastai/vision/models/unet.py)
on ManySeg's feature contract, so its design can be compared against other decoders with an
identical encoder and training recipe. Differences from fastai: works from ``FeatureSpec``s
rather than hooks, and the final 1x1 classifier is ManySeg's head.

Characteristics: BN+ReLU on the encoder output and a
2x-wide "middle" block; ICNR-initialised pixel-shuffle upsampling; BatchNorm on each skip before
concatenation and no norm inside the decoder convs; widths ``up//2 + skip`` (halved only in
the last block); a final shuffle to full resolution, concatenation of the input image and a
residual block.

Reduced output stride (``out_stride``; not in fastai). The decoder is cut, never re-shaped:
every kept module is identical (name, shape, width) to the default decoder's, so
``out_stride=1`` stays parameter- and FLOP-identical to fastai's ``DynamicUnet`` and coarser
strides only remove modules. With a stride-2..32 encoder (ResNet):

* ``out_stride=1`` (default): everything above; ``last_cross`` defaults to True.
* ``out_stride=2``: drops the final pixel shuffle (stride 2 -> 1), the input-image
  concatenation and the input-resolution residual block (``last_cross``/``res``). The output
  is the last up-block (stride 2), which keeps fastai's halved width.
* ``out_stride=4``: additionally drops the last up-block (the one fusing the stride-2 skip).

In general, up-blocks are built for skips with stride >= ``out_stride``; the final shuffle is
kept only if the shallowest kept skip is coarser than ``out_stride`` (e.g. a stride-4..32
transformer at ``out_stride=2``: blocks to stride 4, then the shuffle to stride 2); any
remaining factor is a nearest resize, as in fastai. ``last_cross`` (the input-resolution
cross) exists only at ``out_stride=1``; asking for it at a coarser stride is an error.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..utils.spatial import fit_nearest, is_2x, resize_nearest, upsample2x_nearest
from .base import DECODERS, check_input_spec, check_power_of_two_stride


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
        y = self.shuffle(F.relu(self.conv(x), inplace=True))
        # CUDA's pixel_shuffle returns a contiguous (NCHW) tensor even for channels_last input;
        # the mixed-layout concat after it would then convert the whole buffer
        if x.is_contiguous(memory_format=torch.channels_last) and not y.is_contiguous(
                memory_format=torch.channels_last):
            y = y.contiguous(memory_format=torch.channels_last)
        return y


class _Block(nn.Module):
    def __init__(self, up_c: int, skip_c: int, final_div: bool) -> None:
        super().__init__()
        self.shuf = PixelShuffleICNR(up_c, up_c // 2)
        self.bn = _bn(skip_c)
        ni = up_c // 2 + skip_c
        nf = ni if final_div else ni // 2
        self.conv1, self.conv2 = _conv(ni, nf), _conv(nf, nf)
        self.out_channels = nf
        self.exact = False  # set by the model when inputs are padded to the stride

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # the shuffled map is 2x, the skip 2x or 2x - 1: a crop; the model says when
        # that holds, so export traces no shape test
        up = fit_nearest(self.shuf(x), skip.shape[-2:], self.exact)
        return self.conv2(self.conv1(F.relu(torch.cat([up, self.bn(skip)], dim=1), inplace=True)))


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(_conv(c, c), _conv(c, c, act=False))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.convs(x) + x, inplace=True)


@DECODERS.register("fastai_unet")
class FastaiUNetDecoder(nn.Module):
    """fastai's Dynamic U-Net decoder (``decoder="fastai_unet"``), parameter- and
    FLOP-identical to fastai's, see the module docstring.

    Args:
        in_specs: The input image's spec followed by the encoder features' specs.
        last_cross: Add fastai's last cross stage (the input image concatenated, then a
            residual block) at input resolution; default: when ``out_stride`` is 1.
        out_stride: Output stride (a power of two up to the deepest encoder stride); stages
            below it are dropped.
    """

    wants_input = True

    def __init__(self, in_specs: Sequence[FeatureSpec], last_cross: bool | None = None,
                 out_stride: int = 1) -> None:
        super().__init__()
        check_input_spec(in_specs)
        self.in_channels = in_specs[0].channels
        enc = list(in_specs[1:])
        order = sorted(range(len(enc)), key=lambda i: enc[i].stride)
        check_power_of_two_stride("FastaiUNetDecoder", out_stride, enc[order[-1]].stride)
        if last_cross is None:
            last_cross = out_stride == 1
        elif last_cross and out_stride != 1:
            raise ValueError("last_cross (input-image cross + residual block) is an input-"
                             f"resolution stage; it needs out_stride=1, got {out_stride}")
        self.out_stride = out_stride
        self.deep = order[-1] + 1  # +1: index into [input, *features]
        all_skips = [i + 1 for i in reversed(order[:-1])]  # deep -> shallow
        self.skips = [i for i in all_skips if in_specs[i].stride >= out_stride]
        ni = enc[order[-1]].channels
        self.head_bn = _bn(ni)
        self.middle = nn.Sequential(_conv(ni, ni * 2), _conv(ni * 2, ni))
        blocks, c = [], ni
        prev = enc[order[-1]].stride
        for k, i in enumerate(self.skips):  # widths as in the full decoder (prefix property)
            blk = _Block(c, in_specs[i].channels, final_div=k != len(all_skips) - 1)
            # the shuffle doubles the size: a crop to the skip only if it is 2x (out_indices
            # may skip a level)
            blk.step_2x = prev == 2 * in_specs[i].stride
            prev = in_specs[i].stride
            blocks.append(blk)
            c = blk.out_channels
        self.blocks = nn.ModuleList(blocks)
        shallowest = in_specs[self.skips[-1]].stride if self.skips else enc[order[-1]].stride
        self.final_shuffle = (PixelShuffleICNR(c, c) if shallowest > max(1, out_stride)
                              else None)
        reached = shallowest // 2 if self.final_shuffle is not None else shallowest
        self._resize = out_stride > 1 and reached != out_stride  # decided here, not per call
        self._reached = reached
        self._deepest_stride = enc[order[-1]].stride
        self._exact = False
        self.last_cross = last_cross
        if last_cross:
            c = c + self.in_channels
            self.res = _ResBlock(c)
        self.out_spec = FeatureSpec(c, out_stride, "fastai_unet")

    @property
    def read_levels(self) -> list[int]:
        """Indices of the encoder levels this decoder reads."""
        return sorted({self.deep - 1, *(i - 1 for i in self.skips)})

    def _set_input_divisor(self, divisor: int) -> None:
        """Inputs padded to a multiple of the deepest stride: each shuffled map matches its skip
        and the output the input, so the nearest resizes are crops."""
        self._exact = divisor % self._deepest_stride == 0
        for blk in self.blocks:
            blk.exact = self._exact and blk.step_2x

    def forward(self, features: list[Tensor]) -> Tensor:
        inp = features[0]
        x = self.middle(F.relu(self.head_bn(features[self.deep])))
        for blk, i in zip(self.blocks, self.skips):
            x = blk(x, features[i])
        if self.final_shuffle is not None:
            x = self.final_shuffle(x)
        if self.out_stride > 1:
            if self._resize:
                s = self.out_stride
                size = ((inp.shape[-2] + s - 1) // s, (inp.shape[-1] + s - 1) // s)
                x = resize_nearest(x, size)
            return x
        size = inp.shape[-2:]
        if self._reached == 1:  # already at input resolution: a crop when padded
            x = fit_nearest(x, size, self._exact)
        elif self._reached == 2 and (self._exact or (not torch.compiler.is_compiling() and is_2x(
                x.shape[-2], size[0]) and is_2x(x.shape[-1], size[1]))):
            x = upsample2x_nearest(x, size)  # e.g. ConvNeXt: no stride-2 feature
        else:
            x = resize_nearest(x, size)
        if self.last_cross:
            # the input image is float32 under autocast: concatenated as is, it would make the
            # input-resolution residual block float32 (+20% training memory)
            x = self.res(torch.cat([x, inp.to(x.dtype)], dim=1))
        return x
