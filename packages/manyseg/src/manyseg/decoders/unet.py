"""U-Net decoder (SMP-equivalent reference implementation)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import ConvNormAct, init_decoder_
from .base import DECODERS


UPSAMPLE_MODES = ("nearest", "bilinear", "aligned")


def upsample2x(x: Tensor, size: Sequence[int], mode: str) -> Tensor:
    """Upsample a stride-2s feature map to stride s, output cropped/resized to ``size``.

    ``nearest`` and ``bilinear`` (align_corners=False) treat coarse sample ``j`` as covering
    fine pixels ``2j, 2j+1``. Odd-kernel "same"-padded strided convs instead centre sample ``j``
    on fine pixel ``2j``; ``aligned`` interpolates at fine coordinate ``i / 2`` to match that
    geometry (see experiments/E001_upsample_alignment).
    """
    if mode == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    if mode == "bilinear":
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if mode == "aligned":
        h, w = x.shape[-2], x.shape[-1]
        x = F.pad(x, (0, 1, 0, 1), mode="replicate")
        x = F.interpolate(x, size=(2 * h + 1, 2 * w + 1), mode="bilinear", align_corners=True)
        return x[..., : size[0], : size[1]]
    raise ValueError(f"unknown upsample mode {mode!r}; choose from {UPSAMPLE_MODES}")


class UNetBlock(nn.Module):
    def __init__(
        self, cin: int, cskip: int, cout: int, norm: str | None, upsample: str = "nearest"
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvNormAct(cin + cskip, cout, norm=norm)
        self.conv2 = ConvNormAct(cout, cout, norm=norm)

    def forward(self, x: Tensor, skip: Tensor | None) -> Tensor:
        # Resize to the skip's exact size so odd (ceil-rounded) feature maps stay aligned.
        size = skip.shape[-2:] if skip is not None else (2 * x.shape[-2], 2 * x.shape[-1])
        x = upsample2x(x, size, self.upsample)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


@DECODERS.register("unet")
class UNetDecoder(nn.Module):
    """U-Net decoder: from the deepest feature, repeatedly upsample x2 and fuse a skip.

    One block per factor of two between the deepest stride and ``out_stride``. A block fuses
    the encoder feature with matching stride if there is one.

    Args:
        in_specs: Encoder feature specs (strides must be powers of two).
        channels: Output channels of each block, deepest first. Default: SMP's
            ``(256, 128, 64, 32, 16)`` truncated/extended to the number of blocks.
        out_stride: Stride of the decoder output (1 = input resolution).
        norm: ``"batch"``, ``"group"`` or ``None``.
        upsample: ``"nearest"`` (SMP-equivalent default), ``"bilinear"`` or ``"aligned"``.
    """

    def __init__(
        self,
        in_specs: Sequence[FeatureSpec],
        channels: Sequence[int] | None = None,
        out_stride: int = 1,
        norm: str | None = "batch",
        upsample: str = "nearest",
    ) -> None:
        super().__init__()
        if upsample not in UPSAMPLE_MODES:
            raise ValueError(f"unknown upsample mode {upsample!r}; choose from {UPSAMPLE_MODES}")
        specs = sorted(in_specs, key=lambda s: s.stride)
        for s in specs:
            if s.stride & (s.stride - 1):
                raise ValueError(f"UNetDecoder needs power-of-two strides, got {s.stride}")
        deepest = specs[-1]
        if out_stride > deepest.stride or out_stride & (out_stride - 1):
            raise ValueError(f"invalid out_stride {out_stride}")
        n_blocks = int(math.log2(deepest.stride // out_stride))
        if channels is None:
            channels = tuple(max(16, 256 >> i) for i in range(n_blocks))
        channels = tuple(channels)
        if len(channels) != n_blocks:
            raise ValueError(
                f"channels has {len(channels)} entries but {n_blocks} blocks are needed "
                f"(stride {deepest.stride} -> {out_stride})"
            )
        by_stride = {s.stride: (i, s) for i, s in enumerate(in_specs)}
        self._deepest_index = by_stride[deepest.stride][0]
        self._skip_index: list[int] = []
        blocks = []
        cin, stride = deepest.channels, deepest.stride
        for cout in channels:
            stride //= 2
            idx, spec = by_stride.get(stride, (-1, None))
            self._skip_index.append(idx)
            blocks.append(UNetBlock(cin, spec.channels if spec else 0, cout, norm, upsample))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.channels = channels
        self.out_spec = FeatureSpec(channels[-1], out_stride, "unet")
        init_decoder_(self)

    def forward(self, features: list[Tensor]) -> Tensor:
        x = features[self._deepest_index]
        for block, idx in zip(self.blocks, self._skip_index):
            x = block(x, features[idx] if idx >= 0 else None)
        return x
