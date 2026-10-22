"""U-Net decoder (SMP-equivalent reference implementation)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import ConvNormAct, init_decoder_
from ..utils.spatial import is_2x, resize_bilinear, resize_nearest, upsample2x_nearest
from .base import DECODERS, check_power_of_two_stride

UPSAMPLE_MODES = ("nearest", "bilinear", "aligned")


def upsample2x(x: Tensor, size: Sequence[int], mode: str, exact_2x: bool = False) -> Tensor:
    """Upsample a stride-2s feature map to stride s, output cropped/resized to ``size``.

    ``nearest`` uses ``F.interpolate(scale_factor=2)`` (as SMP; fast backward) when ``size`` is
    known to be 2x the input (``exact_2x``: the model pads inputs to the stride, so no check is
    traced under torch.compile / export) or is, in eager mode, ``2n`` / ``2n - 1``; otherwise
    the integer-index gather (:func:`~manyseg.utils.spatial.resize_nearest`). Both give the
    same values.

    ``nearest`` and ``bilinear`` (align_corners=False) treat coarse sample ``j`` as covering
    fine pixels ``2j, 2j+1``. Odd-kernel "same"-padded strided convs instead centre sample ``j``
    on fine pixel ``2j``; ``aligned`` interpolates at fine coordinate ``i / 2`` to match that
    geometry (see experiments/E001_upsample_alignment).
    """
    if mode == "nearest":
        if exact_2x or (not torch.compiler.is_compiling() and is_2x(x.shape[-2], size[0])
                        and is_2x(x.shape[-1], size[1])):
            return upsample2x_nearest(x, size)
        return resize_nearest(x, size)
    # bilinear outputs keep the input dtype: CUDA autocast returns float32, and the skip
    # concat after this would make the whole block float32 (E027: -7% training memory)
    if mode == "bilinear":
        return resize_bilinear(x, size)
    if mode == "aligned":
        h, w = x.shape[-2], x.shape[-1]
        y = F.pad(x, (0, 1, 0, 1), mode="replicate")
        y = F.interpolate(y, size=(2 * h + 1, 2 * w + 1), mode="bilinear", align_corners=True)
        return y[..., : size[0], : size[1]].to(x.dtype)
    raise ValueError(f"unknown upsample mode {mode!r}; choose from {UPSAMPLE_MODES}")


class UNetBlock(nn.Module):
    def __init__(
        self, cin: int, cskip: int, cout: int, norm: str | None, upsample: str = "nearest"
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.exact_2x = False  # set by the model when inputs are padded to the stride
        self.conv1 = ConvNormAct(cin + cskip, cout, norm=norm)
        self.conv2 = ConvNormAct(cout, cout, norm=norm)

    def forward(self, x: Tensor, skip: Tensor | None, size=None) -> Tensor:
        # Resize to the skip's exact size so odd (ceil-rounded) feature maps stay aligned.
        exact = self.exact_2x or (skip is None and size is None)  # no skip: 2x by construction
        if size is None:
            size = skip.shape[-2:] if skip is not None else (2 * x.shape[-2], 2 * x.shape[-1])
        x = upsample2x(x, size, self.upsample, exact)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


@DECODERS.register("unet")
class UNetDecoder(nn.Module):
    """U-Net decoder: from the deepest feature, repeatedly upsample x2 and fuse a skip.

    One block per factor of two between the deepest stride and ``out_stride``. A block fuses
    the encoder feature with matching stride if there is one. ``out_stride > 1`` builds only
    the blocks down to that stride (the finest blocks are not built, so they cost no
    parameters or FLOPs); the kept blocks are identical to the default decoder's, and the head
    upsamples the coarser output (see ``docs/api_design.md``, "Spatial policy").

    Args:
        in_specs: Encoder feature specs (strides must be powers of two).
        channels: Output channels of each block, deepest first. Default: SMP's
            ``(256, 128, 64, 32, 16)`` truncated/extended to the number of blocks.
        out_stride: Stride of the decoder output: 1 (input resolution, default), 2, 4, ...
        norm: ``"batch"``, ``"group"`` or ``None``.
        upsample: ``"nearest"`` (SMP-equivalent default), ``"bilinear"`` or ``"aligned"``.
        input_cross: Add fastai's full-resolution stage: the raw input image is concatenated
            to the stride-1 output, followed by a residual block (needs ``out_stride=1``).
    """

    def __init__(
        self,
        in_specs: Sequence[FeatureSpec],
        channels: Sequence[int] | None = None,
        out_stride: int = 1,
        norm: str | None = "batch",
        upsample: str = "nearest",
        input_cross: bool = False,
    ) -> None:
        super().__init__()
        # input_cross: fastai DynamicUnet's full-resolution stage (concatenate the raw input
        # image to the stride-1 output, then a residual block). E006 found this stage, not
        # decoder width, behind fastai_unet's lead on Sen1Floods11. The model then passes
        # [input, *features] (``wants_input``) with in_specs[0] a stride-1 input spec.
        self.wants_input = bool(input_cross)
        self._input_channels = 0
        if input_cross:
            if out_stride != 1:
                raise ValueError("input_cross needs out_stride=1 (it acts at input resolution)")
            if in_specs[0].stride != 1:
                raise ValueError("input_cross expects in_specs[0] to be the stride-1 input")
            self._input_channels = in_specs[0].channels
            in_specs = list(in_specs[1:])
        if upsample not in UPSAMPLE_MODES:
            raise ValueError(f"unknown upsample mode {upsample!r}; choose from {UPSAMPLE_MODES}")
        specs = sorted(in_specs, key=lambda s: s.stride)
        for s in specs:
            if s.stride & (s.stride - 1):
                raise ValueError(f"UNetDecoder needs power-of-two strides, got {s.stride}")
        deepest = specs[-1]
        check_power_of_two_stride("UNetDecoder", out_stride, deepest.stride)
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
        self._deepest_stride = deepest.stride
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
        out_ch = channels[-1]
        if input_cross:
            from .fastai_unet import _ResBlock

            out_ch = channels[-1] + self._input_channels
            self.cross = _ResBlock(out_ch)
        self.out_spec = FeatureSpec(out_ch, out_stride, "unet")
        init_decoder_(self)

    def _set_input_divisor(self, divisor: int) -> None:
        """Inputs arrive padded to a multiple of ``divisor``: when that covers the deepest
        stride, every level is exactly twice the next, and the upsampling needs no gather."""
        exact = divisor % self._deepest_stride == 0
        for i, block in enumerate(self.blocks):
            # the final input_cross stage lands on the (unpadded) image grid: keep the check
            last_cross = self.wants_input and i == len(self.blocks) - 1 and \
                self._skip_index[i] < 0
            block.exact_2x = exact and not last_cross

    @staticmethod
    def wants_input_for(kwargs) -> bool:
        """Whether a decoder built with ``kwargs`` needs the input image (factory hook)."""
        return bool(kwargs.get("input_cross", False))

    def forward(self, features: list[Tensor]) -> Tensor:
        if self.wants_input:
            image, features = features[0], features[1:]
        x = features[self._deepest_index]
        last = len(self.blocks) - 1
        for i, (block, idx) in enumerate(zip(self.blocks, self._skip_index)):
            # input_cross: the final stage lands on the image's grid, which need not be twice
            # the previous level when the input is not padded to the stride
            size = image.shape[-2:] if self.wants_input and i == last and idx < 0 else None
            x = block(x, features[idx] if idx >= 0 else None, size)
        if self.wants_input:
            # the input image is float32 under autocast: concatenated as is, it would make the
            # whole full-resolution stage float32 (E026)
            x = self.cross(torch.cat([x, image.to(x.dtype)], dim=1))
        return x
