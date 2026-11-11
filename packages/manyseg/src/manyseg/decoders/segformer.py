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
    """Per-level linear (1x1) to ``channels``, upsample to the shallowest level, concat, fuse
    (computed per level, see ``forward``).
    Output: ``channels`` at the stride of the shallowest level it reads (the at most 4
    deepest levels with stride >= 4): 4 on a standard stride 4..32 pyramid, coarser when the
    encoder has no stride-4 level or more than 4 such levels (``out_stride`` may only restate
    it)."""

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

    @staticmethod
    def _can_split(conv: nn.Module) -> bool:
        """The per-level path reads the conv's weight directly: only for a plain, unhooked
        Conv2d with an ordinary weight (not spectral norm, observers or quantised weights)."""
        plain_weight = torch.compiler.is_compiling() or type(conv.weight) is nn.Parameter
        return (type(conv) is nn.Conv2d and conv.bias is None and plain_weight  # export: fake
                and not conv._forward_hooks and not conv._forward_pre_hooks
                and not nn.modules.module._global_forward_hooks
                and not nn.modules.module._global_forward_pre_hooks)

    def forward(self, features: list[Tensor]) -> Tensor:
        # The fuse is a bias-free 1x1 conv over the concatenated upsampled levels. Bilinear
        # resizing is linear with weights that sum to one, so it commutes with a 1x1 conv: each
        # level gets its slice of the fuse weight at its own resolution, then is upsampled and
        # summed. Same parameters and values (up to rounding) as conv(cat(upsampled)), without
        # the 4C-channel concat at the finest stride (-26 to -38% training memory).
        size = features[self.levels[0]].shape[-2:]
        conv, norm, act = self.fuse  # ConvNormAct: bias-free conv, BatchNorm, ReLU
        if not self._can_split(conv):
            outs = [lin(features[i]) for i, lin in zip(self.levels, self.linear)]
            outs = [y if y.shape[-2:] == size else
                    F.interpolate(y, size=size, mode="bilinear", align_corners=False)
                    for y in outs]
            return self.dropout(self.fuse(torch.cat(outs, dim=1)))
        c = conv.weight.shape[1] // len(self.levels)
        acc, dtype = None, None
        for j, (i, lin) in enumerate(zip(self.levels, self.linear)):
            z = F.conv2d(lin(features[i]), conv.weight[:, j * c:(j + 1) * c])
            dtype = z.dtype
            if z.shape[-2:] != size:
                z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
            # sum in (at least) float32 and round once, as the single conv's accumulator does
            z = z.to(torch.promote_types(z.dtype, torch.float32))
            acc = z if acc is None else acc + z
        return self.dropout(act(norm(acc.to(dtype))))
