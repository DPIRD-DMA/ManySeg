"""Late fusion: one decoder per input, decoder outputs fused before the heads."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from .ops import FUSION_OPS


class LateFusionDecoder(nn.Module):
    """Wraps per-input decoders built by ``make_decoder(specs)`` over the feature groups of a
    ``MultiEncoder(mode="late")``. Outputs are resized to the primary decoder's output size
    (all inputs are on the primary grid or declared same_extent) and fused with ``op``."""

    def __init__(self, in_specs: Sequence[FeatureSpec], groups: Sequence[tuple[int, int]],
                 make_decoder: Callable[[Sequence[FeatureSpec]], nn.Module],
                 op: str = "concat") -> None:
        super().__init__()
        self.groups = [tuple(g) for g in groups]
        self.decoders = nn.ModuleList(make_decoder(tuple(in_specs[a:b])) for a, b in self.groups)
        if any(getattr(d, "wants_input", False) for d in self.decoders):
            raise ValueError("late fusion does not support decoders that need the input image")
        primary = self.decoders[0].out_spec
        chans = [d.out_spec.channels for d in self.decoders]
        self.fuse = FUSION_OPS.get_or_raise(op)(chans, primary.channels)
        self.out_spec = FeatureSpec(primary.channels, primary.stride, "late_fusion")

    def forward(self, features: list[Tensor]) -> Tensor:
        outs = [d(features[a:b]) for d, (a, b) in zip(self.decoders, self.groups)]
        size = outs[0].shape[-2:]
        outs = [outs[0]] + [o if o.shape[-2:] == size else
                            F.interpolate(o, size=size, mode="bilinear", align_corners=False)
                            for o in outs[1:]]
        return self.fuse(outs)
