"""Late fusion: one decoder per input, decoder outputs fused before the heads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from torch import Tensor, nn

from ..features import FeatureSpec
from ..utils.spatial import resize_bilinear
from .encoders import _ceil_div, _resize_valid
from .ops import FUSION_OPS


class LateFusionDecoder(nn.Module):
    """Wraps per-input decoders built by ``make_decoder(specs)`` over the feature groups of a
    ``MultiEncoder(mode="late")``. Outputs are resized to the primary decoder's output size
    (all inputs are on the primary grid or declared same_extent) and fused with ``op``."""

    def __init__(self, in_specs: Sequence[FeatureSpec], groups: Sequence[tuple[int, int]],
                 make_decoder: Callable[[Sequence[FeatureSpec]], nn.Module],
                 op: str = "concat", alignment: Sequence[str] | None = None,
                 op_kwargs: Mapping | None = None) -> None:
        super().__init__()
        self.groups = [tuple(g) for g in groups]
        # Per input (primary first); same_extent outputs are resampled by valid region.
        self.alignment = tuple(alignment or ("same_grid",) * len(self.groups))
        self.decoders = nn.ModuleList(make_decoder(tuple(in_specs[a:b])) for a, b in self.groups)
        if any(getattr(d, "wants_input", False) for d in self.decoders):
            raise ValueError("late fusion does not support decoders that need the input image")
        primary = self.decoders[0].out_spec
        chans = [d.out_spec.channels for d in self.decoders]
        self.fuse = FUSION_OPS.get_or_raise(op)(chans, primary.channels,
                                                **dict(op_kwargs or {}))
        self.out_spec = FeatureSpec(primary.channels, primary.stride, "late_fusion")

    wants_valid = True  # DenseModel passes each input's unpadded (H, W)

    def forward(self, features: list[Tensor], valid: Sequence[tuple] | None = None) -> Tensor:
        """``valid``: each input's unpadded (H, W), primary first. With it, a same_extent
        input's *valid* output region is resampled onto the primary's valid region (as
        intermediate fusion does); without it, outputs of a different size are resized whole,
        which is only right when no input was padded."""
        outs = [d(features[a:b]) for d, (a, b) in zip(self.decoders, self.groups)]
        size = outs[0].shape[-2:]
        if valid is None:
            outs = [outs[0]] + [o if o.shape[-2:] == size else resize_bilinear(o, size)
                                for o in outs[1:]]
            return self.fuse(outs)
        s0 = self.decoders[0].out_spec.stride
        target = (_ceil_div(valid[0][0], s0), _ceil_div(valid[0][1], s0))
        fused = [outs[0]]
        for d, o, (h, w), a in zip(self.decoders[1:], outs[1:], valid[1:], self.alignment[1:]):
            if a == "same_extent":
                s = d.out_spec.stride
                o = _resize_valid(o, (_ceil_div(h, s), _ceil_div(w, s)), target, size)
            elif o.shape[-2:] != size:  # same_grid with a different encoder stride layout
                o = resize_bilinear(o, size)
            fused.append(o)
        return self.fuse(fused)
