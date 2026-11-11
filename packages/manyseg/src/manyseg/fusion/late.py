"""Late fusion: one decoder per input, decoder outputs fused before the heads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from torch import Tensor, nn

from ..features import FeatureSpec
from ..utils.spatial import crop_to, resize_bilinear
from .encoders import _ceil_div, _resize_valid
from .ops import FUSION_OPS


def _align_same_grid(o: Tensor, stride: int, size, s0: int) -> Tensor:
    """Bring a same_grid input's decoder output ``o`` (at ``stride``) onto the primary's output
    grid ``size`` (at ``s0``). Both inputs share the top-left origin and the valid region, but
    their padded extents differ when one was padded up to a fixed-size backbone's input size:
    crop (or zero-pad) ``o`` to the primary's padded extent first, as intermediate fusion
    does, and resize only for a genuine stride difference."""
    eh, ew = size[0] * s0, size[1] * s0  # primary's padded extent in input pixels
    if (o.shape[-2] * stride, o.shape[-1] * stride) != (eh, ew):
        o = crop_to(o, _ceil_div(eh, stride), _ceil_div(ew, stride))
    return o if o.shape[-2:] == size else resize_bilinear(o, size)


class LateFusionDecoder(nn.Module):
    """Wraps per-input decoders built by ``make_decoder(specs)`` over the feature groups of a
    ``MultiEncoder(mode="late")``. Outputs are brought onto the primary decoder's output grid
    and fused with ``op``: a same_extent input's valid region is resampled; a same_grid input's
    output is cropped (or zero-padded) when only the padding differs (an input padded up to a
    fixed-size backbone's size), and resized only for a different output stride."""

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
            else:
                o = _align_same_grid(o, d.out_spec.stride, size, s0)
            fused.append(o)
        return self.fuse(fused)
