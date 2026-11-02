"""Decoder interface.

A decoder is an ``nn.Module`` constructed as ``Decoder(in_specs, **options)`` that maps the
encoder's ``list[Tensor]`` (ordered as ``in_specs``; levels the decoder does not read may be
``None``) to a single tensor described by
``self.out_spec``. Decoders that need the (padded, normalised) input image set the class
attribute ``wants_input = True``; they then receive ``[input, *features]`` and
``in_specs[0]`` is a stride-1 spec for the input.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..features import FeatureSpec
from ..registry import Registry

DECODERS = Registry("decoder")
"""Decoder classes by ``decoder=`` name; ``DECODERS.register("name")`` adds one."""


def pyramid_levels(specs: Sequence[FeatureSpec], min_stride: int = 4,
                   max_levels: int = 4) -> list[int]:
    """Indices of the (at most ``max_levels``) deepest features with stride >= min_stride,
    ordered shallow -> deep."""
    idx = sorted((i for i, s in enumerate(specs) if s.stride >= min_stride),
                 key=lambda i: specs[i].stride)
    if not idx:
        raise ValueError(f"no features with stride >= {min_stride}")
    return idx[-max_levels:]


def check_out_stride(name: str, native: int, out_stride: int | None) -> None:
    """Fixed-stride decoders (``fpn``, ``deeplabv3plus``, ``upernet``, ``segformer``) accept
    ``out_stride`` only to state their stride; any other value is an error, never a silent
    resize."""
    if out_stride is not None and out_stride != native:
        raise ValueError(
            f"{name} outputs stride {native} for these features; out_stride={out_stride} is not "
            "available. Use 'unet', 'unetplusplus' or 'fastai_unet' for other decoder strides, "
            "or a head 'scale' for coarser outputs.")


def check_power_of_two_stride(name: str, out_stride: int, deepest: int) -> None:
    if not (isinstance(out_stride, int) and 1 <= out_stride < deepest
            and out_stride & (out_stride - 1) == 0):
        raise ValueError(f"{name}: out_stride must be a power of two below the deepest feature "
                         f"stride {deepest}, got out_stride={out_stride}")


def check_input_spec(specs: Sequence[FeatureSpec]) -> None:
    if specs[0].stride != 1:
        raise ValueError("decoder wants the input image: in_specs[0] must be a stride-1 spec")
