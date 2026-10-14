"""Decoder interface.

A decoder is an ``nn.Module`` constructed as ``Decoder(in_specs, **options)`` that maps the
encoder's ``list[Tensor]`` (ordered as ``in_specs``) to a single tensor described by
``self.out_spec``. Decoders that need the (padded, normalised) input image set the class
attribute ``wants_input = True``; they then receive ``[input, *features]`` and
``in_specs[0]`` is a stride-1 spec for the input.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..features import FeatureSpec
from ..registry import Registry

DECODERS = Registry("decoder")


def pyramid_levels(specs: Sequence[FeatureSpec], min_stride: int = 4,
                   max_levels: int = 4) -> list[int]:
    """Indices of the (at most ``max_levels``) deepest features with stride >= min_stride,
    ordered shallow -> deep."""
    idx = sorted((i for i, s in enumerate(specs) if s.stride >= min_stride),
                 key=lambda i: specs[i].stride)
    if not idx:
        raise ValueError(f"no features with stride >= {min_stride}")
    return idx[-max_levels:]


def check_input_spec(specs: Sequence[FeatureSpec]) -> None:
    if specs[0].stride != 1:
        raise ValueError("decoder wants the input image: in_specs[0] must be a stride-1 spec")
