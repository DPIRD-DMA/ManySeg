"""Static description of multi-scale feature maps passed between components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one feature map. Tensors themselves are always NCHW.

    Attributes:
        channels: Number of channels ``C``.
        stride: Spatial reduction relative to the (padded) model input.
        name: Originating module name, for debugging only.
    """

    channels: int
    stride: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
