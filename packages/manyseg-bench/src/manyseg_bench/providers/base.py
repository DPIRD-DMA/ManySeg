"""Provider interface: how the benchmark builds and inspects a model from any library.

A provider turns a library-neutral :class:`ModelSpec` into an ``nn.Module`` honouring the
benchmark contract:

* input: float tensor ``(N, in_channels, H, W)``, already normalised by the benchmark;
* output: logits ``(N, num_classes, H, W)`` at input resolution.

Everything else (data, augmentation, optimisation, precision, compilation, metrics) is owned
by the benchmark and identical across providers.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """Library-neutral model request.

    Attributes:
        arch: Decoder architecture, e.g. ``"unet"``.
        encoder: Canonical encoder name, e.g. ``"resnet34"``.
        weights: Canonical weight identifier. ``"tv_in1k"`` = torchvision IMAGENET1K_V1,
            which every built-in provider can load for ResNets. ``None`` = random init.
        in_channels / num_classes: I/O channels.
        options: Provider-specific options (e.g. decoder widths). Recorded in results.
    """

    arch: str
    encoder: str
    weights: str | None = "tv_in1k"
    in_channels: int = 3
    num_classes: int = 2
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltModel:
    model: nn.Module
    encoder_params: list[nn.Parameter]
    decoder_params: list[nn.Parameter]
    weights_fingerprint: str | None
    notes: list[str] = field(default_factory=list)


class Provider(ABC):
    name: str = ""

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def build(self, spec: ModelSpec, img_size: tuple[int, int] | None = None) -> BuiltModel: ...

    def supports(self, spec: ModelSpec) -> bool:  # pragma: no cover - default
        return True


def fingerprint_module(module: nn.Module) -> str:
    """Short hash of the first conv weight: equal fingerprints => same pretrained weights."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach().float().cpu().contiguous()
            return hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16]
    return "none"


def split_params(model: nn.Module, encoder: nn.Module) -> tuple[list, list]:
    enc_ids = {id(p) for p in encoder.parameters()}
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    return enc, dec


__all__ = ["BuiltModel", "ModelSpec", "Provider", "fingerprint_module", "split_params"]
