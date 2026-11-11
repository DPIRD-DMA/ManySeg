"""Provider interface: how the benchmark builds and inspects a model from any library.

A provider turns a library-neutral :class:`ModelSpec` into an ``nn.Module`` honouring the
benchmark contract:

* input: float tensor ``(N, in_channels, H, W)``, already normalised by the benchmark;
* output: logits ``(N, num_classes, H, W)`` at input resolution.

Everything else (data, augmentation, optimisation, precision, compilation, metrics) is owned
by the benchmark and identical across providers. Model construction itself follows each
library's own rules, which differ beyond the architecture; notably, a pretrained 3-channel
stem given ``in_channels != 3`` is adapted differently (see :func:`stem_adaptation_note`,
recorded in each run's ``model_notes``):

* SMP, timm and ManySeg (default ``channel_adapter="repeat"``): 1 channel sums the RGB
  filters; otherwise the RGB filters are cycled over the channels and scaled by ``3 / C``;
* fastai: 1 channel sums; 2 channels take the R and G filters x 1.5; more than 3 keep the RGB
  filters on the first three channels and zero the extra channels' filters.
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
        arch: Decoder architecture, e.g. ``"unet"``; ``"none"`` for image-level models
            (``output="image"``), which build no decoder (the runner records ``"none"`` for
            image-level tasks whatever was requested).
        encoder: Canonical encoder name, e.g. ``"resnet34"``.
        weights: Canonical weight identifier. ``"tv_in1k"`` = torchvision IMAGENET1K_V1,
            which every built-in provider can load for ResNets. ``None`` = random init.
        in_channels / num_classes: I/O channels.
        inputs: For multi-input datasets, ``{name: channels}``; set by the runner.
        vectors: Vector (metadata) inputs, ``{name: ManySeg vector-input config}``; set by the
            runner, used in dict mode only.
        input_mode: ``"stack"`` (inputs concatenated: one tensor, any provider) or ``"dict"``
            (named inputs; only providers whose ``supports_dict_inputs`` is True).
        options: Provider-specific options (e.g. decoder widths). Recorded in results.
        band_names: Band names of the (single) input, in order; set by the runner when
            ``weights`` is a remote-sensing tag of ``manyseg.weights``, which maps them onto the
            weights' band order.
        timeseries: Inputs that are time series ``(N, T, C, H, W)`` (dict mode); set by the
            runner. In stack mode their dates are flattened into channels instead.
        output: ``"pixel"`` (logits ``(N, num_classes, H, W)``) or ``"image"`` (one prediction
            per image, ``(N, num_classes)``: classifiers / regressors; set by the runner from
            the task).
    """

    arch: str
    encoder: str
    weights: str | None = "tv_in1k"
    in_channels: int = 3
    num_classes: int = 2
    inputs: dict[str, int] | None = None
    vectors: dict[str, dict] | None = None
    input_mode: str = "stack"
    options: dict[str, Any] = field(default_factory=dict)
    output: str = "pixel"
    timeseries: list[str] | None = None
    band_names: list[str] | None = None

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
    supports_dict_inputs: bool = False

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def build(self, spec: ModelSpec, img_size: tuple[int, int] | None = None) -> BuiltModel: ...

    def supports(self, spec: ModelSpec) -> bool:  # pragma: no cover - default
        return True


_STEM_RULES = {
    "repeat": "RGB filters cycled over the channels and scaled by 3/C",
    "fastai": "RGB filters kept on channels 0-2, the extra channels' filters zero",
}


def stem_adaptation_note(library: str, in_channels: int, pretrained: bool,
                         rule: str = "repeat") -> list[str]:
    """``model_notes`` entry describing how ``library`` adapted a pretrained 3-channel stem to
    ``in_channels`` (empty when nothing was adapted). ``rule``: ``"repeat"`` (SMP, timm,
    ManySeg's default) or ``"fastai"``; other values (ManySeg ``channel_adapter`` names) are
    recorded as given."""
    c = int(in_channels)
    if not pretrained or c == 3:
        return []
    if c == 1:
        how = "RGB filters summed"
    elif rule == "fastai" and c == 2:
        how = "R and G filters x 1.5"
    else:
        how = _STEM_RULES.get(rule, f"channel_adapter {rule!r}")
    return [f"stem adapted to {c} input channels ({library}): {how}"]


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


__all__ = ["BuiltModel", "ModelSpec", "Provider", "fingerprint_module", "split_params",
           "stem_adaptation_note"]


def check_pixel_output(spec: ModelSpec, provider: str) -> None:
    """Raise for a segmentation-only provider asked for an image-level model (``output !=
    "pixel"``): it would build pixel logits that fail late, in the loss."""
    if spec.output != "pixel":
        raise ValueError(f"the {provider} provider builds segmentation models only (pixel "
                         f"output), got output={spec.output!r}; image-level tasks "
                         "(classification / regression) need the timm or manyseg provider")
