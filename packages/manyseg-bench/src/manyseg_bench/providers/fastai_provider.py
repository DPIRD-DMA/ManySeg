from __future__ import annotations

from torch import nn

from .base import (
    BuiltModel,
    ModelSpec,
    Provider,
    fingerprint_module,
    split_params,
    stem_adaptation_note,
)

_TV_WEIGHTS = {"tv_in1k": "IMAGENET1K_V1"}


class FastaiProvider(Provider):
    """fastai ``DynamicUnet`` on a torchvision body, built exactly as ``unet_learner`` does
    (``create_body`` + ``DynamicUnet``), without fastai's Learner or training loop."""

    name = "fastai"

    def version(self) -> str:
        import fastai

        return fastai.__version__

    def supports(self, spec: ModelSpec) -> bool:
        return spec.arch == "unet" and spec.encoder.startswith("resnet")

    def build(self, spec: ModelSpec, img_size=None) -> BuiltModel:
        import torchvision
        from fastai.vision.learner import create_body
        from fastai.vision.models.unet import DynamicUnet

        if not self.supports(spec):
            raise ValueError(f"fastai provider supports resnet U-Nets only, got {spec}")
        if img_size is None:
            raise ValueError("fastai DynamicUnet needs img_size at construction")
        weights = None if spec.weights is None else _TV_WEIGHTS[spec.weights]
        arch = getattr(torchvision.models, spec.encoder)
        body = create_body(arch(weights=weights), n_in=spec.in_channels,
                           pretrained=weights is not None)
        model = DynamicUnet(body, spec.num_classes, img_size, **spec.options)
        encoder = model[0]
        enc, dec = split_params(model, encoder)
        notes = stem_adaptation_note("fastai", spec.in_channels, weights is not None,
                                     rule="fastai")
        return BuiltModel(_Plain(model), enc, dec, fingerprint_module(encoder), notes)


class _Plain(nn.Module):
    """Unwrap fastai's TensorBase subclass so outputs are plain tensors."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        return y.as_subclass(type(x)) if type(y) is not type(x) else y
