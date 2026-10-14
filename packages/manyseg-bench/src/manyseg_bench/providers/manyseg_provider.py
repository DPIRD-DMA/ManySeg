from __future__ import annotations

from .base import BuiltModel, ModelSpec, Provider, fingerprint_module, split_params

# canonical weight id -> timm pretrained tag
_TIMM_TAGS = {"tv_in1k": "tv_in1k"}


class ManySegProvider(Provider):
    name = "manyseg"

    def version(self) -> str:
        import manyseg

        return manyseg.__version__

    def build(self, spec: ModelSpec, img_size=None) -> BuiltModel:
        from manyseg import create_model

        backbone = spec.encoder
        if spec.weights is not None:
            backbone = f"{spec.encoder}.{_TIMM_TAGS.get(spec.weights, spec.weights)}"
        opts = dict(spec.options)
        model = create_model(
            backbone=backbone,
            pretrained=spec.weights is not None,
            in_channels=spec.in_channels,
            decoder=spec.arch,
            num_classes=spec.num_classes,
            **opts,
        )
        enc, dec = split_params(model, model.backbone)
        return BuiltModel(model, enc, dec, fingerprint_module(model.backbone))
