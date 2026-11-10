from __future__ import annotations

from .base import (
    BuiltModel,
    ModelSpec,
    Provider,
    fingerprint_module,
    split_params,
    stem_adaptation_note,
)

_ARCHS = {"unet": "Unet", "unetplusplus": "UnetPlusPlus", "fpn": "FPN",
          "deeplabv3plus": "DeepLabV3Plus", "segformer": "Segformer", "upernet": "UPerNet"}
# canonical weight id -> (smp encoder name transform, smp encoder_weights)
_WEIGHTS = {"tv_in1k": "imagenet"}


class SMPProvider(Provider):
    """segmentation-models-pytorch. ``tv_in1k`` ResNets use SMP's native ``resnetXX`` encoders,
    whose ``imagenet`` weights are the torchvision V1 weights (verified by fingerprint)."""

    name = "smp"

    def version(self) -> str:
        import segmentation_models_pytorch as smp

        return smp.__version__

    def build(self, spec: ModelSpec, img_size=None) -> BuiltModel:
        import segmentation_models_pytorch as smp

        cls = getattr(smp, _ARCHS[spec.arch])
        weights = None if spec.weights is None else _WEIGHTS.get(spec.weights, spec.weights)
        model = cls(
            encoder_name=spec.encoder,
            encoder_weights=weights,
            in_channels=spec.in_channels,
            classes=spec.num_classes,
            **spec.options,
        )
        enc, dec = split_params(model, model.encoder)
        return BuiltModel(model, enc, dec, fingerprint_module(model.encoder),
                          stem_adaptation_note("smp", spec.in_channels, weights is not None))
