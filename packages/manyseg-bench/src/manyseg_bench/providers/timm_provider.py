"""Plain timm classifiers: the library-neutral baseline for image-level tasks.

``timm.create_model(encoder, num_classes=K, in_chans=C)``: timm's own pooling and head (for
some families a pre-logits norm, which counts as encoder). Only the final classifier
(``get_classifier()``) is "decoder", so a fastai-style frozen epoch trains just the classifier,
as for ManySeg's image heads.
"""

from __future__ import annotations

from .base import BuiltModel, ModelSpec, Provider, fingerprint_module, stem_adaptation_note

_TIMM_TAGS = {"tv_in1k": "tv_in1k"}


class TimmProvider(Provider):
    name = "timm"

    def version(self) -> str:
        import timm

        return timm.__version__

    def build(self, spec: ModelSpec, img_size=None) -> BuiltModel:
        import timm

        if spec.output != "image":
            raise ValueError("the timm provider builds image-level models only "
                             "(classification / regression); output must be 'image'")
        if spec.options:
            raise ValueError(f"the timm provider takes no options, got {sorted(spec.options)}")
        name = spec.encoder
        if spec.weights is not None:
            name = f"{spec.encoder}.{_TIMM_TAGS.get(spec.weights, spec.weights)}"
        model = timm.create_model(name, pretrained=spec.weights is not None,
                                  num_classes=spec.num_classes, in_chans=spec.in_channels)
        head = {id(p) for p in model.get_classifier().parameters()}
        enc = [p for p in model.parameters() if id(p) not in head]
        dec = [p for p in model.parameters() if id(p) in head]
        return BuiltModel(model, enc, dec, fingerprint_module(model),
                          stem_adaptation_note("timm", spec.in_channels,
                                               spec.weights is not None))
