from __future__ import annotations

from .base import BuiltModel, ModelSpec, Provider, fingerprint_module, split_params

# canonical weight id -> timm pretrained tag
_TIMM_TAGS = {"tv_in1k": "tv_in1k"}


class ManySegProvider(Provider):
    name = "manyseg"
    supports_dict_inputs = True

    def version(self) -> str:
        import manyseg

        return manyseg.__version__

    def build(self, spec: ModelSpec, img_size=None) -> BuiltModel:
        from manyseg import create_model

        backbone = spec.encoder
        if spec.weights is not None:
            backbone = f"{spec.encoder}.{_TIMM_TAGS.get(spec.weights, spec.weights)}"
        opts = dict(spec.options)
        n_cls = None if "heads" in opts else spec.num_classes
        # "head": options for the single logits head (e.g. {"scale": 0.5}); the model still
        # returns one tensor.
        head = opts.pop("head", None)
        if head is not None:
            if n_cls is None:
                raise ValueError("'head' configures the single logits head; with 'heads', put "
                                 "the options in each head's config")
            opts["heads"] = {"logits": {"type": "segmentation", "channels": n_cls, **head}}
            n_cls = None
        if spec.input_mode == "dict" and spec.inputs:
            inputs = {n: {"channels": c} for n, c in spec.inputs.items()}
            inputs.update({n: {"kind": "vector", **cfg} for n, cfg in (spec.vectors or {}).items()})
            for n, extra in opts.pop("input_options", {}).items():
                inputs[n].update(extra)
            opts.setdefault("fusion", {"type": "intermediate"})
            model = create_model(backbone=backbone, pretrained=spec.weights is not None,
                                 inputs=inputs, decoder=spec.arch,
                                 num_classes=n_cls, **opts)
        else:
            model = create_model(
                backbone=backbone,
                pretrained=spec.weights is not None,
                in_channels=spec.in_channels,
                decoder=spec.arch,
                num_classes=n_cls,
                **opts,
            )
        if head is not None:
            model.single_output = "logits"
        # Pretrained encoders only: fusion ops / necks count as decoder-side parameters.
        encoder = getattr(model.backbone, "encoders", None) or model.backbone
        enc, dec = split_params(model, encoder)
        return BuiltModel(model, enc, dec, fingerprint_module(model.backbone))
