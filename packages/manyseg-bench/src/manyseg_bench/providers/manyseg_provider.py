from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .base import BuiltModel, ModelSpec, Provider, fingerprint_module, split_params

# canonical weight id -> timm pretrained tag
_TIMM_TAGS = {"tv_in1k": "tv_in1k"}


class BandGroupModel(nn.Module):
    """Splits a stacked (B, C, H, W) tensor into named band groups, block-averaging each group
    by its integer ``factor`` (ceil, so every pixel belongs to one coarse cell; partial edge
    cells average only the pixels they cover), then runs a multi-input ManySeg model."""

    def __init__(self, model: nn.Module, groups: dict) -> None:
        super().__init__()
        self.model, self.groups = model, {n: dict(g) for n, g in groups.items()}

    @staticmethod
    def split(x: Tensor, groups: dict) -> dict[str, Tensor]:
        out = {}
        for name, g in groups.items():
            t = x[:, list(g["bands"])]
            f = int(g.get("factor", 1))
            out[name] = t if f == 1 else F.avg_pool2d(t, f, f, ceil_mode=True)
        return out

    def forward(self, x: Tensor):
        return self.model(self.split(x, self.groups))


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
        if spec.output == "image":
            # Classifier / regressor: one image-scope head pooling the deepest encoder feature
            # (no decoder is built), returned as a single (N, K) tensor.
            if "heads" in opts:
                raise ValueError("output='image' builds its own head; use 'head' for options")
            opts["heads"] = {"logits": {"type": "classification", "channels": spec.num_classes,
                                        "scope": "image", **(head or {})}}
            n_cls, head = None, head or {}
        elif head is not None:
            if n_cls is None:
                raise ValueError("'head' configures the single logits head; with 'heads', put "
                                 "the options in each head's config")
            opts["heads"] = {"logits": {"type": "segmentation", "channels": n_cls, **head}}
            n_cls = None
        groups = opts.pop("band_groups", None)
        group_inputs = opts.pop("band_group_inputs", {})
        if groups is not None:
            # One stacked tensor (e.g. 13 S2 bands resampled to 10 m) -> one input per native
            # resolution group, coarse groups block-averaged and aligned by extent (E015).
            inputs = {n: {"channels": len(g["bands"]),
                          "alignment": "same_extent" if g.get("factor", 1) > 1 else "same_grid",
                          **group_inputs.get(n, {})} for n, g in groups.items()}
            opts.setdefault("fusion", {"type": "intermediate"})
            inner = create_model(backbone=backbone, pretrained=spec.weights is not None,
                                 inputs=inputs, decoder=spec.arch, num_classes=n_cls, **opts)
            if head is not None:
                inner.single_output = "logits"
            model = BandGroupModel(inner, groups)
            enc, dec = split_params(model, inner.backbone.encoders)
            return BuiltModel(model, enc, dec, fingerprint_module(inner.backbone))
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
