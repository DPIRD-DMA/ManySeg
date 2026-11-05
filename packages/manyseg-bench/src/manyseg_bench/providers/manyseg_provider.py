from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import BuiltModel, ModelSpec, Provider, fingerprint_module, split_params

# canonical weight id -> timm pretrained tag
_TIMM_TAGS = {"tv_in1k": "tv_in1k"}


def _decoder(spec: ModelSpec) -> str:
    # arch "none" names image-level models, which build no decoder: ManySeg still checks the
    # decoder name, so pass its default
    return "unet" if spec.arch == "none" else spec.arch


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


class WeightsInput(nn.Module):
    """Maps a dataset's bands onto remote-sensing weights' band order (``index[j]`` = dataset
    band for weights band ``j``, -1 = missing: filled with 0, the pretraining mean after
    normalisation) and clips to [0, 1] where the pretraining did."""

    def __init__(self, index: list[int], clip: bool) -> None:
        super().__init__()
        self.register_buffer("index", torch.tensor([max(i, 0) for i in index]), persistent=False)
        self.register_buffer("present", torch.tensor([i >= 0 for i in index]), persistent=False)
        self.clip = clip

    def forward(self, x: Tensor) -> Tensor:
        y = x.index_select(1, self.index) * self.present.view(1, -1, 1, 1).to(x.dtype)
        return y.clamp(0, 1) if self.clip else y


class WeightsInputModel(nn.Module):
    def __init__(self, model: nn.Module, weights_input: WeightsInput) -> None:
        super().__init__()
        self.model, self.weights_input = model, weights_input

    def forward(self, x):
        return self.model(self.weights_input(x))


def registry_weights(spec):
    """The ``manyseg.weights`` entry for ``spec`` if its ``weights`` is a remote-sensing tag."""
    if spec.weights is None:
        return None
    try:
        from manyseg.weights import WEIGHTS
    except ImportError:  # pragma: no cover
        return None
    return WEIGHTS.get(f"{spec.encoder}.{spec.weights}")


class ManySegProvider(Provider):
    name = "manyseg"
    supports_dict_inputs = True

    def version(self) -> str:
        import manyseg

        return manyseg.__version__

    def build(self, spec: ModelSpec, img_size=None) -> BuiltModel:
        w = registry_weights(spec)
        if w is None:
            backbone = spec.encoder
            if spec.weights is not None:
                backbone = f"{spec.encoder}.{_TIMM_TAGS.get(spec.weights, spec.weights)}"
            return self._build(spec, backbone, spec.weights is not None, img_size)
        # Remote-sensing weights (manyseg.weights): the encoder takes the weights' bands in
        # their order; the dataset's bands are mapped onto them in front of the model.
        if not spec.band_names:
            raise ValueError(f"weights {w.key} need the dataset's band names "
                             "(ModelSpec.band_names) to map bands")
        if spec.input_mode == "dict" or spec.options.get("band_groups"):
            raise NotImplementedError("remote-sensing weights: single stacked input only")
        index = [spec.band_names.index(b) if b in spec.band_names else -1 for b in w.bands]
        inner = self._build(ModelSpec(**{**spec.to_dict(), "in_channels": w.in_channels}),
                            spec.encoder, w.tag, img_size)
        model = WeightsInputModel(inner.model, WeightsInput(index, w.clip))
        missing = [b for b, i in zip(w.bands, index) if i < 0]
        notes = [f"bands filled with the pretraining mean: {', '.join(missing) or '-'}"]
        return BuiltModel(model, inner.encoder_params, inner.decoder_params, w.key, notes)

    def _build(self, spec: ModelSpec, backbone: str, pretrained, img_size=None) -> BuiltModel:
        from manyseg import create_model

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
            opts["heads"] = {"logits": {"type": "classification", "out_channels": spec.num_classes,
                                        "scope": "image", **(head or {})}}
            n_cls, head = None, head or {}
        elif head is not None:
            if n_cls is None:
                raise ValueError("'head' configures the single logits head; with 'heads', put "
                                 "the options in each head's config")
            opts["heads"] = {"logits": {"type": "segmentation", "out_channels": n_cls, **head}}
            n_cls = None
        groups = opts.pop("band_groups", None)
        group_inputs = opts.pop("band_group_inputs", {})
        if groups is not None:
            # One stacked tensor (e.g. 13 S2 bands resampled to 10 m) -> one input per native
            # resolution group, coarse groups block-averaged and aligned by extent (E015).
            inputs = {n: {"in_channels": len(g["bands"]),
                          "alignment": "same_extent" if g.get("factor", 1) > 1 else "same_grid",
                          **group_inputs.get(n, {})} for n, g in groups.items()}
            # A group sharing an encoder with another band count reaches it through a projection
            # (E015-E017, E020); create_model wants that stated.
            for cfg in inputs.values():
                target = str(cfg.get("backbone", "")).removeprefix("shared:")
                if target in inputs and cfg["in_channels"] != inputs[target]["in_channels"]:
                    cfg.setdefault("channel_adapter", "projection")
            opts.setdefault("fusion", {"type": "intermediate"})
            inner = create_model(backbone=backbone, pretrained=pretrained,
                                 inputs=inputs, decoder=_decoder(spec), num_classes=n_cls, **opts)
            if head is not None:
                inner.single_output = "logits"
            model = BandGroupModel(inner, groups)
            enc, dec = split_params(model, inner.backbone.encoders)
            return BuiltModel(model, enc, dec, fingerprint_module(inner.backbone))
        if spec.input_mode == "dict" and spec.inputs:
            inputs = {n: {"in_channels": c} for n, c in spec.inputs.items()}
            for n in spec.timeseries or []:
                inputs[n]["kind"] = "timeseries"
            inputs.update({n: {"kind": "vector", **cfg} for n, cfg in (spec.vectors or {}).items()})
            for n, extra in opts.pop("input_options", {}).items():
                inputs[n].update(extra)
            opts.setdefault("fusion", {"type": "intermediate"})
            model = create_model(backbone=backbone, pretrained=pretrained,
                                 inputs=inputs, decoder=_decoder(spec),
                                 num_classes=n_cls, **opts)
        else:
            model = create_model(
                backbone=backbone,
                pretrained=pretrained,
                in_channels=spec.in_channels,
                decoder=_decoder(spec),
                num_classes=n_cls,
                **opts,
            )
        if head is not None:
            model.single_output = "logits"
        # Pretrained encoders only: fusion ops / necks count as decoder-side parameters.
        encoder = getattr(model.backbone, "encoders", None) or model.backbone
        enc, dec = split_params(model, encoder)
        return BuiltModel(model, enc, dec, fingerprint_module(model.backbone))

