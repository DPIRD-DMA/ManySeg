"""Injecting a metadata embedding into a dense model.

Modes (chosen per vector input, ``{"conditioning": ...}``):

* ``film``: feature-wise affine modulation ``x * (1 + gamma(e)) + beta(e)`` of (a) every
  encoder feature level a decoder reads, just before it enters the decoder, and (b) the
  decoder output. Applied around each decoder (shared, per-head task decoders, late-fusion
  decoder) without touching decoder code, so it works with every registered decoder.
* ``head``: the embedding enters the heads. For an image head ``Linear([pool(f), e])``
  equals ``Linear(pool(f)) + Linear_e(e)``, so it is implemented as a per-sample additive
  term on the head output; for pixel heads (``pixel=True``) the same term is broadcast over
  space (a 1x1 kernel on the spatially broadcast embedding).
* ``channels``: the encoded fields are broadcast as extra input channels of one raster input
  (early fusion through that input's channel adapter).

``film`` and ``head`` projections are zero-initialised, so a fresh conditioned model computes
exactly what the unconditioned model with the same weights computes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from ..features import FeatureSpec
from .encoder import MetadataEncoder


class FiLM(nn.Module):
    """``x * (1 + gamma(e)) + beta(e)`` per channel; zero-initialised, i.e. exactly the
    identity at init."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        gamma, beta = self.proj(e).to(x.dtype).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


def used_levels(decoder: nn.Module, specs: Sequence[FeatureSpec],
                input_channels: int | None = None) -> list[int]:
    """Indices of the feature levels ``decoder`` actually reads, found once at construction by
    a gradient probe on small random features (decoders such as DeepLabV3+ or FPN skip some
    levels; FiLM on an unread level would be a parameter without gradient)."""
    size = 2 * max(s.stride for s in specs)
    p = next(iter(decoder.parameters()), None)
    dev = p.device if p is not None else torch.device("cpu")
    feats = [torch.randn(1, s.channels, -(-size // s.stride), -(-size // s.stride), device=dev,
                         requires_grad=True) for s in specs]
    inp = feats
    if getattr(decoder, "wants_input", False):
        inp = [torch.randn(1, input_channels, size, size, device=dev), *feats]
    was_training = decoder.training
    decoder.eval()
    try:
        with torch.enable_grad():
            grads = torch.autograd.grad(decoder(inp).sum(), feats, allow_unused=True)
    finally:
        decoder.train(was_training)
    return [i for i, g in enumerate(grads) if g is not None]


class DecoderFiLM(nn.Module):
    """FiLM on the feature levels a decoder reads (``"features"``) and on its output
    (``"output"``)."""

    def __init__(self, cond_dim: int, specs: Sequence[FeatureSpec], levels: Sequence[int],
                 out_spec: FeatureSpec, on: Sequence[str] = ("features", "output")) -> None:
        super().__init__()
        unknown = set(on) - {"features", "output"}
        if unknown or not on:
            raise ValueError(f"film 'on' must be a non-empty subset of ('features', 'output'), "
                             f"got {list(on)}")
        self.levels = tuple(levels) if "features" in on else ()
        self.features = nn.ModuleDict({str(i): FiLM(cond_dim, specs[i].channels)
                                       for i in self.levels})
        self.output = FiLM(cond_dim, out_spec.channels) if "output" in on else None

    def apply_features(self, feats: list[Tensor], e: Tensor) -> list[Tensor]:
        if not self.levels:
            return feats
        feats = list(feats)
        for i in self.levels:
            feats[i] = self.features[str(i)](feats[i], e)
        return feats

    def apply_output(self, y: Tensor, e: Tensor) -> Tensor:
        return y if self.output is None else self.output(y, e)


class Conditioner(nn.Module):
    """Splits the vector inputs off the model's input dict, encodes them and applies them.

    Used by :class:`~manyseg.models.DenseModel`: ``forward(x)`` returns ``(raster_x, cond)``
    where ``raster_x`` is what the raster model takes (the raster dict, or the single raster
    tensor if the raster model is single-input) and ``cond = (film_embedding, head_embedding)``
    (either may be ``None``). ``features`` / ``output`` / ``head`` are then called by the model
    around each decoder (keyed ``"decoder"`` or ``"task_<head>"``) and on each head output.

    Args:
        encoders: ``{vector input name: MetadataEncoder}``.
        modes: ``{vector input name: "film" | "head" | "channels"}``.
        rasters: Raster input names; ``single`` names the raster if the raster model takes a
            plain tensor.
        channel_targets: ``{vector input (channels mode): raster input}``.
        film: ``{decoder key: DecoderFiLM}``.
        head_bias: ``{head name: Linear(head_dim, out_channels)}`` (zero-initialised).
    """

    def __init__(self, encoders: Mapping[str, MetadataEncoder], modes: Mapping[str, str],
                 rasters: Sequence[str], single: str | None = None,
                 channel_targets: Mapping[str, str] | None = None,
                 film: Mapping[str, DecoderFiLM] | None = None,
                 head_bias: Mapping[str, nn.Module] | None = None) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.vectors = tuple(encoders)
        self.modes = dict(modes)
        self.rasters = tuple(rasters)
        self.single = single
        self.channel_targets = dict(channel_targets or {})
        self.film = nn.ModuleDict(film or {})
        self.head_bias = nn.ModuleDict(head_bias or {})
        self._by_mode = {m: tuple(n for n in self.vectors if self.modes[n] == m)
                         for m in ("film", "head", "channels")}

    def _embed(self, x: Mapping[str, Tensor], names: Sequence[str]) -> Tensor | None:
        if not names:
            return None
        es = [self.encoders[n](x[n], x.get(f"{n}_mask")) for n in names]
        return es[0] if len(es) == 1 else torch.cat(es, dim=1)

    def forward(self, x: Mapping[str, Tensor]):
        if not isinstance(x, Mapping):
            raise TypeError(f"this model takes a dict of inputs with keys "
                            f"{list(self.rasters) + list(self.vectors)}")
        for n in self.vectors:
            if n not in x:
                raise KeyError(f"missing vector input {n!r}; expected keys "
                               f"{list(self.rasters) + list(self.vectors)}")
        rasters = {n: x[n] for n in self.rasters}
        for n in self._by_mode["channels"]:
            t = self.channel_targets[n]
            r = rasters[t]
            e = self.encoders[n](x[n], x.get(f"{n}_mask")).to(r.dtype)
            e = e[:, :, None, None].expand(-1, -1, r.shape[-2], r.shape[-1])
            rasters[t] = torch.cat([r, e], dim=1)
        cond = (self._embed(x, self._by_mode["film"]), self._embed(x, self._by_mode["head"]))
        return (rasters[self.single] if self.single is not None else rasters), cond

    def features(self, key: str, feats: list[Tensor], cond) -> list[Tensor]:
        if cond[0] is None or key not in self.film:
            return feats
        return self.film[key].apply_features(feats, cond[0])

    def output(self, key: str, y: Tensor, cond) -> Tensor:
        if cond[0] is None or key not in self.film:
            return y
        return self.film[key].apply_output(y, cond[0])

    def head(self, name: str, y: Tensor, cond) -> Tensor:
        if cond[1] is None or name not in self.head_bias:
            return y
        b = self.head_bias[name](cond[1]).to(y.dtype)
        return y + (b if y.ndim == 2 else b[:, :, None, None])
