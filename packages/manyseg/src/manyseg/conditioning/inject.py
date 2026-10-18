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
    """Indices of the feature levels ``decoder`` actually reads (decoders such as DeepLabV3+ or
    FPN skip some levels; FiLM on an unread level would be a parameter without gradient).

    A decoder may declare them as ``read_levels`` (indices into the encoder feature list, not
    counting the input image of ``wants_input`` decoders); otherwise they are found once at
    construction by a gradient probe: the union over a square and an odd, non-square input
    size. Read sets that depend on the input size in other ways are not supported.

    The probe is side-effect free: inputs come from a private generator (the global RNG is
    untouched), and every submodule's ``training`` flag and every buffer are restored.
    """
    declared = getattr(decoder, "read_levels", None)
    if declared is not None:
        levels = sorted({int(i) for i in declared})
        if not all(0 <= i < len(specs) for i in levels):
            raise ValueError(f"read_levels {list(declared)} out of range for {len(specs)} "
                             "feature levels")
        return levels
    m = max(s.stride for s in specs)
    p = next(iter(decoder.parameters()), None)
    dev = p.device if p is not None else torch.device("cpu")
    gen = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(dev)

    modes = [(mod, mod.training) for mod in decoder.modules()]
    buffers = [(b, b.detach().clone()) for b in decoder.buffers()]
    read: set[int] = set()
    decoder.eval()
    try:
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []), \
                torch.enable_grad():
            for h, w in ((2 * m, 2 * m), (2 * m + 1, 3 * m)):
                feats = [randn(1, s.channels, -(-h // s.stride), -(-w // s.stride))
                         .requires_grad_() for s in specs]
                inp = feats
                if getattr(decoder, "wants_input", False):
                    inp = [randn(1, input_channels, h, w), *feats]
                grads = torch.autograd.grad(decoder(inp).sum(), feats, allow_unused=True)
                read |= {i for i, g in enumerate(grads) if g is not None}
    finally:
        for mod, training in modes:
            mod.training = training
        with torch.no_grad():
            for b, saved in buffers:
                b.copy_(saved)
    return sorted(read)


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
        film: ``{decoder key: DecoderFiLM}``, plus ``"image"``: a :class:`FiLM` on the deepest
            encoder feature when image-scope heads pool it.
        head_bias: ``{head name: Linear(head_dim, out_channels)}`` (zero-initialised).
    """

    def __init__(self, encoders: Mapping[str, MetadataEncoder], modes: Mapping[str, str],
                 rasters: Sequence[str], single: str | None = None,
                 channel_targets: Mapping[str, str] | None = None,
                 film: Mapping[str, DecoderFiLM] | None = None,
                 head_bias: Mapping[str, nn.Module] | None = None,
                 temporal: Sequence[str] = ()) -> None:
        super().__init__()
        # time-series rasters: their "<name>_mask" / "<name>_dates" entries pass through
        self.temporal_extras = tuple(f"{n}_{s}" for n in temporal for s in ("mask", "dates"))
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
        expected = list(self.rasters) + list(self.vectors)
        for n in expected:
            if n not in x:
                kind = "vector" if n in self.vectors else "raster"
                raise KeyError(f"missing {kind} input {n!r}; expected keys {expected}")
        optional = [f"{n}_mask" for n in self.vectors] + list(getattr(self, "temporal_extras", ()))
        unknown = [k for k in x if k not in expected and k not in optional]
        if unknown:  # a misspelt "<name>_mask" would otherwise be ignored silently
            raise ValueError(f"unknown input keys {unknown}; this model takes {expected} and "
                             f"optionally {optional}")
        rasters = {n: x[n] for n in self.rasters}
        rasters.update({k: x[k] for k in getattr(self, "temporal_extras", ()) if k in x})
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

    def image(self, feat: Tensor, cond) -> Tensor:
        """FiLM on the encoder feature that image-scope heads pool."""
        if cond[0] is None or "image" not in self.film:
            return feat
        return self.film["image"](feat, cond[0])

    def head(self, name: str, y: Tensor, cond) -> Tensor:
        if cond[1] is None or name not in self.head_bias:
            return y
        b = self.head_bias[name](cond[1]).to(y.dtype)
        return y + (b if y.ndim == 2 else b[:, :, None, None])
