"""The composed model: backbone -> decoder -> named heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.spatial import crop_to, pad_to_multiple


class DenseModel(nn.Module):
    """Dense prediction model: encoder -> (shared and/or task-specific) decoders -> named heads.

    Args:
        backbone: Module returning ``list[Tensor]`` with ``feature_specs`` and ``input_divisor``
            (a single-input backbone, or a multi-input encoder taking a dict).
        decoder: Shared decoder mapping the feature list to one tensor, or ``None`` if every
            pixel head has its own decoder.
        heads: Named heads.
        head_sources: ``{name: "decoder" | "encoder" | "task"}`` (default ``"decoder"``);
            ``"task"`` heads read ``task_decoders[name]``.
        task_decoders: Optional per-head decoders.
        conditioner: Optional :class:`~manyseg.conditioning.Conditioner` for vector
            (metadata) inputs: it splits them off the input dict and modulates the decoders
            (keyed ``"decoder"`` / ``"task_<head>"``) and heads.
        single_output: If set, ``forward`` returns only this head's tensor.
        pad_to_stride: Zero-pad bottom/right to ``backbone.input_divisor`` and crop outputs
            back, which keeps outputs pixel-aligned with inputs. If False, pixel outputs are
            resized to the input size instead.

    Pixel heads return the primary input's ``H x W``, or ``ceil(H / f) x ceil(W / f)`` for a head
    with ``scale = 1/f`` (output pixel ``(i, j)`` covers input pixels ``[i*f, (i+1)*f) x
    [j*f, (j+1)*f)``; see :mod:`manyseg.utils.resolution`).
    """

    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module | None,
        heads: Mapping[str, nn.Module],
        head_sources: Mapping[str, str] | None = None,
        single_output: str | None = None,
        pad_to_stride: bool = True,
        task_decoders: Mapping[str, nn.Module] | None = None,
        conditioner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.backbone = backbone
        self.decoder = decoder
        self.task_decoders = nn.ModuleDict(task_decoders or {})
        self.heads = nn.ModuleDict(heads)
        sources = dict(head_sources or {})
        self._head_sources = {n: sources.get(n, "decoder") for n in self.heads}
        for n, src in self._head_sources.items():
            if src == "task" and n not in self.task_decoders:
                raise ValueError(f"head {n!r} uses a task decoder but none was given")
            if src == "decoder" and decoder is None:
                raise ValueError(f"head {n!r} uses the shared decoder but decoder is None")
        self._pixel_heads = {
            n for n, h in self.heads.items() if getattr(h, "scope", "pixel") == "pixel"
        }
        if single_output is not None and single_output not in self.heads:
            raise ValueError(f"single_output {single_output!r} is not a head")
        self.single_output = single_output
        self.pad_to_stride = pad_to_stride
        self.divisor = int(getattr(backbone, "input_divisor", 1)) if pad_to_stride else 1
        self._deepest = max(
            range(len(backbone.feature_specs)), key=lambda i: backbone.feature_specs[i].stride
        )

    @property
    def feature_specs(self):
        return self.backbone.feature_specs

    @staticmethod
    def _decode(decoder: nn.Module, x: Tensor | None, feats: list[Tensor]) -> Tensor:
        return decoder([x, *feats] if getattr(decoder, "wants_input", False) else feats)

    def _decode_cond(self, key: str, decoder: nn.Module, x, feats: list[Tensor], cond) -> Tensor:
        if cond is None:
            return self._decode(decoder, x, feats)
        c = self.conditioner
        return c.output(key, self._decode(decoder, x, c.features(key, feats, cond)), cond)

    def _run_head(self, name: str, dec, feats: list[Tensor], x, h: int, w: int,
                  cond=None) -> Tensor:
        src = self._head_sources[name]
        if src == "decoder":
            inp = dec
        elif src == "encoder":
            inp = feats[self._deepest]
        else:
            inp = self._decode_cond(f"task_{name}", self.task_decoders[name], x, feats, cond)
        head = self.heads[name]
        y = head(inp)
        if name in self._pixel_heads:
            f = getattr(head, "factor", 1)  # 1 / scale; floor division keeps sizes symbolic
            hs, ws = (h + f - 1) // f, (w + f - 1) // f
            if self.pad_to_stride:
                y = crop_to(y, hs, ws)
            else:
                y = F.interpolate(y, size=(hs, ws), mode="bilinear", align_corners=False)
        return y if cond is None else self.conditioner.head(name, y, cond)

    def _pad(self, x: Tensor) -> Tensor:
        fixed = getattr(self.backbone, "fixed_size", None)
        if fixed is None:
            return pad_to_multiple(x, self.divisor)
        h, w = x.shape[-2], x.shape[-1]
        if h > fixed[0] or w > fixed[1]:
            raise ValueError(
                f"backbone has a fixed input size {fixed}; input {(h, w)} is larger. "
                "Use tiled inference or rebuild with a larger img_size."
            )
        return F.pad(x, (0, fixed[1] - w, 0, fixed[0] - h))

    def forward(self, x: Tensor | dict[str, Tensor]) -> Tensor | dict[str, Tensor]:
        cond = None
        if self.conditioner is not None:  # vector inputs -> (raster inputs, embeddings)
            x, cond = self.conditioner(x)
        primary = getattr(self.backbone, "primary", None)
        if primary is not None:  # multi-input encoder: it pads each input itself
            if not isinstance(x, Mapping):
                raise TypeError(f"this model takes a dict of inputs with keys "
                                f"{list(self.backbone.names)}")
            h, w = x[primary].shape[-2], x[primary].shape[-1]
            xin = None
            feats = self.backbone(x)
        else:
            h, w = x.shape[-2], x.shape[-1]
            xin = self._pad(x)
            feats = self.backbone(xin)
        dec = (self._decode_cond("decoder", self.decoder, xin, feats, cond)
               if self.decoder is not None else None)
        if self.single_output is not None:
            return self._run_head(self.single_output, dec, feats, xin, h, w, cond)
        return {name: self._run_head(name, dec, feats, xin, h, w, cond) for name in self.heads}
