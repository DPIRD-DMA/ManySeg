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
        # Image heads read the deepest encoder level; with late fusion the specs are every
        # input's levels concatenated, and the heads are built for the primary input's group.
        late = getattr(backbone, "mode", None) == "late"
        levels = range(*backbone.groups[0]) if late else range(len(backbone.feature_specs))
        self._deepest = max(levels, key=lambda i: backbone.feature_specs[i].stride)

    @property
    def feature_specs(self):
        return self.backbone.feature_specs

    def select_heads(self, names) -> DenseModel:
        """Keep only the heads ``names`` (in place; returns the model): their decoders stay,
        decoders, FiLM and head-bias layers only the dropped heads used are removed. For
        deployment after training with auxiliary heads, e.g. a classifier trained with a pixel
        head (E020). In eval mode the kept heads' outputs are unchanged (in training mode,
        dropout in removed modules no longer consumes random numbers, so masks differ). The
        output stays a dict. Removed modules leave the state dict: to load a pruned
        checkpoint, rebuild the model, call ``select_heads`` with the same names, then load."""
        names = list(names)
        if self.single_output is not None:
            if names != [self.single_output]:
                raise ValueError(f"single-output model: its only head is {self.single_output!r}")
            return self
        if not names:
            raise ValueError("keep at least one head")
        unknown = sorted(set(names) - set(self.heads))
        if unknown:
            raise ValueError(f"unknown heads {unknown}; the model has {list(self.heads)}")
        cond = self.conditioner
        for n in [n for n in self.heads if n not in names]:
            del self.heads[n]
            self._head_sources.pop(n)
            self._pixel_heads.discard(n)
            if n in self.task_decoders:
                del self.task_decoders[n]
            if cond is not None:
                for d, key in ((cond.film, f"task_{n}"), (cond.head_bias, n)):
                    if key in d:
                        del d[key]
        if "decoder" not in self._head_sources.values():
            self.decoder = None
            if cond is not None and "decoder" in cond.film:
                del cond.film["decoder"]
        if cond is not None and "image" in cond.film and not any(
                self._head_sources[n] == "encoder" and n not in self._pixel_heads
                for n in self.heads):
            del cond.film["image"]
        return self

    def describe(self, sample: Tensor | Mapping[str, Tensor] | None = None,
                 format: str = "text") -> str:
        """Stage-level description of the built model (``"text"``, ``"mermaid"`` or
        ``"dot"``); with ``sample``, annotated with real tensor shapes and FLOPs. See
        :func:`manyseg.describe.describe`."""
        from ..describe import describe

        return describe(self, sample, format)

    @staticmethod
    def _decode(decoder: nn.Module, x: Tensor | None, feats: list[Tensor], valid=None) -> Tensor:
        if getattr(decoder, "wants_valid", False):  # late fusion: per-input unpadded sizes
            return decoder(feats, valid=valid)
        return decoder([x, *feats] if getattr(decoder, "wants_input", False) else feats)

    def _decode_cond(self, key: str, decoder: nn.Module, x, feats: list[Tensor], cond,
                     valid=None) -> Tensor:
        if cond is None:
            return self._decode(decoder, x, feats, valid)
        c = self.conditioner
        return c.output(key, self._decode(decoder, x, c.features(key, feats, cond), valid), cond)

    def _run_head(self, name: str, dec, feats: list[Tensor], x, h: int, w: int,
                  cond=None, valid=None) -> Tensor:
        src = self._head_sources[name]
        if src == "decoder":
            inp = dec
        elif src == "encoder":
            inp = feats[self._deepest]
            if cond is not None and name not in self._pixel_heads:
                inp = self.conditioner.image(inp, cond)
        else:
            inp = self._decode_cond(f"task_{name}", self.task_decoders[name], x, feats, cond,
                                    valid)
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
                f"backbone {getattr(self.backbone, 'name', '')!r} has a fixed input size {fixed}; "
                f"input {(h, w)} is larger. Use tiled inference, or rebuild with "
                f"backbone_kwargs={{'img_size': (H, W)}} for a larger fixed size (see "
                "docs/backbone_compatibility.md)."
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
            valid = [(x[n].shape[-2], x[n].shape[-1]) for n in self.backbone.names]
            feats = self.backbone(x)
        else:
            h, w = x.shape[-2], x.shape[-1]
            xin, valid = self._pad(x), None
            feats = self.backbone(xin)
        dec = (self._decode_cond("decoder", self.decoder, xin, feats, cond, valid)
               if self.decoder is not None else None)
        if self.single_output is not None:
            return self._run_head(self.single_output, dec, feats, xin, h, w, cond, valid)
        return {name: self._run_head(name, dec, feats, xin, h, w, cond, valid)
                for name in self.heads}
