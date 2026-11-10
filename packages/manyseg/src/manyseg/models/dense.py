"""The composed model: backbone -> decoder -> named heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch.nn.functional as F
from torch import Tensor, nn

from ..decoders.base import deepest_index
from ..utils.spatial import crop_to, pad_to_multiple

if TYPE_CHECKING:
    from ..features import FeatureSpec


class DenseModel(nn.Module):
    """Dense prediction model: encoder -> (shared and/or task-specific) decoders -> named heads.

    Args:
        backbone: Module returning ``list[Tensor | None]`` (``None`` for levels nothing
            reads, e.g. a pruned ViT neck level) with ``feature_specs`` and ``input_divisor``
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
        head_inputs: ``{head: input name}`` for per-input heads, whose task decoder reads that
            input's own (unfused) features.
        pad_to_stride: Zero-pad bottom/right to ``backbone.input_divisor`` and crop outputs
            back, which keeps outputs pixel-aligned with inputs. If False, pixel outputs are
            resized to the input size instead. A multi-input encoder (one taking a dict) always
            pads its inputs itself, and a fixed-size backbone always gets its input padded to
            that size, so their outputs are always cropped and this has no effect.

    Image-scope heads pool over the valid (unpadded) region only: when the input was padded,
    their feature map is first cropped to ``ceil(H / s) x ceil(W / s)`` (``s`` its stride).

    Pixel heads return the primary input's ``H x W``, or ``ceil(H / f) x ceil(W / f)`` for a head
    with ``scale = 1/f`` (output pixel ``(i, j)`` covers input pixels ``[i*f, (i+1)*f) x
    [j*f, (j+1)*f)``; see :mod:`manyseg.utils.resolution`).
    """

    # set by manyseg.create_model (the arguments, for manyseg.save) and by manyseg.load
    build_config: dict[str, Any] | None = None
    saved_metadata: dict[str, Any] | None = None

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
        head_inputs: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.backbone = backbone
        self.decoder = decoder
        self.task_decoders = nn.ModuleDict(task_decoders or {})
        self.heads = nn.ModuleDict(heads)
        sources = dict(head_sources or {})
        self._head_sources = {n: sources.get(n, "decoder") for n in self.heads}
        # per-input heads: their task decoder reads this input's own (unfused) features
        self._head_inputs = dict(head_inputs or {})
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
        # a multi-input encoder pads each input itself (whatever pad_to_stride says), so its
        # outputs are cropped to the unpadded region, never resized (that would stretch the
        # padding into the prediction)
        # (likewise a fixed-size backbone: _pad pads up to its size whatever pad_to_stride says)
        self._pads_itself = getattr(backbone, "primary", None) is not None
        self._crops = (pad_to_stride or self._pads_itself
                       or getattr(backbone, "fixed_size", None) is not None)
        self.divisor = int(getattr(backbone, "input_divisor", 1)) if pad_to_stride else 1
        # Image heads read the deepest encoder level; with late fusion the specs are every
        # input's levels concatenated, and the heads are built for the primary input's group.
        late = getattr(backbone, "mode", None) == "late"
        levels = range(*backbone.groups[0]) if late else range(len(backbone.feature_specs))
        self._deepest = deepest_index(backbone.feature_specs, levels)
        # padded inputs: decoders can skip size checks in their upsampling
        if pad_to_stride and getattr(backbone, "fixed_size", None) is None:
            for dec in [self.decoder, *self.task_decoders.values()]:
                if hasattr(dec, "_set_input_divisor"):
                    dec._set_input_divisor(self.divisor)

    @property
    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return self.backbone.feature_specs

    def pack_metadata(self, name: str, values) -> Tensor:
        """Per-field values ``{field: (B,) or (B, columns)}`` for the vector input ``name``
        -> the packed ``(B, N)`` tensor the model takes (NaN marks a missing value)."""
        encoders = getattr(self.conditioner, "encoders", {})
        if name not in encoders:
            raise ValueError(f"{name!r} is not a vector input of this model; vector inputs: "
                             f"{list(encoders)}")
        return encoders[name].pack(values)

    def select_heads(self, names) -> DenseModel:
        """Keep only the heads ``names`` (in place; returns the model): their decoders stay,
        decoders, FiLM and head-bias layers only the dropped heads used are removed. For
        deployment after training with auxiliary heads, e.g. a classifier trained with a pixel
        head. In eval mode the kept heads' outputs are unchanged (in training mode,
        dropout in removed modules no longer consumes random numbers, so masks differ). The
        output stays a dict. Removed modules leave the state dict: to load a pruned
        checkpoint, rebuild the model, call ``select_heads`` with the same names, then load."""
        names = [names] if isinstance(names, str) else list(names)
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
            self._head_inputs.pop(n, None)
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
                self._head_sources[n] == "encoder" for n in self.heads):
            del cond.film["image"]
        # levels only the dropped heads read; inputs the kept heads ignore are fine here
        self._restrict_to_read_levels(check_inputs=False)
        return self

    def _restrict_to_read_levels(self, check_inputs: bool = True) -> None:
        """Remove per-level modules nothing reads (fusers, learned temporal aggregations, ViT
        neck projections): they would be parameters without a gradient, which DDP rejects by
        default. Read levels: those each decoder reads (gradient probe, see
        ``conditioning.inject.used_levels``), plus the deepest when a head reads the encoder
        directly. Called by ``create_model`` (``check_inputs``: an input, or trailing levels of
        one, that nothing reads is an error) and ``select_heads``; idempotent."""
        bb = self.backbone
        if not hasattr(bb, "_restrict_levels") or not _prunable(bb):
            return
        from ..conditioning.inject import used_levels

        specs = tuple(bb.feature_specs)
        in_ch = getattr(bb, "in_channels", None)
        read, per_input, probed = set(), {}, {}
        decoders = [(self.decoder, None)] if self.decoder is not None else []
        decoders += [(d, self._head_inputs.get(h)) for h, d in self.task_decoders.items()]
        for d, inp in decoders:
            # a decoder shared by per-input heads reads each of their inputs: probe it once,
            # apply its levels to every input
            sp = specs if inp is None else tuple(bb.encoders[inp].feature_specs)
            if (id(d), sp) not in probed:
                probed[id(d), sp] = used_levels(d, sp, in_ch)
            if inp is None:
                read.update(probed[id(d), sp])
            else:
                per_input.setdefault(inp, set()).update(probed[id(d), sp])
        if any(src == "encoder" for src in self._head_sources.values()):
            read.add(self._deepest)
        kw = {"check_inputs": check_inputs} if hasattr(bb, "encoders") else {}
        if per_input:
            kw["per_input"] = per_input
        bb._restrict_levels(read, **kw)

    def tiling_spec(self) -> dict:
        """How to tile this model for inference (read by ManyPred, which ManySeg does not
        import). Plain data:

        * ``inputs``: ``{name: {"kind": "raster" | "timeseries" | "vector", "scale": r}}``,
          ``r`` the resolution ratio to the primary input: 1 on a shared grid, None for a
          ``same_extent`` input (its ratio is read from its shape). Empty for a model that
          takes a single tensor.
        * ``outputs``: ``{head: {"scope": "pixel" | "image", "kind": ..., "scale": s,
          "activation": a, "classes": names}}`` (``scale`` for pixel heads only; ``activation``
          and ``classes`` when the model was saved with them, see :func:`manyseg.save`); only
          ``single_output``'s head when set, as the model then returns that one tensor.
        * ``pad_multiple``: the primary input is padded to a multiple of this (1 with
          ``pad_to_stride=False`` on a single-input model; a multi-input encoder always pads).
        * ``max_size``: the largest ``(h, w)`` primary input the model takes, else None: the
          element-wise minimum of the fixed sizes of the primary's encoder and of every
          ``same_grid`` input's encoder (those inputs share the primary's size). A
          ``same_extent`` input's fixed-size encoder is not reflected here: its ratio to the
          primary is only known when the data arrives, so its limit is enforced only at
          forward time, where a larger input raises a ``ValueError`` naming the fixed size.
        * ``normalize``: the normalisation stored by :func:`manyseg.save` (a ManyPred
          normalisation spec), or None.
        * ``bands``: the input band names stored by :func:`manyseg.save`: a list for a model
          that takes a single tensor, else ``{input: names}``; or None.
        """
        saved = getattr(self, "saved_metadata", None) or {}
        backbone = self.backbone
        names = getattr(backbone, "names", None)
        primary = getattr(backbone, "primary", None)
        encoders = getattr(backbone, "encoders", {})
        inputs: dict[str, dict] = {}
        if names is not None:
            alignment = getattr(backbone, "alignment", {})
            for n in names:
                temporal = getattr(encoders[n], "temporal", False) if n in encoders else False
                inputs[n] = {"kind": "timeseries" if temporal else "raster",
                             "scale": None if alignment.get(n) == "same_extent" else 1.0}
        if self.conditioner is not None:
            for n in self.conditioner.rasters:
                inputs.setdefault(n, {"kind": "raster", "scale": 1.0})
            for n in self.conditioner.vectors:
                inputs[n] = {"kind": "vector"}
        outputs: dict[str, dict] = {}
        classes = saved.get("classes") or {}
        if not isinstance(classes, Mapping):  # save takes a list for a single head
            classes = {next(iter(self.heads)): classes}
        returned = [self.single_output] if self.single_output is not None else list(self.heads)
        for name in returned:
            head = self.heads[name]
            entry: dict = {"scope": getattr(head, "scope", "pixel"),
                           "kind": getattr(head, "kind", None)}
            if entry["scope"] == "pixel":
                entry["scale"] = 1 / getattr(head, "factor", 1)
            if name in saved.get("activations", {}):
                entry["activation"] = saved["activations"][name]
            if classes.get(name):
                entry["classes"] = [str(c) for c in classes[name]]
            outputs[name] = entry
        main = dict(encoders).get(primary, backbone)  # ModuleDict has no .get
        fixed = getattr(main, "fixed_size", None) or getattr(
            getattr(main, "backbone", None), "fixed_size", None)
        # Every same_grid input has the primary's size, so its fixed size limits the primary
        # too: report the most restrictive. A same_extent input's limit is in its own pixels
        # and its ratio to the primary is only known from the data, so it is left out (its
        # encoder still refuses a larger input at forward time).
        fixed_sizes = getattr(backbone, "fixed_sizes", None) or {}
        alignment = getattr(backbone, "alignment", {})
        for n, fs in fixed_sizes.items():
            if alignment.get(n) == "same_extent":
                continue
            fixed = tuple(fs) if fixed is None else (min(fixed[0], fs[0]), min(fixed[1], fs[1]))
        return {
            "inputs": inputs,
            "outputs": outputs,
            "pad_multiple": (int(getattr(backbone, "grid_divisor",
                                         getattr(backbone, "input_divisor", 1)))
                             if self.pad_to_stride or self._pads_itself else 1),
            "max_size": tuple(fixed) if fixed is not None else None,
            "normalize": saved.get("normalize"),
            "bands": self._band_names(saved.get("bands")),
        }

    def _band_names(self, bands: Mapping[str, Sequence[str]] | Sequence[str] | None
                    ) -> dict[str, list[str]] | list[str] | None:
        """``bands`` as :meth:`tiling_spec` reports them: :func:`manyseg.save` also takes a list
        for one named image input, which the spec keys by that input."""
        if not bands:
            return None
        if isinstance(bands, Mapping):
            return {n: [str(b) for b in names] for n, names in bands.items()}
        config = getattr(self, "build_config", None) or {}
        images = [n for n, c in (config.get("inputs") or {}).items()
                  if c.get("kind", "raster") != "vector"]
        names = [str(b) for b in bands]
        return {images[0]: names} if images else names

    def describe(self, sample: Tensor | Mapping[str, Tensor] | None = None,
                 format: str = "text") -> str:
        """Stage-level description of the built model (``"text"``, ``"mermaid"`` or
        ``"dot"``); with ``sample``, annotated with real tensor shapes and FLOPs. See
        :func:`manyseg.describe.describe`."""
        from ..describe import describe

        return describe(self, sample, format)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Gradient checkpointing for every encoder in the model (see
        :meth:`manyseg.backbones.TimmBackbone.set_grad_checkpointing`): the encoder's
        activations are recomputed in the backward pass instead of stored. Less training
        memory for more compute; values and BatchNorm statistics are unchanged."""
        from ..backbones.timm_backbone import TimmBackbone

        encoders = [m for m in self.modules() if isinstance(m, TimmBackbone)]
        if not encoders:
            raise NotImplementedError("set_grad_checkpointing: this model has no timm encoder")
        for m in encoders:  # all or nothing: check every encoder before changing one
            m._ckpt_target()
        for m in encoders:
            m.set_grad_checkpointing(enable)

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
                  cond=None, valid=None, input_feats=None) -> Tensor:
        src = self._head_sources[name]
        if name in self._head_inputs:
            feats = input_feats[self._head_inputs[name]]
        if src == "decoder":
            inp = dec
        elif src == "encoder":
            inp = feats[self._deepest]
            if cond is not None:  # FiLM on the deepest feature, if built
                inp = self.conditioner.image(inp, cond)
        else:
            inp = self._decode_cond(f"task_{name}", self.task_decoders[name], x, feats, cond,
                                    valid)
        head = self.heads[name]
        if name not in self._pixel_heads and self._crops:
            # image heads pool: crop the padding away first so it does not dilute the mean
            # (floor division keeps the sizes symbolic for torch.export)
            s = self._source_stride(name, src)
            inp = crop_to(inp, (h + s - 1) // s, (w + s - 1) // s)
        # a head bias goes before a built-in output activation, so softplus heads stay > 0
        late = (cond is not None and getattr(head, "activation", None) is not None
                and name in self.conditioner.head_bias)
        y = head(inp, activate=False) if late else head(inp)
        if name in self._pixel_heads:
            f = getattr(head, "factor", 1)  # 1 / scale; floor division keeps sizes symbolic
            hs, ws = (h + f - 1) // f, (w + f - 1) // f
            if self._crops:
                y = crop_to(y, hs, ws)
            else:
                y = F.interpolate(y, size=(hs, ws), mode="bilinear", align_corners=False)
        if cond is not None:
            y = self.conditioner.head(name, y, cond)
        return head.activate(y) if late else y

    def _source_stride(self, name: str, src: str) -> int:
        """Stride of the feature map head ``name`` reads."""
        if src == "decoder":
            return self.decoder.out_spec.stride
        if src == "encoder":
            return self.backbone.feature_specs[self._deepest].stride
        return self.task_decoders[name].out_spec.stride

    def _pad(self, x: Tensor) -> Tensor:
        fixed = getattr(self.backbone, "fixed_size", None)
        if fixed is None:
            return pad_to_multiple(x, self.divisor)
        h, w = x.shape[-2], x.shape[-1]
        if h > fixed[0] or w > fixed[1]:
            raise ValueError(
                f"backbone {getattr(self.backbone, 'name', '')!r} has a fixed input size {fixed}; "
                f"input {(h, w)} is larger. Use tiled inference (ManyPred), or rebuild with "
                f"backbone_kwargs={{'img_size': (H, W)}} for a larger fixed size (see "
                "https://manyseg.readthedocs.io/en/latest/docs/backbone_compatibility/)."
            )
        return F.pad(x, (0, fixed[1] - w, 0, fixed[0] - h))

    def _encode(self, x, inputs: bool = False):
        """Conditioner + (padded) encoder: ``(feats, input_feats, cond, xin, h, w, valid)``."""
        cond, input_feats = None, None
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
            if inputs:
                feats, input_feats = self.backbone(x, return_inputs=True)
            else:
                feats = self.backbone(x)
        else:
            if isinstance(x, Mapping):
                raise TypeError("this model takes a single tensor, not a dict (build it with "
                                "inputs={...} to name its inputs)")
            h, w = x.shape[-2], x.shape[-1]
            xin, valid = self._pad(x), None
            feats = self.backbone(xin)
        return feats, input_feats, cond, xin, h, w, valid

    def features(self, x: Tensor | dict[str, Tensor]) -> list[Tensor | None]:
        """The encoder pyramid (fused, for multi-input models) for ``x``, each level cropped to
        the input's valid region (``ceil(H / stride) x ceil(W / stride)``). Late fusion: every
        input's levels side by side (``backbone.groups``), each cropped to its own input.

        Levels the model never reads are built minimally: a plain ViT's unread neck levels are
        ``None``, and a multi-input model does not fuse at unread levels (the primary input's
        feature passes through)."""
        feats, _, _, _, h, w, valid = self._encode(x)

        def crop(f, s, vh, vw):  # crop_to (negative padding), not slicing: export-safe
            return None if f is None else crop_to(f, (vh + s.stride - 1) // s.stride,
                                                  (vw + s.stride - 1) // s.stride)

        if getattr(self.backbone, "mode", None) == "late":
            out = []
            for (a, b), (vh, vw) in zip(self.backbone.groups, valid):
                out += [crop(f, s, vh, vw) for f, s in zip(feats[a:b], self.feature_specs[a:b])]
            return out
        return [crop(f, s, h, w) for f, s in zip(feats, self.feature_specs)]

    def embed(self, x: Tensor | dict[str, Tensor], source: str = "encoder", level: int = -1,
              pool: str | None = None, normalize: bool = False) -> Tensor:
        """Feature vectors for retrieval, clustering, probing: per pixel ``(B, C, h, w)`` or
        pooled ``(B, C)``.

        Args:
            source: ``"encoder"`` (a level of :meth:`features`; default the deepest) or
                ``"decoder"`` (the shared decoder's output, metadata FiLM applied).
            level: Encoder level index (``-1`` = deepest); with late fusion, an index into the
                primary input's levels.
            pool: ``None`` (per pixel), ``"mean"`` or ``"max"`` over the valid region.
            normalize: L2-normalise along channels.
        """
        if source not in ("encoder", "decoder"):
            raise ValueError(f"source must be 'encoder' or 'decoder', got {source!r}")
        if pool not in (None, "mean", "max"):
            raise ValueError(f"pool must be None, 'mean' or 'max', got {pool!r}")
        if source == "encoder":
            feats = self.features(x)
            if getattr(self.backbone, "mode", None) == "late":
                a, b = self.backbone.groups[0]  # the primary input's levels
                feats = feats[a:b]
            e = feats[level]
            if e is None:
                raise ValueError(f"encoder level {level} is not computed by this model (nothing "
                                 "reads it); choose another level")
        else:
            if self.decoder is None:
                raise ValueError("source='decoder' needs a shared decoder (this model has only "
                                 "image heads or per-head decoders)")
            feats, _, cond, xin, h, w, valid = self._encode(x)
            e = self._decode_cond("decoder", self.decoder, xin, feats, cond, valid)
            s = self.decoder.out_spec.stride
            e = crop_to(e, (h + s - 1) // s, (w + s - 1) // s)
        if pool == "mean":
            e = e.mean(dim=(-2, -1))
        elif pool == "max":
            e = e.amax(dim=(-2, -1))
        return F.normalize(e, dim=1) if normalize else e

    def embedder(self, **kwargs) -> Embedder:
        """An ``nn.Module`` whose forward is ``self.embed(x, **kwargs)`` (exportable)."""
        return Embedder(self, **kwargs)

    def forward(self, x: Tensor | dict[str, Tensor]) -> Tensor | dict[str, Tensor]:
        feats, input_feats, cond, xin, h, w, valid = self._encode(
            x, inputs=bool(self._head_inputs))
        dec = (self._decode_cond("decoder", self.decoder, xin, feats, cond, valid)
               if self.decoder is not None else None)
        if self.single_output is not None:
            return self._run_head(self.single_output, dec, feats, xin, h, w, cond, valid,
                                  input_feats)
        return {name: self._run_head(name, dec, feats, xin, h, w, cond, valid, input_feats)
                for name in self.heads}


class Embedder(nn.Module):
    """``model.embed(x, **kwargs)`` as a module, e.g. for ``manyseg.export.export_model``."""

    def __init__(self, model: DenseModel, **kwargs) -> None:
        super().__init__()
        self.model, self.kwargs = model, dict(kwargs)

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, x):
        return self.model.embed(x, **self.kwargs)


def _prunable(bb: nn.Module) -> bool:
    from ..backbones.necks import ViTPyramidNeck
    from ..temporal import TemporalEncoder

    if getattr(bb, "mode", None) in ("intermediate", "late") and len(bb.names) > 1:
        return True  # also checks that every input is read
    return any(isinstance(m, (ViTPyramidNeck, TemporalEncoder)) or
               (hasattr(m, "fusers") and len(m.fusers)) for m in bb.modules())
