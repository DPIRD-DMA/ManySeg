"""Stage-level summaries and diagrams of a built :class:`~manyseg.models.DenseModel`.

``describe(model)`` reads the constructed modules, never the ``create_model`` arguments, so it
shows what was actually built: inputs and channel adapters, encoders (feature levels,
pretrained or not), fusion, metadata conditioning, decoders and heads, with parameters per
stage and the decoder that feeds each head. With a ``sample`` it also runs one ``no_grad``
forward pass with hooks and records the tensor shapes at the stage boundaries (padded encoder
input, encoder levels, fusion inputs and output, decoder outputs, raw and cropped head
outputs) plus FLOPs per stage (``torch.utils.flop_counter``).

Formats are plain strings: ``"text"`` (an indented tree), ``"mermaid"`` (a flowchart) and
``"dot"`` (Graphviz). :func:`render_dot` calls the ``dot`` binary if it is installed.
:func:`model_graph` returns the underlying nodes and edges for programmatic checks.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

FORMATS = ("text", "mermaid", "dot")
"""``describe(format=...)`` values."""
STAGES = ("input", "adapter", "encoder", "fusion", "decoder", "head")
_STAGE_TITLES = {"input": "inputs", "adapter": "channel adapters", "encoder": "encoders",
                 "fusion": "fusion", "decoder": "decoders", "head": "heads"}
_COLORS = {"input": "#e8f0fe", "adapter": "#f1f3f4", "encoder": "#e6f4ea", "fusion": "#fef7e0",
           "decoder": "#fce8e6", "head": "#e0f2f1", "vector": "#f3e8fd"}


@dataclass
class Node:
    """One stage component. ``attrs`` are ``(key, value, in_diagram)``; ``shapes`` come from
    the sample forward pass."""

    id: str
    stage: str
    name: str
    title: str
    attrs: list[tuple[str, str, bool]] = field(default_factory=list)
    shapes: list[tuple[str, str]] = field(default_factory=list)
    params: int | None = None
    flops: int | None = None
    style: str = ""  # colour key override (vector inputs)

    def attr(self, key: str) -> str | None:
        return next((v for k, v, _ in self.attrs if k == key), None)

    def shape(self, key: str) -> str | None:
        return next((v for k, v in self.shapes if k == key), None)


@dataclass
class Edge:
    """An edge of the model graph that :meth:`~manyseg.DenseModel.describe` draws, from node
    ``src`` to ``dst``."""

    src: str
    dst: str
    label: str = ""
    kind: str = "data"  # "data" | "condition" (metadata) | "image" (raw input to a decoder)


@dataclass
class ModelGraph:
    """Nodes (in stage order) and edges of a model description."""

    nodes: dict[str, Node]
    edges: list[Edge]
    summary: list[tuple[str, str]]
    stage_params: dict[str, int]
    # manyseg.weights.attribution: the pretrained weights, their licences and citations
    pretrained_weights: list[dict] = field(default_factory=list)

    def by_stage(self, stage: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.stage == stage]

    def find(self, stage: str, name: str) -> Node:
        for n in self.nodes.values():
            if n.stage == stage and n.name == name:
                return n
        raise KeyError(f"no {stage} node named {name!r}")

    def sources(self, node_id: str, kind: str | None = None) -> list[Node]:
        return [self.nodes[e.src] for e in self.edges
                if e.dst == node_id and (kind is None or e.kind == kind)]

    def targets(self, node_id: str, kind: str | None = None) -> list[Node]:
        return [self.nodes[e.dst] for e in self.edges
                if e.src == node_id and (kind is None or e.kind == kind)]

    def render(self, format: str = "text") -> str:
        if format == "text":
            return _to_text(self)
        if format == "mermaid":
            return _to_mermaid(self)
        if format == "dot":
            return _to_dot(self)
        raise ValueError(f"unknown format {format!r}; choose from {FORMATS}")

    def __str__(self) -> str:
        return _to_text(self)


# ---------------------------------------------------------------------------------------------
# helpers


def _registry_name(registry: Mapping, obj: nn.Module) -> str:
    for k, v in registry.items():
        if v is type(obj):
            return k
    return type(obj).__name__


def _nparams(*modules: nn.Module | None) -> int:
    seen, n = set(), 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
    return n


def _fmt_count(n: int | None) -> str:
    if n is None:
        return "?"
    for div, suf in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return str(n)


def _fmt_shape(t) -> str:
    if isinstance(t, Tensor):
        return "x".join(str(d) for d in t.shape)
    if isinstance(t, (list, tuple)):
        return ", ".join(_fmt_shape(x) for x in t)
    if isinstance(t, Mapping):
        return ", ".join(f"{k} {_fmt_shape(v)}" for k, v in t.items())
    return type(t).__name__


def _fmt_hidden(seq) -> str:
    """``64 (d8) -> 32 (d16), bn + ReLU`` for a head's hidden conv layers."""
    convs = [m for m in seq if isinstance(m, torch.nn.Conv2d)]
    norm = next((type(m).__name__ for m in seq if not isinstance(
        m, (torch.nn.Conv2d, torch.nn.ReLU))), None)
    norm = {"BatchNorm2d": "bn", "GroupNorm": "gn"}.get(norm, norm)
    layers = " -> ".join(f"{c.out_channels}" + (f" (d{c.dilation[0]})" if c.dilation[0] > 1
                                                 else "") for c in convs)
    return f"{layers}, {norm + ' + ' if norm else ''}ReLU" if convs else "none"


def _fmt_scale(f: int) -> str:
    return "1" if f == 1 else f"1/{f}"


def _levels(specs) -> str:
    return ", ".join(f"{s.channels}@/{s.stride}" for s in specs)


def _strides(specs, idx) -> str:
    return ", ".join(f"/{specs[i].stride}" for i in idx)


def _read_levels(dec: nn.Module) -> list[int] | None:
    """Encoder levels a decoder reads, from its own attributes (no probe)."""
    from .decoders import (
        DeepLabV3PlusDecoder,
        FastaiUNetDecoder,
        UNetDecoder,
        UNetPlusPlusDecoder,
    )

    declared = getattr(dec, "read_levels", None)
    if declared is not None:
        return sorted(int(i) for i in declared)
    if isinstance(dec, UNetDecoder):
        return sorted({dec._deepest_index, *(i for i in dec._skip_index if i >= 0)})
    if isinstance(dec, UNetPlusPlusDecoder):
        return sorted(dec._order)
    if isinstance(dec, FastaiUNetDecoder):
        return sorted({dec.deep - 1, *(i - 1 for i in dec.skips)})
    if isinstance(dec, DeepLabV3PlusDecoder):
        return sorted({dec.low, dec.high})
    levels = getattr(dec, "levels", None)
    return sorted(levels) if levels is not None else None


def _image_use(dec: nn.Module) -> str:
    """How a ``wants_input`` decoder uses the raw image: ``fastai_unet`` below stride 1 only
    reads its size."""
    from .decoders import FastaiUNetDecoder

    if isinstance(dec, FastaiUNetDecoder) and not dec.last_cross:
        return "size only"
    return "concatenated at stride 1"


def _decoder_options(dec: nn.Module) -> list[tuple[str, str, bool]]:
    from .decoders import (
        DeepLabV3PlusDecoder,
        FastaiUNetDecoder,
        UNetDecoder,
        UNetPlusPlusDecoder,
    )

    if isinstance(dec, UNetDecoder):
        return [("block channels", str(tuple(dec.channels)), False),
                ("upsample", dec.blocks[0].upsample if len(dec.blocks) else "-", False),
                ("input_cross", str(dec.wants_input), True)]
    if isinstance(dec, UNetPlusPlusDecoder):
        return [("nodes", str(len(dec.nodes)), False), ("tail blocks", str(len(dec.tail)), False)]
    if isinstance(dec, FastaiUNetDecoder):
        return [("last_cross", str(bool(dec.last_cross)), True),
                ("final shuffle", str(dec.final_shuffle is not None), False)]
    if isinstance(dec, DeepLabV3PlusDecoder):
        return [("atrous rates", str(dec.atrous_rates), False)]
    return []


class _Capture:
    """Forward hooks recording inputs and outputs per module, in call order."""

    def __init__(self) -> None:
        self.calls: dict[int, list[tuple[tuple, object]]] = {}
        self._handles = []

    def watch(self, module: nn.Module | None) -> None:
        if module is None or id(module) in self.calls:
            return
        self.calls[id(module)] = []

        def hook(mod, args, out):
            self.calls[id(mod)].append((args, out))

        self._handles.append(module.register_forward_hook(hook))

    def get(self, module: nn.Module, call: int = 0):
        calls = self.calls.get(id(module), [])
        return calls[call] if call < len(calls) else None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------------------------
# graph construction


class _Builder:
    def __init__(self, model: nn.Module, sample=None) -> None:
        from .fusion import EarlyFusionEncoder, LateFusionDecoder, MultiEncoder
        from .weights import attribution

        self.model = model
        # by backbone module: a loaded model's backbones were rebuilt without weights, so they
        # no longer say what they started from; the record saved with the model does
        self.pretrained_weights = attribution(model)
        self.weights_by_id = {id(model.get_submodule(e["module"])): e
                              for e in self.pretrained_weights}
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.paths = {}
        for name, m in model.named_modules():
            self.paths.setdefault(id(m), name)
        bb = model.backbone
        self.bb = bb
        self.cond = getattr(model, "conditioner", None)
        self.early = isinstance(bb, EarlyFusionEncoder)
        self.multi = isinstance(bb, MultiEncoder)
        self.late = self.multi and bb.mode == "late"
        self.LateFusionDecoder = LateFusionDecoder
        if self.cond is not None:
            self.rasters = list(self.cond.rasters)
        elif self.multi or self.early:
            self.rasters = list(bb.names)
        else:
            self.rasters = ["x"]  # model(x) takes a plain tensor
        self.cap = _Capture()
        self.flops: dict[str, int] = {}
        self.output = None
        if sample is not None:
            self._run(sample)

    # --- sample pass -------------------------------------------------------------------------

    def _watched(self):
        m, bb = self.model, self.bb
        mods = [bb, m.conditioner if self.cond is not None else None]
        if self.multi:
            mods += list(bb.encoders.values()) + list(bb.adapters.values())
            mods += list(getattr(bb, "fusers", {}).values())
        if self.early:
            mods.append(bb.backbone)
        decs = ([m.decoder] if m.decoder is not None else []) + list(m.task_decoders.values())
        for d in decs:
            mods.append(d)
            if isinstance(d, self.LateFusionDecoder):
                mods += list(d.decoders) + [d.fuse]
        return mods + list(m.heads.values())

    def _run(self, sample) -> None:
        from torch.utils.flop_counter import FlopCounterMode

        for mod in self._watched():
            self.cap.watch(mod)
        modes = [(mod, mod.training) for mod in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                try:
                    with FlopCounterMode(display=False) as fc:
                        self.output = self.model(sample)
                    root = type(self.model).__name__
                    for k, v in fc.get_flop_counts().items():
                        if k == root:
                            self.flops[""] = sum(v.values())
                        elif k.startswith(root + "."):
                            self.flops[k[len(root) + 1:]] = sum(v.values())
                except Exception:  # noqa: BLE001 - FLOP counting is optional; shapes are not
                    self.flops.clear()
                    for calls in self.cap.calls.values():
                        calls.clear()
                    self.output = self.model(sample)
        finally:
            self.cap.remove()
            for mod, training in modes:
                mod.training = training
        self.sample = sample

    def _flops(self, *modules) -> int | None:
        if not self.flops:
            return None
        return sum(self.flops.get(self.paths.get(id(m), "?"), 0) for m in modules if m is not None)

    # --- nodes -------------------------------------------------------------------------------

    def add(self, node_id: str, stage: str, name: str, title: str, modules=(), **kw) -> Node:
        node_id = re.sub(r"\W", "_", node_id)
        base, k = node_id, 2
        while node_id in self.nodes:
            node_id, k = f"{base}_{k}", k + 1
        mods = [m for m in modules if m is not None]
        n = Node(node_id, stage, name, title, params=_nparams(*mods) if mods else None,
                 flops=self._flops(*mods) if mods else None, **kw)
        self.nodes[node_id] = n
        return n

    def edge(self, src: Node, dst: Node, label: str = "", kind: str = "data") -> None:
        self.edges.append(Edge(src.id, dst.id, label, kind))

    def build(self) -> ModelGraph:
        self.inputs = self._inputs()
        feats = self._encoders()
        self.dec_out: dict[str, Node] = {}
        m = self.model
        if m.decoder is not None:
            self.dec_out["decoder"] = self._decoder("decoder", "decoder", m.decoder, feats)
        first = {}
        for h, d in m.task_decoders.items():
            inp = m._head_inputs.get(h)
            src = self.per_input_nodes[inp] if inp else feats  # per-input heads: own encoder
            specs = self.bb.encoders[inp].feature_specs if inp else None
            label = f"decoder[{h}]" + (f" on {inp}" if inp else "")
            node = self._decoder(f"task_{h}", label, d, src, specs)
            if id(d) in first:
                node.attrs.append(("shares", f"decoder and head of {first[id(d)]!r}", True))
            first.setdefault(id(d), h)
            self.dec_out[f"task_{h}"] = node
        self._heads(feats)
        self._conditioning_edges()
        return ModelGraph(self.nodes, self.edges, self._summary(), self._stage_params(),
                          self.pretrained_weights)

    def _vector_widths(self) -> dict[str, int]:
        """Extra channels that ``channels``-mode metadata adds to each raster."""
        out: dict[str, int] = {}
        if self.cond is not None:
            for v, t in self.cond.channel_targets.items():
                out[t] = out.get(t, 0) + self.cond.encoders[v].width
        return out

    def _raster_channels(self, name: str) -> int | None:
        bb = self.bb
        if self.early:
            return bb.channels.get(name)
        if self.multi:
            if name in bb.adapters:
                return bb.adapters[name].proj.in_channels
            return getattr(bb.encoders[name], "in_channels", None)
        return getattr(bb, "in_channels", None)

    def _inputs(self) -> dict[str, Node]:
        nodes = {}
        extra = self._vector_widths()
        primary = getattr(self.bb, "primary", None)
        for r in self.rasters:
            c = self._raster_channels(r)
            if c is not None and r in extra:
                ch = f"{c - extra[r]} ch + {extra[r]} metadata ch"
            else:
                ch = f"{c} ch" if c is not None else "? ch"
            n = self.add(f"in_{r}", "input", r, f"raster, {ch}")
            if self.multi:
                n.attrs.append(("alignment", self.bb.alignment[r], True))
            elif self.early:
                n.attrs.append(("alignment", "same_grid (early fusion)", False))
            if r == primary or (primary is None and len(self.rasters) == 1):
                n.attrs.append(("primary", "yes", False))
            if self.output is not None:
                s = self.sample
                t = s.get(r) if isinstance(s, Mapping) else s
                if isinstance(t, Tensor):
                    n.shapes.append(("sample", _fmt_shape(t)))
            nodes[r] = n
        if self.cond is not None:
            from .conditioning.fields import FIELD_TYPES

            for v in self.cond.vectors:
                enc = self.cond.encoders[v]
                mode = self.cond.modes[v]
                cols = f"{enc.columns} col" + ("s" if enc.columns != 1 else "")
                n = self.add(f"in_{v}", "input", v, f"vector, {cols}, {mode}", modules=[enc],
                             style="vector")
                if enc.field_names == tuple(str(i) for i in range(len(enc.field_names))):
                    n.attrs.append(("fields", f"{len(enc.field_names)} pre-normalised floats",
                                    True))
                else:
                    fields = ", ".join(f"{f} ({_registry_name(FIELD_TYPES, enc.fields[f])})"
                                       for f in enc.field_names)
                    n.attrs.append(("fields", fields, True))
                mlp = "no MLP" if isinstance(enc.mlp, nn.Identity) else f"MLP -> {enc.dim}"
                n.attrs.append(("encoding", f"width {enc.width}, {mlp}", True))
                if enc.dropout:
                    n.attrs.append(("field dropout", str(enc.dropout), False))
                if self.output is not None and isinstance(self.sample, Mapping):
                    n.shapes.append(("sample", _fmt_shape(self.sample[v])))
                nodes[v] = n
        return nodes

    def _encoder_node(self, enc: nn.Module, name: str, node_id: str) -> Node:
        from .adapters import ProjectionAdapter
        from .backbones import TimmBackbone
        from .temporal import TemporalEncoder

        if isinstance(enc, TemporalEncoder):
            n = self._encoder_node(enc.backbone, name, node_id)
            n.attrs.insert(0, ("time series", f"(B, T, C, H, W); encoder per date, {enc.kind} "
                                              "over dates at every level", True))
            return n

        if isinstance(enc, TimmBackbone):
            rec = self.weights_by_id.get(id(enc))
            pretrained = enc.pretrained or rec is not None
            weights = getattr(enc, "weights", None) or (
                rec["weights"] if rec and rec.get("provider") == "manyseg" else None)
            init = (f"weights {weights}" if weights
                    else "pretrained" if pretrained else "random init")
            n = self.add(node_id, "encoder", name, f"{enc.name} ({init})", modules=[enc])
            cfg = getattr(enc.model, "pretrained_cfg", {}) or {}
            # remote-sensing weights come with the input's band count: no stem adaptation
            native = enc.in_channels if weights else (cfg.get("input_size") or (3,))[0]
            if isinstance(enc.input_adapter, ProjectionAdapter):
                p = enc.input_adapter.proj
                src = getattr(enc, "rgb_indices", None) or tuple(
                    o % p.in_channels for o in range(p.out_channels))
                stem = (f"{enc.in_channels} ch, projection 1x1 conv "
                        f"{p.in_channels}->{p.out_channels} (starts by passing ch "
                        f"{', '.join(map(str, src))} to the pretrained stem)")
            elif enc.in_channels != native and not pretrained:
                stem = f"{enc.in_channels} ch, random init (no pretrained weights to adapt)"
            elif enc.in_channels != native:
                stem = (f"{enc.in_channels} ch, '{enc.channel_adapter}' weights from {native} "
                        f"ch ({_stem_mapping(enc, native)})")
            else:
                stem = f"{enc.in_channels} ch, unchanged"
            n.attrs.append(("stem", stem, True))
            if enc.neck is not None:
                blocks = ",".join(map(str, enc.block_indices))
                n.attrs.append(("layout", f"plain ViT, blocks {blocks} at "
                                f"/{enc.neck.patch_stride} -> pyramid neck", True))
            elif enc.nhwc:
                n.attrs.append(("layout", "NHWC features, permuted to NCHW", False))
            if enc.fixed_size is not None:
                n.attrs.append(("fixed size", str(enc.fixed_size), True))
        else:
            n = self.add(node_id, "encoder", name, type(enc).__name__, modules=[enc])
        n.attrs.append(("levels", _levels(enc.feature_specs), True))
        n.attrs.append(("divisor", str(getattr(enc, "input_divisor", "?")), False))
        return n

    def _shapes(self, node: Node, mod: nn.Module, inp: bool = True) -> None:
        got = self.cap.get(mod)
        if got is None:
            return
        args, out = got
        if inp and args and isinstance(args[0], Tensor):
            node.shapes.append(("padded in", _fmt_shape(args[0])))
        node.shapes.append(("out", _fmt_shape(out)))

    def _encoders(self):
        """Encoder (and fusion) nodes; returns the feature source: one node, or per input name
        (late fusion)."""
        bb = self.bb
        if self.early:
            fuse = self.add("fusion", "fusion", "fusion",
                            f"early, concat {len(bb.names)} inputs")
            fuse.attrs.append(("order", " + ".join(bb.names), True))
            for r in bb.names:
                self.edge(self.inputs[r], fuse)
            enc = self._encoder_node(bb.backbone, "encoder", "enc")
            self.edge(fuse, enc, f"{getattr(bb.backbone, 'in_channels', '?')} ch")
            self._shapes(enc, bb.backbone)
            return enc
        if not self.multi:
            enc = self._encoder_node(bb, "encoder", "enc")
            self.edge(self.inputs[self.rasters[0]], enc)
            self._shapes(enc, bb)
            return enc

        # Encoders share weights either as one module ("share_norm": True) or as weight-sharing
        # copies with their own BatchNorm (the default): group by a shared parameter.
        def key(mod):
            return id(next(iter(mod.parameters()), mod))

        users: dict[int, list[str]] = {}
        for r in bb.names:
            users.setdefault(key(bb.encoders[r]), []).append(r)
        enc_nodes, calls = {}, {}
        for r in bb.names:
            mod = bb.encoders[r]
            group = users[key(mod)]
            if r == group[0]:
                name = r if len(group) == 1 else f"shared ({', '.join(group)})"
                node = self._encoder_node(mod, name, f"enc_{r}")
                one = len({id(bb.encoders[g]) for g in group}) == 1
                node.attrs.append(("sharing", "independent" if len(group) == 1 else
                                   f"one module for {', '.join(group)}" if one else
                                   f"weights shared by {', '.join(group)}; BatchNorm per input",
                                   True))
                if not one:  # count each group member's own BatchNorm once, shared weights once
                    members = list({id(bb.encoders[g]): bb.encoders[g] for g in group}.values())
                    node.params, node.flops = _nparams(*members), self._flops(*members)
                enc_nodes[key(mod)] = node
            node = enc_nodes[key(mod)]
            src = self.inputs[r]
            if r in bb.adapters:
                ad = bb.adapters[r]
                a = self.add(f"adapt_{r}", "adapter", r, type(ad).__name__, modules=[ad])
                p = getattr(ad, "proj", None)
                if isinstance(p, nn.Conv2d):
                    a.attrs.append(("channels", f"{p.in_channels} -> {p.out_channels}", True))
                self._shapes(a, ad, inp=False)
                self.edge(src, a)
                src = a
            self.edge(src, node)
            k = calls.get(id(mod), 0)
            calls[id(mod)] = k + 1
            got = self.cap.get(mod, k)
            if got is not None:
                pre = f"[{r}] " if len(group) > 1 else ""
                node.shapes.append((f"{pre}padded in", _fmt_shape(got[0][0])))
                node.shapes.append((f"{pre}out", _fmt_shape(got[1])))
        per_input = {r: enc_nodes[key(bb.encoders[r])] for r in bb.names}
        self.per_input_nodes = per_input
        if self.late:
            return per_input

        op = "-"
        if len(bb.fusers):
            from .fusion import FUSION_OPS

            op = _registry_name(FUSION_OPS, next(iter(bb.fusers.values())))
        fuse = self.add("fusion", "fusion", "fusion", f"intermediate, op {op}",
                        modules=list(bb.fusers.values()))
        fused = ", ".join(f"/{s}" for s in bb.fused_strides) or "none"
        fuse.attrs.append(("fused strides", fused, True))
        pspecs = bb.feature_specs
        plan = "; ".join(f"/{pspecs[li].stride}: " + "+".join(n for n, _ in members)
                         for li, members in enumerate(bb._plan) if len(members) > 1)
        fuse.attrs.append(("plan", plan or "-", False))
        extent = [r for r in bb.names if r != bb.primary and bb.alignment[r] == "same_extent"]
        if extent:
            fuse.attrs.append(("resampled", f"{', '.join(extent)} (same_extent: valid region, "
                               "bilinear, onto the primary grid)", True))
        fuse.attrs.append(("output levels", _levels(pspecs), False))
        for r in bb.names:
            if r == bb.primary:
                label = "all levels"
            else:
                label = ", ".join(f"/{pspecs[li].stride}" for li, members in enumerate(bb._plan)
                                  if any(n == r for n, _ in members[1:])) or "unused"
            self.edge(per_input[r], fuse, label)
        for li, f in bb.fusers.items():
            got = self.cap.get(f)
            if got is not None:
                fuse.shapes.append((f"/{pspecs[int(li)].stride} in", _fmt_shape(got[0][0])))
        got = self.cap.get(bb)
        if got is not None:
            fuse.shapes.append(("out", _fmt_shape(got[1])))
        return fuse

    def _film(self, key: str) -> nn.Module | None:
        if self.cond is None:
            return None
        return self.cond.film[key] if key in self.cond.film else None  # noqa: SIM401 (ModuleDict has no get)

    def _decoder_node(self, node_id: str, name: str, dec: nn.Module, specs) -> Node:
        from .decoders import DECODERS

        kind = _registry_name(DECODERS, dec)
        out = dec.out_spec
        n = self.add(node_id, "decoder", name, f"{kind}, out_stride {out.stride}, "
                     f"width {out.channels}", modules=[dec])
        reads = _read_levels(dec)
        if reads is not None:
            n.attrs.append(("reads", _strides(specs, reads), True))
        n.attrs += _decoder_options(dec)
        if getattr(dec, "wants_input", False):
            n.attrs.append(("input image", _image_use(dec), True))
        self._shapes(n, dec, inp=False)
        return n

    def _decoder(self, key: str, name: str, dec: nn.Module, feats, specs=None) -> Node:
        specs = self.bb.feature_specs if specs is None else specs
        film = self._film(key)
        if isinstance(dec, self.LateFusionDecoder):
            from .fusion import FUSION_OPS

            fuse = self.add(f"late_{key}", "fusion", f"fusion -> {name}",
                            f"late, op {_registry_name(FUSION_OPS, dec.fuse)}", modules=[dec.fuse])
            fuse.attrs.append(("fuses", f"{len(dec.decoders)} decoder outputs", True))
            # Read from the built decoder: it resamples same_extent outputs by valid region.
            align = getattr(dec, "alignment", ("same_grid",) * len(dec.groups))
            extent = [r for r, a in zip(self.bb.names[1:], align[1:]) if a == "same_extent"]
            if extent:
                fuse.attrs.append(("resampled", f"{', '.join(extent)} (same_extent: valid "
                                   "region, bilinear, onto the primary's valid region)", True))
            for r, sub, (a, b) in zip(self.bb.names, dec.decoders, dec.groups):
                n = self._decoder_node(f"dec_{key}_{r}", f"{name} [{r}]", sub, specs[a:b])
                self.edge(feats[r], n)
                self.edge(n, fuse)
            got = self.cap.get(dec)
            if got is not None:
                fuse.shapes.append(("out", _fmt_shape(got[1])))
            target = fuse
        else:
            target = self._decoder_node(f"dec_{key}", name, dec, specs)
            reads = _read_levels(dec)
            self.edge(feats, target, _strides(specs, reads) if reads is not None else "")
            if getattr(dec, "wants_input", False):
                src = self.inputs[self.rasters[0]]
                label = "input image" + (" (size only)" if _image_use(dec) == "size only" else "")
                self.edge(src, target, label, kind="image")
        if film is not None:
            if isinstance(dec, self.LateFusionDecoder):
                lv = "; ".join(f"{r} " + (", ".join(f"/{specs[i].stride}" for i in film.levels
                                                    if a <= i < b) or "none")
                               for r, (a, b) in zip(self.bb.names, dec.groups))
            else:
                lv = ", ".join(f"/{specs[i].stride}" for i in film.levels) or "none"
            out = "yes" if film.output is not None else "no"
            target.attrs.append(("FiLM", f"levels {lv}; output {out}", True))
            target.attrs.append(("FiLM params", _fmt_count(_nparams(film)), False))
        return target

    def _heads(self, feats) -> None:
        from .heads.heads import ImageHead, PixelHead

        m = self.model
        bb = self.bb
        deepest = m._deepest
        if isinstance(feats, Mapping):  # late fusion: which input owns the deepest level
            owner = next(r for r, (a, b) in zip(bb.names, bb.groups) if a <= deepest < b)
            enc_src = feats[owner]
        else:
            enc_src = feats
        dspec = bb.feature_specs[deepest]
        for name, head in m.heads.items():
            src = m._head_sources[name]
            kind = getattr(head, "kind", type(head).__name__)
            scope = getattr(head, "scope", "pixel")
            if isinstance(head, PixelHead):
                ch = head.conv.out_channels
            elif isinstance(head, ImageHead):
                ch = head.fc.out_features
            else:  # a registered custom head class
                ch = getattr(head, "out_channels", None)
            unc = getattr(head, "uncertainty", False)
            chs = (f"{ch // 2} ch + log-var ({ch} ch)" if unc and ch else
                   f"{ch} ch" if ch is not None else "custom")
            bias = self.cond is not None and name in self.cond.head_bias
            n = self.add(f"head_{name}", "head", name, f"{kind}, {scope}, {chs}",
                         modules=[head, self.cond.head_bias[name] if bias else None])
            if src == "decoder":
                feed, label = self.dec_out["decoder"], "decoder"
            elif src == "task":
                feed, label = self.dec_out[f"task_{name}"], f"decoder[{name}]"
            else:
                where = ("fused level" if self.multi and self.bb.mode == "intermediate"
                         else "encoder level")
                feed, label = enc_src, f"{where} {deepest} ({dspec.channels}@/{dspec.stride})"
            n.attrs.append(("from", label, True))
            if isinstance(head, PixelHead):
                n.attrs.append(("scale", _fmt_scale(head.factor), True))
                res = ("none" if head.up == 1 and head.down == 1 else
                       f"bilinear x{head.up}" if head.up > 1 else f"block mean /{head.down}")
                n.attrs.append(("resample", f"/{head.stride} feature, {res}", True))
                if head.hidden is not None:
                    n.attrs.append(("hidden", _fmt_hidden(head.hidden), True))
                if head.activation is not None:
                    n.attrs.append(("activation", head.activation, True))
            elif isinstance(head, ImageHead):
                n.attrs.append(("pooling", "global average -> linear", False))
            if unc:
                n.attrs.append(("uncertainty", "yes", True))
            if bias:
                n.attrs.append(("metadata", "head bias", True))
            if m.single_output is not None and name != m.single_output:
                n.attrs.append(("runs", f"no (single_output={m.single_output!r})", True))
            elif m.single_output is not None:
                n.attrs.append(("returned", "as the model's only output", True))
            self.edge(feed, n)
            got = self.cap.get(head)
            if got is not None:
                n.shapes.append(("raw", _fmt_shape(got[1])))
            out = self.output
            if isinstance(out, Mapping) and name in out:
                n.shapes.append(("out", _fmt_shape(out[name])))
            elif isinstance(out, Tensor) and name == m.single_output:
                n.shapes.append(("out", _fmt_shape(out)))

    def _conditioning_edges(self) -> None:
        c = self.cond
        if c is None:
            return
        for v in c.vectors:
            src = self.inputs[v]
            mode = c.modes[v]
            if mode == "film":
                for key in c.film:
                    if key == "image":  # FiLM on the encoder feature heads read directly
                        m = self.model
                        for h in m.heads:
                            if m._head_sources[h] == "encoder":
                                self.edge(src, self.nodes[f"head_{h}"], "FiLM",
                                          kind="condition")
                    else:
                        self.edge(src, self.dec_out[key], "FiLM", kind="condition")
            elif mode == "head":
                for h in c.head_bias:
                    self.edge(src, self.nodes[f"head_{h}"], "head bias", kind="condition")
            elif mode == "channels":
                t = c.channel_targets[v]
                self.edge(src, self.inputs[t], f"+{c.encoders[v].width} ch", kind="condition")

    def _stage_params(self) -> dict[str, int]:
        m, bb = self.model, self.bb
        enc = adapters = fusion = 0
        decs = [m.decoder, *m.task_decoders.values()]
        dec = heads = 0
        if self.early:
            enc = _nparams(bb.backbone)
        elif self.multi:
            enc = _nparams(*bb.encoders.values())
            adapters = _nparams(*bb.adapters.values())
            fusion = _nparams(*getattr(bb, "fusers", {}).values())
        else:
            enc = _nparams(bb)
        for d in decs:
            if isinstance(d, self.LateFusionDecoder):
                fusion += _nparams(d.fuse)
                dec += _nparams(*d.decoders)
            elif d is not None:
                dec += _nparams(d)
        heads = _nparams(*m.heads.values())
        out = {"adapters": adapters, "encoders": enc, "fusion": fusion,
               "conditioning": _nparams(self.cond), "decoders": dec, "heads": heads}
        return out

    def _summary(self) -> list[tuple[str, str]]:
        m = self.model
        s = [("params", _fmt_count(_nparams(m)))]
        if "" in self.flops:
            s.append(("FLOPs", f"{_fmt_count(self.flops[''])} for this sample"))
        if m.single_output is not None:
            s.append(("returns", f"Tensor (head {m.single_output!r})"))
        else:
            s.append(("returns", f"dict of {len(m.heads)} heads"))
        if self.multi or self.early:
            kind = "late" if self.late else "early" if self.early else "intermediate"
            s.append(("fusion", kind))
        fixed = getattr(self.bb, "fixed_size", None)
        if fixed is not None:
            s.append(("input", f"fixed size {fixed}, zero-padded"))
        elif m.pad_to_stride:
            s.append(("input", f"pad to multiple of {getattr(self.bb, 'input_divisor', 1)}"
                      if not self.multi else "padded per input to each encoder's divisor"))
        else:
            s.append(("input", "not padded; outputs resized"))
        return s


def model_graph(model: nn.Module, sample: Tensor | Mapping[str, Tensor] | None = None
                ) -> ModelGraph:
    """Stage-level graph of a :class:`~manyseg.models.DenseModel` (see :func:`describe`)."""
    for attr in ("backbone", "heads", "task_decoders", "_head_sources"):
        if not hasattr(model, attr):
            raise TypeError(f"describe() needs a DenseModel; {type(model).__name__} has no "
                            f"{attr!r}")
    return _Builder(model, sample).build()


def describe(model: nn.Module, sample: Tensor | Mapping[str, Tensor] | None = None,
             format: str = "text") -> str:
    """Describe a built model stage by stage.

    Args:
        model: A :class:`~manyseg.models.DenseModel`.
        sample: Optional input (tensor or dict, as for ``model(sample)``). If given, one
            ``no_grad`` forward pass in eval mode (train/eval flags are restored afterwards)
            annotates real tensor shapes at the stage boundaries and FLOPs per stage.
        format: ``"text"`` (indented tree), ``"mermaid"`` (flowchart) or ``"dot"`` (Graphviz).

    Returns:
        The description as a string. :func:`render_dot` turns a ``"dot"`` string into an image.
    """
    if format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; choose from {FORMATS}")
    return model_graph(model, sample).render(format)


def render_dot(source: str, path: str, format: str | None = None) -> str:
    """Render a ``describe(..., format="dot")`` string with Graphviz's ``dot`` binary.

    ``format`` defaults to the file extension (``png``, ``svg``, ``pdf``, ...). Returns
    ``path``. Raises ``RuntimeError`` if ``dot`` is not on ``PATH``.
    """
    exe = shutil.which("dot")
    if exe is None:
        raise RuntimeError("Graphviz 'dot' was not found on PATH; install Graphviz (e.g. "
                           "'apt install graphviz' or 'conda install graphviz'), or paste the "
                           "string into any Graphviz viewer")
    fmt = format or (path.rsplit(".", 1)[-1] if "." in path else "png")
    subprocess.run([exe, f"-T{fmt}", "-o", path], input=source.encode(), check=True)
    return path


def _channel_list(idx) -> str:
    """``[0, 4, 5, 6]`` -> ``"0, 4-6"``."""
    runs: list[list[int]] = []
    for i in sorted(idx):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _stem_mapping(enc, native: int) -> str:
    """Which pretrained filters each input channel of an adapted stem starts from: shows, for
    example, that ``repeat`` gives mask or elevation channels copies of the RGB filters."""
    c, method = enc.in_channels, enc.channel_adapter
    if method == "repeat":
        if c == 1:
            return f"the sum of the {native} pretrained filters"
        return f"ch i starts from pretrained filter i % {native}, x{native}/{c}"
    if method == "mean":
        return f"every ch starts from the mean pretrained filter, x{native}/{c}"
    keep = getattr(enc, "rgb_indices", None) or tuple(range(native))
    rest = [i for i in range(c) if i not in keep]
    init = "zero" if method == "rgb_plus_zero" else "random"
    return (f"ch {', '.join(map(str, keep))} keep the pretrained filters; "
            f"ch {_channel_list(rest)} start {init}")


# ---------------------------------------------------------------------------------------------
# renderers


def _params_note(n: Node) -> str:
    parts = []
    if n.params is not None:
        parts.append(f"{_fmt_count(n.params)} params")
    if n.flops is not None:
        parts.append(f"{_fmt_count(n.flops)} FLOPs")
    return ", ".join(parts)


def _to_text(g: ModelGraph) -> str:
    lines = ["DenseModel"]
    lines += [f"  {k}: {v}" for k, v in g.summary]
    sp = g.stage_params
    lines.append("  params by stage: " + ", ".join(f"{k} {_fmt_count(v)}" for k, v in sp.items()
                                                  if v))
    for stage in STAGES:
        nodes = g.by_stage(stage)
        if not nodes:
            continue
        lines.append(_STAGE_TITLES[stage])
        for n in nodes:
            note = _params_note(n)
            lines.append(f"  {n.name}: {n.title}" + (f"  [{note}]" if note else ""))
            for k, v, _ in n.attrs:
                lines.append(f"      {k}: {v}")
            for e in g.edges:
                if e.dst == n.id:
                    src = g.nodes[e.src]
                    tag = f" ({e.label})" if e.label else ""
                    via = "" if e.kind == "data" else f" [{e.kind}]"
                    lines.append(f"      <- {_heading(src)}{tag}{via}")
            for k, v in n.shapes:
                lines.append(f"      shape {k}: {v}")
    if g.pretrained_weights:
        lines.append("pretrained weights (fine-tuned weights keep these licences; cite when "
                     "publishing)")
        for e in g.pretrained_weights:
            lines.append(f"  {e['module']}: {e['provider']} {e['weights']} ({e['license']})")
            if e.get("citation"):
                lines.append(f"      cite: {e['citation']}")
            if e.get("source"):
                lines.append(f"      source: {e['source']}")
            if e.get("normalization"):
                lines.append(f"      pretrained on: {_normalization(e['normalization'])}")
    return "\n".join(lines) + "\n"


def _normalization(n: dict) -> str:
    """``(x - mean) / std`` with the values; one number when every channel shares it."""
    def vals(v) -> str:
        v = [f"{float(x):g}" for x in v]
        return v[0] if len(set(v)) == 1 else "(" + ", ".join(v) + ")"

    out = f"(x - {vals(n['mean'])}) / {vals(n['std'])}"
    if n.get("units"):
        out += f"; x = {n['units']}"
    return out + (", clipped to [0, 1]" if n.get("clip") else "")


def _diagram_lines(n: Node) -> list[str]:
    lines = [n.title]
    lines += [f"{k}: {v}" for k, v, brief in n.attrs if brief]
    note = _params_note(n)
    if note:
        lines.append(note)
    lines += [f"{k}: {v}" for k, v in n.shapes]
    return lines


def _wrap(s: str, width: int = 48) -> list[str]:
    """Break long values at "; " first, then at ", ", so diagram boxes stay narrow."""
    if len(s) <= width:
        return [s]
    out, cur = [], ""
    for chunk in re.split(r"(?<=;) ", s):
        parts = [chunk] if len(chunk) <= width else re.split(r"(?<=,) ", chunk)
        for part in parts:
            if cur and len(cur) + len(part) + 1 > width:
                out.append(cur)
                cur = "  " + part
            else:
                cur = f"{cur} {part}" if cur else part
    return out + [cur]


def _heading(n: Node) -> str:
    """Bold first line of a diagram box: the name, prefixed by its stage where that helps."""
    if n.name.startswith(n.stage) or n.name.startswith("fusion"):
        return n.name
    return f"{n.stage}: {n.name}"


def _dot_label(n: Node) -> str:
    rows = [f"<b>{html.escape(_heading(n))}</b>"]
    for line in _diagram_lines(n):
        rows += [html.escape(w) for w in _wrap(line)]
    return "<" + '<br align="left"/>'.join(rows) + '<br align="left"/>>'


def _q(s) -> str:
    """A DOT double-quoted string (user names may contain quotes or backslashes)."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _to_dot(g: ModelGraph) -> str:
    out = ["digraph manyseg {",
           '  rankdir=LR; nodesep=0.25; ranksep=0.5; newrank=true;',
           '  graph [fontname="Helvetica", fontsize=11, labeljust=l];',
           '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, '
           'margin="0.12,0.06"];',
           '  edge [fontname="Helvetica", fontsize=9, color="#5f6368"];']
    summary = _q("\n".join(f"{k}: {v}" for k, v in g.summary)).replace("\n", "\\l")
    out.append(f"  label={summary[:-1]}\\l\"; labelloc=t;")
    for n in g.nodes.values():
        color = _COLORS[n.style or n.stage]
        # Fusion boxes are square-cornered so they stand apart from encoders and decoders.
        style = '"filled"' if n.stage in ("fusion", "adapter") else '"rounded,filled"'
        out.append(f'  {_q(n.id)} [label={_dot_label(n)}, style={style}, fillcolor="{color}"];')
    inputs = [n.id for n in g.by_stage("input")]
    out.append("  { rank=same; " + " ".join(f"{_q(i)};" for i in inputs) + " }")
    heads = [n.id for n in g.by_stage("head")]
    out.append("  { rank=same; " + " ".join(f"{_q(i)};" for i in heads) + " }")
    for e in g.edges:
        attrs = []
        if e.label:
            attrs.append(f"label={_q(e.label)}")
        if e.kind == "condition":
            attrs += ['style=dashed', 'color="#8e24aa"', 'fontcolor="#8e24aa"',
                      'constraint=false' if g.nodes[e.dst].stage == "input" else ""]
        elif e.kind == "image":
            attrs += ['style=dotted', 'color="#1a73e8"', 'fontcolor="#1a73e8"']
        attrs = [a for a in attrs if a]
        out.append(f"  {_q(e.src)} -> {_q(e.dst)}" + (f" [{', '.join(attrs)}]" if attrs else "")
                   + ";")
    out.append("}")
    return "\n".join(out) + "\n"


def _mermaid_text(s: str) -> str:
    # Mermaid entity codes (#quot; ...) rather than HTML ones, which some renderers show raw.
    return (s.replace("->", "→").replace('"', "#quot;").replace("<", "#lt;")
            .replace(">", "#gt;").replace("|", "/"))


_MERMAID_SHAPES = {"input": ('(["', '"])'), "adapter": ('["', '"]'), "encoder": ('["', '"]'),
                   "fusion": ('{{"', '"}}'), "decoder": ('["', '"]'), "head": ('("', '")')}


def _to_mermaid(g: ModelGraph) -> str:
    out = ["flowchart LR"]
    for n in g.nodes.values():
        a, b = _MERMAID_SHAPES[n.stage]
        rows = [f"<b>{_mermaid_text(_heading(n))}</b>"]
        for line in _diagram_lines(n):
            rows += [_mermaid_text(w.strip()) for w in _wrap(line)]
        out.append(f"  {n.id}{a}{'<br/>'.join(rows)}{b}")
    for e in g.edges:
        label = _mermaid_text(e.label)
        arrow = {"data": "-->", "condition": "-.->", "image": "-.->"}[e.kind]
        out.append(f"  {e.src} {arrow}" + (f"|{label}|" if label else "") + f" {e.dst}")
    used = {n.style or n.stage for n in g.nodes.values()}
    for key, color in _COLORS.items():
        if key in used:
            out.append(f"  classDef {key} fill:{color},stroke:#5f6368")
    for n in g.nodes.values():
        out.append(f"  class {n.id} {n.style or n.stage}")
    return "\n".join(out) + "\n"


__all__ = ["FORMATS", "Edge", "ModelGraph", "Node", "describe", "model_graph", "render_dot"]
