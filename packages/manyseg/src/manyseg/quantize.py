"""Quantisation entry points (deployment is kept separate from model construction).

Two routes, both through torchao (``pip install 'manyseg[quant]'``):

* ``quantize_pt2e``: static int8 for CPU deployment. ``torch.export`` (with
  ``export_model``'s dynamic H/W) → ``prepare_pt2e`` → calibration → ``convert_pt2e`` →
  ``torch.export`` again, so the result is an ``ExportedProgram`` like ``export_model``'s.
  Weights and activations are int8; ``torch.compile`` lowers the graph to oneDNN int8 kernels.
* ``quantize_weights``: int8 weights (optionally int8 dynamic activations for Linear layers)
  on the eager ``nn.Module``, for CPU or GPU, no calibration and no export.

``model_size`` and ``compare_outputs`` report what the conversion saved and what it cost.

This is a module of its own rather than part of ``manyseg.export`` because it carries an
optional dependency and because ``quantize_weights`` does not export at all. The PT2E path
reuses ``manyseg.export.dynamic_shapes`` rather than duplicating it.

Both save through :func:`manyseg.save` and load through :func:`manyseg.load`, with their metadata
and ``tiling_spec()``. A ``.pt2`` file is written without pickled data. ``torch.export.load`` can
run code from a crafted archive in two ways: it unpickles sample inputs (retrying with
``weights_only=False`` if the safe load fails) and entries marked as pickled, and it passes the
graph's symbolic-shape strings to ``sympy.sympify``, which evaluates them. :func:`manyseg.load`
checks for both before calling it, and allows only known graph operators. That narrows the risk
but is no sandbox, and the program runs when called: only load ``.pt2`` files from sources you
trust.
"""

from __future__ import annotations

import copy
import io
import json
import re
import warnings
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .export import dynamic_shapes, eval_mode

WEIGHT_MODES = ("int8_weight_only", "int8_dynamic")


def _require_torchao(what: str = "quantisation") -> None:
    try:
        import torchao  # noqa: F401
    except ImportError as e:
        raise ImportError(f"{what} needs torchao: pip install 'manyseg[quant]'") from e


# --- PT2E: static int8, exported -----------------------------------------------------------

def _has_vnni() -> bool:
    """Whether this CPU has VNNI; unknown counts as no (``reduce_range`` is then the safe
    choice)."""
    probe = getattr(torch.cpu, "_is_vnni_supported", None)
    try:
        return bool(probe()) if probe is not None else False
    except Exception:
        return False


def _x86_quantizer(reduce_range: bool | None = None):
    """torchao's x86 quantizer. ``reduce_range`` (7-bit activations) defaults to "this CPU has
    no VNNI": there, inductor's u8 x s8 kernels saturate their 16-bit intermediate sums, and
    in E008 the compiled int8 model lost ~6 mIoU on Oxford Pet (AMD Zen 3; the eager
    reference was fine). torchao's own docs give the same advice."""
    from torchao.quantization.pt2e.quantizer import x86_inductor_quantizer as xiq

    rr = (not _has_vnni()) if reduce_range is None else bool(reduce_range)
    cfg = xiq.get_default_x86_inductor_quantization_config(reduce_range=rr)
    return xiq.X86InductorQuantizer().set_global(cfg)


# timm's window-attention classes, by (module, class): their dynamic-size export hangs rather
# than fails (``architecture_review.md``). Matched exactly so that an unrelated module with a
# ``window_size`` attribute is not rejected.
_WINDOW_ATTENTION = {
    ("timm.models.swin_transformer", "WindowAttention"),
    ("timm.models.swin_transformer_v2", "WindowAttention"),
    ("timm.models.swin_transformer_v2_cr", "WindowMultiHeadAttention"),
    ("timm.models.davit", "WindowAttention"),
}


def _windowed(model: nn.Module) -> str | None:
    """Class name of the first timm window-attention module (Swin, Swin V2, DaViT), if any."""
    return next((type(m).__name__ for m in model.modules()
                 if (type(m).__module__, type(m).__name__) in _WINDOW_ATTENTION), None)


def _shapes(model: nn.Module, example, dynamic: bool, max_size: int, dynamic_batch: bool):
    """``export_model``'s dynamic-shape spec, or None (fixed size). Computed once from the float
    model: the quantised graph is re-exported with the same spec."""
    if not dynamic:
        return None
    if (block := _windowed(model)) is not None:
        raise ValueError(f"window-attention backbones ({block}) cannot be exported with dynamic "
                         "H/W; pass dynamic=False to quantise at the example's fixed size")
    return dynamic_shapes(model, example, max_size, dynamic_batch)


def _export(module: nn.Module, example, shapes) -> torch.export.ExportedProgram:
    """``torch.export`` as ``export_model`` does it, for the float model and the int8 graph."""
    try:
        return torch.export.export(module, (example,), dynamic_shapes=shapes,
                                   prefer_deferred_runtime_asserts_over_guards=True)
    except Exception as e:  # torch.export raises several types (UserError, RuntimeError, ...)
        if shapes is None:
            raise
        raise ValueError(
            "could not export this model with dynamic H/W; pass dynamic=False to quantise at "
            f"the example's fixed size. Export error: {type(e).__name__}: {e}") from e


def _drop_unused_attrs(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """``convert_pt2e`` adds int8 copies of the weights but leaves the float originals
    registered (unused); drop every parameter and buffer the graph no longer reads. An attribute
    is read if a ``get_attr`` or ``call_module`` target is it or a dotted prefix of it (e.g.
    ``body`` for ``body.weight``)."""
    targets = {n.target for n in gm.graph.nodes if n.op in ("get_attr", "call_module")}

    def used(name: str) -> bool:
        parts = name.split(".")
        return any(".".join(parts[:i]) in targets for i in range(1, len(parts) + 1))

    for name in [n for n, _ in [*gm.named_parameters(), *gm.named_buffers()] if not used(n)]:
        owner, _, leaf = name.rpartition(".")
        mod = gm.get_submodule(owner)
        mod._parameters.pop(leaf, None)
        mod._buffers.pop(leaf, None)
    gm.delete_all_unused_submodules()
    return gm


def quantize_pt2e(model: nn.Module, example, calibration: Iterable, *, quantizer=None,
                  dynamic: bool = True, max_size: int = 4096,
                  dynamic_batch: bool = False,
                  reduce_range: bool | None = None) -> torch.export.ExportedProgram:
    """Static int8 (weights and activations) by PT2E post-training quantisation.

    ``example`` is a Tensor or, for multi-input models, a ``dict[str, Tensor]``, exactly as the
    model takes it. ``calibration`` is an iterable of inputs of the same form (e.g. a few dozen
    representative batches); activation ranges are observed on it. ``quantizer`` defaults to
    torchao's ``X86InductorQuantizer`` (per-channel weights, per-tensor activations); pass any
    other PT2E quantizer (e.g. ``ArmInductorQuantizer``) to target another backend.
    ``reduce_range`` (default: on when *this* CPU lacks VNNI) uses 7-bit activations, which
    is needed for correct compiled int8 on such CPUs; set it for the CPU you deploy on.

    H/W stay dynamic as in ``export_model`` (and the batch too with ``dynamic_batch``, which
    needs an example with batch size >= 2). ``model`` is exported in eval mode and then
    returned to its previous mode.
    Window-attention backbones (Swin, DaViT) cannot export dynamically and need
    ``dynamic=False``: the result then only accepts the example's size.

    Returns an ``ExportedProgram``, like ``export_model``: save it with its metadata by
    ``manyseg.save(ep, "model.pt2", source=model, ...)``, or run ``ep.module()``. Eager calls
    run int8 as quantise/dequantise pairs in float, which shows the accuracy cost but is not
    faster; for speed, compile it with inductor freezing::

        torch._inductor.config.freezing = True
        fast = torch.compile(ep.module())
    """
    _require_torchao()
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    shapes = _shapes(model, example, dynamic, max_size, dynamic_batch)
    with eval_mode(model):
        gm = _export(model, example, shapes).module()
    rr = None  # unknown with a custom quantizer
    if quantizer is None:
        rr = (not _has_vnni()) if reduce_range is None else bool(reduce_range)
        if not rr:
            warnings.warn(
                "this int8 program uses full-range activations, which are only accurate on CPUs "
                "with VNNI (e.g. Intel Cascade Lake or newer, AMD Zen 4 or newer); on others, "
                "compiled int8 kernels saturate silently (-6.5 mIoU on Oxford Pet in E008). To "
                "deploy on any x86 CPU, quantise with reduce_range=True (~0.1 mIoU). "
                "manyseg.quantize.check_deployment(ep) verifies the deployment CPU.",
                UserWarning, stacklevel=2)
    prepared = prepare_pt2e(gm, quantizer if quantizer is not None else _x86_quantizer(rr))
    n, bs = 0, _batch_of(example)
    with torch.no_grad():
        for x in calibration:
            if not dynamic_batch and _batch_of(x) != bs:
                raise ValueError(f"calibration batch size {_batch_of(x)} differs from the "
                                 f"example's {bs}; the export fixes the batch dimension unless "
                                 "dynamic_batch=True")
            prepared(x)
            n += 1
    if n == 0:
        raise ValueError("calibration must yield at least one input")
    converted = _drop_unused_attrs(convert_pt2e(prepared))
    if rr is not None:  # survives torch.export.save / load; read by check_deployment
        converted.register_buffer(_RR_KEY, torch.tensor(rr))
    return _export(converted, example, shapes)


_RR_KEY = "manyseg_int8_reduce_range"


def check_deployment(ep: torch.export.ExportedProgram) -> None:
    """Check that a ``quantize_pt2e`` program is accurate on *this* CPU: raises if it was
    quantised with full-range activations (``reduce_range=False``) and this CPU lacks VNNI, where
    compiled int8 kernels saturate silently (E008). Call it in the deployment script, e.g. after
    ``torch.export.load``; takes a loaded :class:`QuantizedProgram` too. Warns if the program
    carries no record (not produced by ``quantize_pt2e``'s default quantizer)."""
    ep = ep.program if isinstance(ep, QuantizedProgram) else ep
    rec = ep.state_dict.get(_RR_KEY)
    if rec is None:
        warnings.warn("program was not produced by manyseg.quantize.quantize_pt2e with the "
                      "default quantizer; cannot check its activation range", UserWarning,
                      stacklevel=2)
        return
    if not bool(rec) and not _has_vnni():
        raise RuntimeError(
            "this int8 program was quantised with full-range activations, but this CPU has no "
            "VNNI: compiled int8 results will be silently wrong. Re-quantise with "
            "quantize_pt2e(..., reduce_range=True).")


# --- saved programs ------------------------------------------------------------------------

class QuantizedProgram:
    """A ``quantize_pt2e`` program loaded by :func:`manyseg.load`, with its metadata.

    Deliberately not an ``nn.Module``: a module from ``ep.module()`` raises on ``.eval()``, also
    when nested, and callers such as ManyPred call ``.to(device, dtype).eval()`` on every module.
    Call it like the model; its device is the ``map_location`` it was loaded on.

    Attributes:
        program: The ``ExportedProgram``.
        module: Its ``ep.module()``, e.g. to ``torch.compile`` with freezing.
        saved_metadata: What :func:`manyseg.save` stored.
        batch_size: The only batch size the program takes, or None if it is dynamic
            (``quantize_pt2e(..., dynamic_batch=True)``).
        fixed_size: The only ``(h, w)`` it takes, or None if H/W are dynamic (``dynamic=False``,
            needed by window-attention backbones).
        input_shapes: ``{input: shape}`` with None for dynamic dimensions (``"x"`` for a model
            taking a single tensor).
    """

    def __init__(self, program: torch.export.ExportedProgram, metadata: dict,
                 device: str | torch.device = "cpu"):
        recipe = metadata["quantization"]
        self.program = program
        self.saved_metadata = metadata
        self.device = torch.device(device)
        self.module = program.module().to(self.device)
        self.batch_size: int | None = recipe["batch_size"]
        self.fixed_size: tuple[int, int] | None = (tuple(recipe["fixed_size"])
                                                   if recipe["fixed_size"] else None)
        self.input_shapes: dict[str, list] = recipe["input_shapes"]

    def tiling_spec(self) -> dict:
        """The float model's ``tiling_spec()`` as saved, for ManyPred. ``max_size`` is the
        program's fixed size when it has one."""
        spec = copy.deepcopy(self.saved_metadata["tiling_spec"])
        if spec.get("max_size") is not None:
            spec["max_size"] = tuple(spec["max_size"])
        return spec

    def _check(self, name: str, t: Tensor) -> None:
        if t.device.type != self.device.type:
            raise ValueError(f"This program is on {self.device} but input {name!r} is on "
                             f"{t.device}: load it with map_location={str(t.device)!r}, or run "
                             f"it on {self.device} (with ManyPred, device={str(self.device)!r}).")
        want = self.input_shapes.get(name)
        if want is None:
            return
        if len(want) != t.ndim:
            raise ValueError(f"Input {name!r} must have {len(want)} dimensions, got {t.ndim}.")
        if want[0] is not None and t.shape[0] != want[0]:
            raise ValueError(
                f"This program was exported for batch size {want[0]}, got {t.shape[0]} for "
                f"input {name!r}. Run it with that batch size (with ManyPred, "
                f"batch_size={want[0]}), or quantise with dynamic_batch=True.")
        for i, (w, have) in enumerate(zip(want, t.shape, strict=True)):
            if w is not None and w != have:
                raise ValueError(
                    f"This program was exported at a fixed input shape {want} (None: dynamic) "
                    f"and input {name!r} has {list(t.shape)} (dimension {i}). Quantise with "
                    f"dynamic=True where the backbone allows it, or use this size.")

    def __call__(self, x: Any) -> Any:
        for name, t in x.items() if isinstance(x, Mapping) else [("x", x)]:
            self._check(name, t)
        return self.module(x)

    def __repr__(self) -> str:
        return (f"QuantizedProgram(batch_size={self.batch_size}, fixed_size={self.fixed_size}, "
                f"device={self.device})")


def _input_shapes(ep: torch.export.ExportedProgram) -> dict[str, list]:
    """``{input: shape}`` of the program's inputs, None for its dynamic dimensions, read from
    the graph. Dict inputs by key; a single tensor is ``"x"``."""
    from torch.utils._pytree import tree_flatten_with_path

    paths, _ = tree_flatten_with_path(ep.example_inputs)
    names = [str(getattr(path[-1], "key", "x")) if len(path) > 2 else "x" for path, _ in paths]
    wanted = {s.arg.name for s in ep.graph_signature.input_specs
              if s.kind == torch.export.graph_signature.InputKind.USER_INPUT}
    nodes = [n for n in ep.graph.nodes if n.op == "placeholder" and n.name in wanted]
    return {name: [d if isinstance(d, int) else None for d in n.meta["val"].shape]
            for name, n in zip(names, nodes, strict=True)}


def _fixed_size(shapes: Mapping[str, list]) -> list[int] | None:
    """``(h, w)`` of the first raster input if the program fixes it."""
    raster = next((s for s in shapes.values() if len(s) >= 4), None)
    return raster[-2:] if raster is not None and None not in raster[-2:] else None


def _is_quantized_program(ep: torch.export.ExportedProgram) -> bool:
    return any(n.op == "call_function" and "quantize_per" in str(n.target)
               for n in ep.graph.nodes)


def _check_source(ep: torch.export.ExportedProgram, source: nn.Module) -> None:
    """The metadata comes from ``source``, so check it is the same model: the program's int8
    weights have shapes the model has, its outputs have the same names and shapes on the program's
    example, and they correlate (a warning: a poor calibration lowers it too, but other weights
    of the same architecture show up nowhere else)."""
    own = {tuple(p.shape) for p in source.parameters()}
    other = {tuple(t.shape) for t in ep.state_dict.values()
             if t.dtype == torch.int8 and t.ndim >= 2} - own
    if other:
        raise ValueError(f"source= does not look like the model this program came from: it has "
                         f"no weights of shape {sorted(other)[:3]}.")
    device = next(source.parameters()).device
    args = ep.example_inputs[0]
    moved = torch.utils._pytree.tree_map(lambda t: t.to(device), args)

    def shape_of(out):
        return ({k: tuple(v.shape) for k, v in out.items()} if isinstance(out, Mapping)
                else tuple(out.shape))

    def flat(out):
        parts = out.values() if isinstance(out, Mapping) else [out]
        return torch.cat([p.detach().cpu().float().flatten() for p in parts])

    with torch.no_grad(), eval_mode(source):
        want = source(*moved)
    with torch.no_grad():
        have = ep.module()(*args)
    if shape_of(want) != shape_of(have):
        raise ValueError(f"source= does not look like the model this program came from: on its "
                         f"example the program returns {shape_of(have)}, source returns "
                         f"{shape_of(want)}.")
    r = torch.corrcoef(torch.stack([flat(want), flat(have)]))[0, 1]
    if not r > 0.5:
        warnings.warn(f"source= may not be the model this program came from: their outputs "
                      f"correlate at only {r:.2f} on the program's example. Its metadata will be "
                      f"stored with the program.", UserWarning, stacklevel=4)


def _save_program(ep: torch.export.ExportedProgram, source: nn.Module, path: Path,
                  meta: dict) -> Path:
    """Write ``ep`` and ``meta`` to a ``.pt2`` file holding no pickled data."""
    from .io import _KEY, QUANTIZED_FORMAT

    _require_torchao("saving a quantised program")
    import torchao

    if path.suffix != ".pt2":
        raise ValueError(f"A quantised program saves to a .pt2 file, got {path.name!r}.")
    if not _is_quantized_program(ep):
        raise ValueError("This ExportedProgram is not quantised: save takes programs from "
                         "manyseg.quantize.quantize_pt2e. For a float model, save the model.")
    _check_source(ep, source)
    shapes = _input_shapes(ep)
    fixed = _fixed_size(shapes)
    first = next(iter(shapes.values()))
    rr = ep.state_dict.get(_RR_KEY)
    meta["format"] = QUANTIZED_FORMAT
    meta["quantization"] = {
        "route": "pt2e",
        "batch_size": first[0],
        "fixed_size": fixed,
        "input_shapes": shapes,
        "reduce_range": None if rr is None else bool(rr),
        "torchao_version": torchao.__version__,
        "torch_version": torch.__version__,
    }
    meta["tiling_spec"] = _saved_spec(source, meta, fixed)
    # the sample inputs are the one thing torch.export.save always pickles, and running the
    # program doesn't need them
    stripped = copy.copy(ep)
    stripped._example_inputs = None
    buf = io.BytesIO()
    try:
        torch.export.save(stripped, buf, extra_files={f"{_KEY}.json": json.dumps(meta)})
    except Exception as e:  # torch's SerializeError and the assertions under it
        hint = (" Its dynamic H/W use an operation torch can't serialise: quantise with "
                "dynamic=False to save it at a fixed size." if fixed is None else "")
        raise ValueError(f"torch.export.save can't write this program.{hint} "
                         f"({type(e).__name__}: {str(e).splitlines()[0]})") from e
    data = buf.getvalue()
    _check_archive(io.BytesIO(data), "This program")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _saved_spec(source: nn.Module, meta: dict, fixed: list[int] | None) -> dict:
    """``source.tiling_spec()`` as it will be once ``meta`` is stored, as plain data."""
    had = getattr(source, "saved_metadata", None)
    source.saved_metadata = {k: meta[k] for k in ("activations", "normalize", "classes", "bands")}
    try:
        spec = source.tiling_spec()
    finally:
        source.saved_metadata = had
    if fixed is not None:
        spec["max_size"] = fixed
        # ManyPred rounds a patch down to pad_multiple: that must not leave the fixed size
        if any(size % spec["pad_multiple"] for size in fixed):
            spec["pad_multiple"] = 1
    return json.loads(json.dumps(spec))


# Everything a ManySeg .pt2 archive may hold. Exact on purpose: other entries (legacy
# ``data/weights/model.pt``, AOTInductor packages, custom and opaque objects) reach torch.load or
# pickle.loads in torch.export.load.
_ARCHIVE_ENTRIES = re.compile(
    r"(archive_format|archive_version|byteorder|\.data/version|\.data/serialization_id"
    r"|models/model\.json|data/weights/model_weights_config\.json"
    r"|data/constants/model_constants_config\.json|data/weights/weight_\d+"
    r"|data/constants/tensor_\w+|data/sample_inputs/model\.pt|extra/manyseg\.json)")


# What the symbolic-shape strings in a graph may be made of. sympify evaluates them, so a call
# must name one of these (sympy's srepr forms, and torch's own symbolic functions), a Symbol must
# have exactly the form export writes, and nothing may reach attributes or call a result.
_SYMPY_CALLS = frozenset({
    "Integer", "Rational", "Float", "Add", "Mul", "Pow", "Max", "Min", "Mod", "Abs", "floor",
    "ceiling", "And", "Or", "Not", "Equality", "Unequality", "StrictLessThan", "LessThan",
    "StrictGreaterThan", "GreaterThan", "FloorDiv", "ModularIndexing", "Where", "PythonMod",
    "CleanDiv", "CeilToInt", "FloorToInt", "CeilDiv", "LShift", "RShift", "PowByNatural",
    "FloatPow", "FloatTrueDiv", "IntTrueDiv", "IsNonOverlappingAndDenseIndicator", "TruncToFloat",
    "TruncToInt", "RoundToInt", "RoundDecimal", "ToFloat"})
_SYMBOL = re.compile(r"Symbol\('[A-Za-z_]\w*'(, (positive|integer)=(True|False))*\)")
_EXPRESSION = re.compile(r"[\w\s+\-*/%<>=!.,()]*")
_CALL = re.compile(r"([A-Za-z_]\w*)\s*\(")


def _safe_expression(expr: str) -> bool:
    plain = _SYMBOL.sub("0", expr)
    return (_EXPRESSION.fullmatch(plain) is not None and "__" not in plain
            and ")(" not in plain.replace(" ", "")
            and re.search(r"\.\s*[A-Za-z_(]|[A-Za-z_)]\s*\.", plain) is None
            and all(name in _SYMPY_CALLS for name in _CALL.findall(plain)))


# What a graph node may call: PyTorch operators and the arithmetic of symbolic shapes. Others,
# such as "#namespace:op" extension operators, run Python while loading or when called.
_OPERATORS = "add|sub|mul|floordiv|truediv|mod|pow|neg|lt|le|gt|ge|eq|ne|and_|or_|not_"
_TARGET = re.compile(rf"torch\.ops\.(?!__)\w+\.\w+(\.\w+)?|_operator\.({_OPERATORS})"
                     r"|torch\.sym_\w+|math\.(ceil|floor|trunc)")


def _strings_under(node: Any, name: str):
    """Every value under the key ``name`` in a serialised graph."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == name and isinstance(value, str):
                yield value
            elif key == name and not isinstance(value, (dict, list)):
                yield str(value)
            else:
                yield from _strings_under(value, name)
    elif isinstance(node, list):
        for value in node:
            yield from _strings_under(value, name)


def _declared_size(entry: Mapping) -> int | None:
    """The elements a payload entry declares, or None if its sizes aren't plain integers."""
    total = 1
    for size in entry["tensor_meta"]["sizes"]:
        value = size.get("as_int") if isinstance(size, Mapping) else None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        total *= value
    return total


def _check_archive(file: Any, label: Any) -> None:
    """Raise ValueError unless ``torch.export.load`` would neither unpickle anything from the
    archive nor evaluate anything but a plain symbolic shape. Run before calling it."""
    def refuse(why: str):
        raise ValueError(
            f"{label} holds or may hold content that can run code when loaded ({why}), so it "
            f"was refused. If you wrote it or trust its source, use torch.export.load directly.")

    with zipfile.ZipFile(file) as z:
        names = z.namelist()
        if len(set(names)) != len(names):
            refuse("duplicate entries")
        roots = {n.split("/", 1)[0] for n in names}
        if len(roots) != 1 or any("/" not in n for n in names):
            refuse("an unexpected layout")
        root = roots.pop()
        local = {n[len(root) + 1:]: n for n in names}
        for name in local:
            if _ARCHIVE_ENTRIES.fullmatch(name) is None:
                refuse(f"entry {name!r}")
        if z.getinfo(local["data/sample_inputs/model.pt"]).file_size if (
                "data/sample_inputs/model.pt" in local) else 0:
            refuse("sample inputs")
        for name in ("data/weights/model_weights_config.json",
                     "data/constants/model_constants_config.json"):
            if name not in local:
                refuse(f"no {name}")
            for fqn, entry in json.loads(z.read(local[name]))["config"].items():
                path = str(entry.get("path_name"))
                if entry.get("use_pickle") is not False or not re.fullmatch(
                        r"(weight|tensor)_\w+", path):
                    refuse(f"entry {fqn!r}")
                # torch allocates what a zero-byte entry declares, so it must fit the bytes there
                folder = name.rsplit("/", 1)[0]
                try:
                    bytes_there = z.getinfo(local[f"{folder}/{path}"]).file_size
                    size = _declared_size(entry)
                    device = entry["tensor_meta"]["device"]["type"]
                except (KeyError, TypeError):
                    refuse(f"malformed entry {fqn!r}")
                if device != "cpu":
                    refuse(f"entry {fqn!r} on {device}; save the program from the CPU")
                if size is None or size > bytes_there:
                    refuse(f"entry {fqn!r} declares more than it holds")
        if "models/model.json" not in local:
            refuse("no models/model.json")
        graph = json.loads(z.read(local["models/model.json"]))
        for expr in _strings_under(graph, "expr_str"):
            if not _safe_expression(expr):
                refuse(f"graph expression {expr[:80]!r}")
        for target in _strings_under(graph, "target"):
            if _TARGET.fullmatch(target) is None:
                refuse(f"graph operator {target[:80]!r}")


def _load_program(path: str | Path, meta: dict, device: str | torch.device) -> QuantizedProgram:
    _check_archive(path, path)
    program = torch.export.load(path)
    if meta["quantization"].get("reduce_range") is False and not _has_vnni():
        warnings.warn(
            f"{path} was quantised with full-range activations and this CPU has no VNNI: "
            "torch.compile'd int8 kernels will saturate silently (eager calls are fine). "
            "Re-quantise with quantize_pt2e(..., reduce_range=True).", UserWarning, stacklevel=3)
    return QuantizedProgram(program, meta, device)


def _batch_of(x) -> int:
    if isinstance(x, Mapping):
        x = next(v for v in x.values() if torch.is_tensor(v))
    return int(x.shape[0])


# --- torchao: weight-only / dynamic, eager -------------------------------------------------

def _is_linear(mod: nn.Module, fqn: str) -> bool:
    """Linear with more than one input feature. torchao fails on ``nn.Linear(1, n)`` (e.g. a
    one-feature metadata input), whose weights are tiny, so they stay float."""
    return isinstance(mod, nn.Linear) and mod.weight.shape[1] > 1


def _is_dense_conv(mod: nn.Module, fqn: str) -> bool:
    """Conv2d with more than one input channel per group. Depthwise weights (C, 1, k, k) are
    tiny and torchao's int8 conv weights reject them, so they stay float."""
    return isinstance(mod, nn.Conv2d) and mod.weight.shape[1] > 1


def quantize_weights(model: nn.Module, mode: str = "int8_weight_only", *, conv: bool = True,
                     inplace: bool = False) -> nn.Module:
    """int8 weights through torchao, for eager or ``torch.compile`` inference on CPU or GPU.

    ``mode``:

    * ``"int8_weight_only"``: Linear and Conv2d weights stored int8 (symmetric, per output
      channel), activations stay float. About 4x smaller; on GPU torchao's int8 matmul makes
      Linear layers faster under ``torch.compile``.
    * ``"int8_dynamic"``: Linear layers also quantise their activations to int8 at run time.
      torchao has no dynamic-activation conv, so Conv2d weights are int8 weight-only.

    Conv2d weights are dequantised on the fly, so for CNNs this saves memory and disk, not
    time. ``conv=False`` leaves convs in float (Linear only). Depthwise convs and one-feature
    Linear layers always stay float.
    The model is copied first unless ``inplace``.

    :func:`manyseg.save` stores the result with its metadata, keeping the size saving, and
    :func:`manyseg.load` gives it back with int8 weights.
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got {mode!r}")
    _require_torchao()
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        IntxWeightOnlyConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerAxis

    if not inplace:
        model = copy.deepcopy(model)
    if conv:
        quantize_(model, IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
                  filter_fn=_is_dense_conv)
    linear = (Int8WeightOnlyConfig() if mode == "int8_weight_only"
              else Int8DynamicActivationInt8WeightConfig())
    quantize_(model, linear, filter_fn=_is_linear)
    model.quantization = {"mode": mode, "conv": conv}  # what manyseg.load needs to rebuild it
    return model


# --- reports -------------------------------------------------------------------------------

def model_size(model: nn.Module | torch.export.ExportedProgram | QuantizedProgram) -> int:
    """Serialized size in bytes: ``torch.export.save`` for an ``ExportedProgram`` (graph and
    weights, as deployed), ``torch.save(model.state_dict())`` for a module (weights only)."""
    model = model.program if isinstance(model, QuantizedProgram) else model
    buf = io.BytesIO()
    if isinstance(model, torch.export.ExportedProgram):
        torch.export.save(model, buf)
    else:
        torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


class _Diff:
    """Running agreement statistics for one output."""

    def __init__(self):
        self.max, self.abs_sum, self.n, self.agree, self.count = 0.0, 0.0, 0, 0, 0
        self.kind: str | None = None  # "argmax" (float logits), "exact" (integer/bool) or None
        self.dtype_changed = False

    def add(self, ref: Tensor, out: Tensor) -> None:
        out = out.to(ref.device)
        self.dtype_changed |= ref.dtype != out.dtype
        d = (ref.double() - out.double()).abs()
        self.max = max(self.max, d.max().item())
        self.abs_sum += d.sum().item()
        self.n += d.numel()
        if not ref.is_floating_point():  # label maps, counts: compare exactly
            self.kind, agree = "exact", ref == out
        elif ref.ndim in (2, 4) and ref.shape[1] > 1:  # (B, C) or (B, C, H, W) logits
            self.kind, agree = "argmax", ref.argmax(1) == out.argmax(1)
        else:
            return
        self.agree += agree.sum().item()
        self.count += agree.numel()

    def report(self, prefix: str = "") -> dict[str, float]:
        r = {f"{prefix}max_abs_diff": self.max, f"{prefix}mean_abs_diff": self.abs_sum / self.n}
        if self.kind is not None:
            r[f"{prefix}{self.kind}_agreement"] = self.agree / self.count
        if self.dtype_changed:
            r[f"{prefix}dtype_changed"] = 1.0
        return r


def compare_outputs(reference, candidate, inputs: Iterable) -> dict[str, float]:
    """Numerical agreement of ``candidate`` (e.g. a quantised model) with ``reference`` over
    ``inputs`` (an iterable of model inputs, Tensor or dict). Either model may be an
    ``nn.Module``, an ``ExportedProgram`` or a loaded :class:`QuantizedProgram`.

    Returns ``max_abs_diff`` and ``mean_abs_diff`` of the raw outputs, plus:

    * ``argmax_agreement`` for floating outputs shaped like logits, ``(B, C)`` or
      ``(B, C, H, W)`` with ``C > 1``: the fraction of images or pixels whose argmax over dim 1
      is unchanged;
    * ``exact_agreement`` for integer or bool outputs (e.g. label maps): the fraction of equal
      elements;
    * ``dtype_changed`` = 1.0, with a warning, if the candidate's dtype differs.

    For dict outputs every key is prefixed with the head name, e.g.
    ``"segmentation/argmax_agreement"``.
    """
    reference, candidate = (m.module() if isinstance(m, torch.export.ExportedProgram) else m
                            for m in (reference, candidate))
    diffs: dict[str, _Diff] = {}
    with torch.no_grad():
        for x in inputs:
            ref, out = reference(x), candidate(x)
            if isinstance(ref, Mapping):
                for k, v in ref.items():
                    diffs.setdefault(f"{k}/", _Diff()).add(v, out[k])
            else:
                diffs.setdefault("", _Diff()).add(ref, out)
    if not diffs:
        raise ValueError("compare_outputs needs at least one input")
    if changed := [p.rstrip("/") or "output" for p, d in diffs.items() if d.dtype_changed]:
        warnings.warn(f"candidate output dtype differs from the reference for: {changed}",
                      stacklevel=2)
    return {k: v for prefix, d in diffs.items() for k, v in d.report(prefix).items()}
