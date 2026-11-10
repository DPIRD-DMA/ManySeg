"""Save a trained model with what inference needs, and load it back.

``save(model, path, ...)`` writes one ``.safetensors`` file: the weights, plus JSON metadata with
the ``create_model`` arguments the model was built with, its heads, and what inference needs to
know that the weights don't say: how inputs were normalised, each head's activation, class
names and band names. It also records the pretrained weights the model started from, with their
licences and citations (``pretrained_weights``, see :func:`manyseg.weights.attribution`):
fine-tuned weights are still subject to them, and most require attribution. ``load(path)``
rebuilds the model (without downloading pretrained weights) and loads the weights; its
:meth:`~manyseg.DenseModel.tiling_spec` then reports the stored normalisation and activations,
so ManyPred uses them without being told.

Compatibility policy. The file stores a save ``format`` number and the ``manyseg_version`` that
wrote it, and holds only plain data (no pickled classes), so it survives refactors that move or
rename modules. A change that renames or reinterprets ``create_model`` arguments, modules or
parameters bumps ``FORMAT`` and adds a migration that :func:`read_metadata` applies to older
files. Format 2 is the first released one. The committed golden file in ``tests/data`` checks
that today's files keep loading to the same outputs. A file with a newer
format fails with an upgrade message. A newer ManySeg may also add options within a format (a
new head option, say): an older ManySeg then cannot build the model, and :func:`load` names
both versions. ``torch.save(model)`` pickles are not covered: they depend on internal module
paths.

Normalisation is stored as a ManyPred normalisation spec (plain data): a list of steps such as
``[{"type": "scale", "value": 10000}, {"type": "zscore", "mean": [...], "std": [...]}]``, or
``{input_name: steps}`` for several inputs.

Quantised models use the same format number; a ``"quantization"`` entry in the metadata marks
them (format 3 marked them before the first release: such files still load, and the next format
is 4):

* ``quantize_weights`` models go in a ``.safetensors`` file too. Each int8 weight is stored as
  its plain tensors (``"<name>:qdata"``, ``":scale"``, ``":zero_point"``), with the arguments it
  was quantised with in the metadata. :func:`load` rebuilds the float model, quantises it with
  those arguments and puts the stored tensors back, so the file depends on no torchao class or
  prototype API.
* ``quantize_pt2e`` programs go in a ``.pt2`` file (see :mod:`manyseg.quantize`): the
  metadata travels as an extra file in the archive, and :func:`load` returns a
  :class:`~manyseg.quantize.QuantizedProgram`. Neither kind holds pickled data.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import __version__
from ._normalize_spec import check_normalize
from .models import DenseModel
from .weights import attribution

FORMAT = 2  # the first released format; quantised files too (their "quantization" entry)
_LEGACY_QUANTIZED = 3  # pre-release quantised files; read as FORMAT. The next format is 4.
_KEY = "manyseg"
ACTIVATIONS = ("softmax", "sigmoid", "none")


def _keep_rng():
    """Isolate a rebuild from the caller's random stream. A rebuild's weight init is thrown away
    (the checkpoint's weights replace it), so it must not change later dropout, augmentation or
    sampling. Rebuilds run on the CPU, so only the CPU generator needs saving."""
    return torch.random.fork_rng(devices=[])


def _plain(value: Any) -> Any:
    """``value`` as JSON-compatible plain data, or raise TypeError."""
    return json.loads(json.dumps(value))


def _weights_normalize(config: Mapping[str, Any], *, partial: bool = False) -> Any:
    """The normalisation the pretrained weights expect, as a ManyPred spec, when the model was
    built with remote-sensing weights tags (``pretrained="ssl4eo_s2_moco"``); else None.
    ``partial`` (for error messages): for a multi-input model without early fusion, return
    ``{input: steps or None}`` even when only some inputs have such a tag."""
    from .weights import get_weights

    def steps(arch: str, tag: Any) -> list | None:
        if not isinstance(tag, str):
            return None
        try:
            w = get_weights(arch, tag)
        except (KeyError, ValueError):
            return None
        out: list = [{"type": "zscore", "mean": list(w.mean), "std": list(w.std)}]
        if w.clip:
            out.append({"type": "clip", "min": 0.0, "max": 1.0})
        return out

    inputs = config.get("inputs")
    if not inputs:
        backbone = config.get("backbone")
        return steps(backbone, config.get("pretrained")) if isinstance(backbone, str) else None
    def encoder(name: str, seen: tuple = ()) -> tuple[Any, Any]:
        """The backbone and weights tag of input ``name``'s encoder, following
        ``"shared:<input>"`` to the input whose encoder it shares."""
        cfg = inputs.get(name) or {}
        arch = cfg.get("backbone", config.get("backbone"))
        if isinstance(arch, str) and arch.startswith("shared:"):
            target = arch.split(":", 1)[1]
            if cfg.get("channel_adapter") == "projection":
                raise ValueError(
                    f"normalize='weights': input {name!r} reaches the shared encoder through a "
                    f"learned projection from its own {cfg.get('in_channels')} bands, so the "
                    f"pretrained weights' normalisation does not describe it. Pass a "
                    f"normalisation spec for each input instead.")
            return encoder(target, (*seen, name)) if target not in (*seen, name) else (None, None)
        return arch, cfg.get("pretrained", config.get("pretrained"))

    fusion = config.get("fusion")
    if (fusion.get("type") if isinstance(fusion, Mapping) else fusion) == "early":
        # One encoder sees the inputs stacked along channels in key order, so each input gets
        # its own slice of the weights' per-band statistics.
        rasters = {n: c for n, c in inputs.items() if c.get("kind") != "vector"}
        # as create_model: an encoder key set on any input (they must agree) overrides the
        # top-level argument
        def shared(key: str) -> Any:
            return next((c[key] for c in rasters.values() if key in c), config.get(key))

        arch, tag = shared("backbone"), shared("pretrained")
        found = steps(arch, tag) if isinstance(arch, str) else None
        if found is None:
            return None
        if len(found[0]["mean"]) != sum(c["in_channels"] for c in rasters.values()):
            return None
        per_input, start = {}, 0
        for name, cfg in rasters.items():
            stop = start + cfg["in_channels"]
            per_input[name] = [{**s, "mean": s["mean"][start:stop], "std": s["std"][start:stop]}
                               if s["type"] == "zscore" else s for s in found]
            start = stop
        return per_input or None

    per_input = {}
    for name, cfg in inputs.items():
        if cfg.get("kind") == "vector":
            continue
        arch, tag = encoder(name)
        per_input[name] = steps(arch, tag) if isinstance(arch, str) else None
    if partial:
        return per_input
    if not per_input or None in per_input.values():
        return None  # all or nothing: a partial spec would silently skip inputs
    return per_input


def _require_dense(model: Any) -> None:
    if isinstance(model, DenseModel):
        return
    inner = next((n for n, m in model.named_children() if isinstance(m, DenseModel)), None) \
        if isinstance(model, torch.nn.Module) else None
    hint = (f" It wraps one as .{inner}: save that, and rebuild the wrapper around "
            f"manyseg.load(path)." if inner else "")
    if isinstance(model, torch.fx.GraphModule):
        hint = (" An ep.module() can't be saved: save the ExportedProgram itself, "
                "manyseg.save(ep, 'model.pt2', source=float_model).")
    raise TypeError(
        f"save takes a manyseg DenseModel, got {type(model).__name__}.{hint} Unwrap "
        f"compiled, DataParallel or wrapped models first (e.g. model._orig_mod, model.module, "
        f"model.model)."
    )


def save(model: DenseModel | torch.export.ExportedProgram, path: str | Path, *,
         source: DenseModel | None = None, normalize: Any = None,
         activations: Mapping[str, str] | None = None,
         classes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
         bands: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
         extra: Mapping[str, Any] | None = None) -> Path:
    """Save ``model`` (built with :func:`manyseg.create_model`) to ``path`` (``.safetensors``).

    A float or ``quantize_weights`` model always saves as safetensors: a ``.pt2`` path is
    refused, as ``.pt2`` files hold ``quantize_pt2e`` programs.

    Quantised models save too, and need ``torchao`` again to load (``manyseg[quant]``):

    * A :func:`~manyseg.quantize.quantize_weights` model: as above, to a ``.safetensors`` file.
    * A :func:`~manyseg.quantize.quantize_pt2e` program: to a ``.pt2`` file, with
      ``source=`` the float model it was quantised from (an ``ExportedProgram`` doesn't know
      its heads, scales or padding). A program loaded back by :func:`load` (the
      :class:`~manyseg.quantize.QuantizedProgram` or its ``.program``) saves the same way.

    Args:
        source: The float model an ``ExportedProgram`` came from; only for those.
        normalize: How inputs must be normalised for this model, as a ManyPred normalisation
            spec: pass the normalisation you trained with, as steps. Several raster inputs take
            ``{input_name: steps}`` with every one present (``{"type": "none"}`` for one left as
            is); a plain spec with no steps suits any model. A spec with an unknown step, a
            misspelt option or an invalid value is refused here, not at inference. ``"weights"``
            uses what the
            remote-sensing pretrained weights the model was built with expect
            (``pretrained="ssl4eo_s2_moco"``), which is right only if you trained with it too.
            None stores nothing (with a warning): whoever runs the model must then know.
        activations: ``{head: "softmax" | "sigmoid" | "none"}``: how each head's raw output
            becomes a prediction (softmax for one class per pixel or image, sigmoid for
            multi-label, none for raw values). Required for every segmentation and
            classification head, as the head type can't tell softmax from sigmoid; regression
            heads default to "none", as do heads with a built-in output ``activation``
            (``"softplus"`` density heads: the model's output already is the prediction).
        classes: Class names, per head (``{head: [...]}``) or for a single head. They reach
            ``tiling_spec()``, and ManyPred writes them into its output files. ``{head: None}``
            names none for that head.
        bands: Band names, per input (``{input: [...]}``) or for a single input. Likewise.
        extra: Any other metadata, as plain JSON data (dicts with string keys, lists,
            strings, numbers, booleans, None), stored as is. Anything JSON would change, such
            as int keys or tuples, is refused rather than read back different.
    """
    names = {"normalize": normalize, "activations": activations, "classes": classes,
             "bands": bands, "extra": extra}
    from .quantize import QuantizedProgram

    if isinstance(model, QuantizedProgram):  # loaded from a .pt2 file: save its program
        model = model.program
    if isinstance(model, torch.export.ExportedProgram):
        if source is None:
            raise TypeError("save(ExportedProgram) needs source=<the float DenseModel it was "
                            "quantised from>: the program doesn't know its heads, scales or "
                            "padding.")
        from .quantize import _save_program

        _require_dense(source)
        return _save_program(model, source, Path(path), _metadata(source, **names))
    if source is not None:
        raise TypeError("source= is only for saving an ExportedProgram.")
    _require_dense(model)
    if Path(path).suffix == ".pt2":
        raise ValueError(
            "A DenseModel saves to a .safetensors file; .pt2 is for quantised programs. To "
            "deploy an int8 program, quantise it with manyseg.quantize.quantize_pt2e and save "
            "that: manyseg.save(ep, 'model.pt2', source=model).")
    quantised = _quantization(model)
    meta = _metadata(model, **names)
    if quantised is not None:
        meta["quantization"] = quantised
    from safetensors.torch import save_file

    tensors, aliases = _dedup_state(model)
    tensors, parts = _split_quantized(tensors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {_KEY: json.dumps(meta), _ALIASES: json.dumps(aliases)}
    if parts:
        stored[_PARTS] = json.dumps(parts)
    save_file(tensors, str(path), metadata=stored)
    return path


def _metadata(model: DenseModel, *, normalize: Any, activations, classes, bands, extra) -> dict:
    """Check what :func:`save` was given and return the metadata every file stores."""
    config = getattr(model, "build_config", None)
    if config is None:
        raise ValueError(
            "This model has no build_config: save works for models built with "
            "manyseg.create_model from plain arguments (names and numbers, not modules or "
            "callables)."
        )
    # a None entry ({head: None}) means "no names" for that head or input: store none
    classes, bands = (({k: v for k, v in value.items() if v is not None} or None)
                      if isinstance(value, Mapping) else value for value in (classes, bands))
    heads = list(model.heads)
    acts = {n: "none" for n in heads if getattr(model.heads[n], "kind", None) == "regression"}
    for name, act in dict(activations or {}).items():
        if name not in model.heads:
            raise ValueError(f"activations given for unknown head {name!r}; heads: {heads}.")
        if act not in ACTIVATIONS:
            raise ValueError(f"Head {name!r}: activation must be one of {ACTIVATIONS}, got "
                             f"{act!r}.")
        if act != "none" and getattr(model.heads[name], "activation", None) is not None:
            raise ValueError(f"Head {name!r} applies {model.heads[name].activation} itself; its "
                             f"output is the prediction, so its activation is 'none'.")
        acts[name] = act
    # a single-input model's input has no name, so its bands can only be a list
    image_inputs = [n for n, c in (config.get("inputs") or {None: {}}).items()
                    if c.get("kind", "raster") != "vector"]
    for arg, value, names, what in (("classes", classes, heads, "head"),
                                    ("bands", bands, image_inputs, "input")):
        if value is None:
            continue
        if isinstance(value, Mapping) and names == [None]:
            raise ValueError(f"This model has one unnamed input: pass {arg}=[...].")
        if isinstance(value, Mapping):
            unknown = [k for k in value if k not in names]
            if unknown:
                raise ValueError(f"{arg} given for unknown {what}s {unknown}; {what}s: {names}.")
        elif isinstance(value, str) or len(names) != 1:
            raise ValueError(f"{arg} as a list is for a model with one {what}; this one has "
                             f"{names}: pass {arg}={{{what}: [...]}}.")
    _check_names(model, config, classes, bands)
    missing = [n for n in heads if n not in acts]
    if missing:
        raise ValueError(
            f"Say how heads {missing} turn into predictions: activations={{head: 'softmax' "
            f"(one class), 'sigmoid' (multi-label) or 'none' (raw values)}}."
        )
    if normalize == "weights":
        normalize = _weights_normalize(config)
        if normalize is None:
            per_input = _weights_normalize(config, partial=True)
            missing = ([n for n, v in per_input.items() if v is None]
                       if isinstance(per_input, Mapping) else [])
            if 0 < len(missing) < len(per_input or ()):
                raise ValueError(
                    f"normalize='weights', but inputs {missing} were not built with "
                    f"remote-sensing pretrained weights (a weights tag), so the weights don't "
                    f"say how to normalise them. Pass a normalisation spec for each input "
                    f"instead: normalize={{input: steps}} for {list(per_input)}.")
            raise ValueError("normalize='weights', but the model was not built with "
                             "remote-sensing pretrained weights (a weights tag).")
    elif normalize is None:
        warnings.warn(
            "No normalisation stored: whoever runs this model must normalise inputs as in "
            "training. Pass normalize=<the spec you trained with>.",
            stacklevel=3,
        )
    if normalize is not None:
        cfgs = config.get("inputs") or {None: config}
        # every image input; a time series' bands are its channels per date
        band_counts = {n: cfgs[n].get("in_channels") or 3 for n in image_inputs}
        check_normalize(normalize, None if image_inputs == [None] else image_inputs,
                        band_counts)
    _check_rebuildable(model, config)
    _check_extra(extra)
    # the model's main dtype; load casts to it, then gives each tensor its own saved dtype
    first = next(iter(model.parameters()))
    meta = {
        "format": FORMAT,
        "manyseg_version": __version__,
        "dtype": str(first.dtype).removeprefix("torch."),
        "create_model": config,
        "heads": heads,
        "single_output": model.single_output,
        "normalize": normalize,
        "activations": acts,
        "classes": classes,
        "bands": bands,
        "pretrained_weights": attribution(model),
        "extra": dict(extra or {}),
    }
    try:
        return _plain(meta)
    except TypeError as e:
        raise ValueError(f"save metadata must be plain JSON-compatible data: {e}") from None


def _check_extra(extra: Any) -> None:
    """``extra`` must come back from JSON unchanged (no int keys, tuples, NaN, ...)."""
    if extra is None:
        return
    if not isinstance(extra, Mapping):
        raise ValueError(f"extra must be a dict, got {type(extra).__name__}.")
    try:
        back = json.loads(json.dumps(dict(extra), allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValueError(f"extra must be plain JSON data: {e}") from None
    if not _same_json(back, dict(extra)):
        raise ValueError("extra would not load back as given: JSON turns tuples into lists and "
                         "non-string keys into strings. Use lists and string keys.")


def _same_json(back: Any, given: Any) -> bool:
    """``back == given``, but a list is not a tuple and ``True``, ``1`` and ``1.0`` differ."""
    if isinstance(given, Mapping):
        return (isinstance(back, dict) and list(back) == list(given)
                and all(_same_json(back[k], given[k]) for k in back))
    if isinstance(given, list):
        return (isinstance(back, list) and len(back) == len(given)
                and all(map(_same_json, back, given)))
    return (back == given and isinstance(back, bool) == isinstance(given, bool)
            and isinstance(back, float) == isinstance(given, float))


_ALIASES = "manyseg_aliases"
_PARTS = "manyseg_quantized"  # {weight name: its plain tensors' names}


def _is_quantized(t: torch.Tensor) -> bool:
    """A torchao quantised weight: a tensor subclass made of plain tensors."""
    return hasattr(t, "tensor_data_names")


def _quantization(model: DenseModel) -> dict | None:
    """What :func:`load` needs to quantise the rebuilt model the same way; None for a float
    model. Raises for weights quantised some other way, which load could not rebuild."""
    recipe = getattr(model, "quantization", None)
    state = model.state_dict(keep_vars=True).values()
    foreign = {type(t).__name__ for t in state if not _is_quantized(t)
               and type(t) not in (torch.Tensor, torch.nn.Parameter)}
    if foreign:
        import torchao

        raise ValueError(
            f"Can't save weights of type {sorted(foreign)}: they don't split into plain tensors. "
            f"This torchao ({torchao.__version__}) quantises some layers that way; upgrade it "
            f"(manyseg[quant] needs 0.18 or newer), then quantise again."
        )
    if not any(_is_quantized(t) for t in state):
        return None
    if recipe is None:
        raise ValueError(
            "This model has torchao quantised weights but did not come from "
            "manyseg.quantize.quantize_weights, so load could not rebuild it. Quantise it with "
            "quantize_weights, or quantize_pt2e and save the ExportedProgram."
        )
    import torchao

    return {"route": "weights", **recipe, "torchao_version": torchao.__version__}


def _split_quantized(tensors: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    """Replace each quantised weight by its plain tensors, which safetensors can store;
    ``parts`` maps the weight's name to their names."""
    plain, parts = {}, {}
    for name, t in tensors.items():
        if not _is_quantized(t):
            plain[name] = t
            continue
        present = [p for p in (*t.tensor_data_names,
                               *(getattr(t, "optional_tensor_data_names", None) or ()))
                   if getattr(t, p) is not None]
        parts[name] = present
        for p in present:
            plain[f"{name}:{p}"] = getattr(t, p).detach().contiguous().clone()
    return plain, parts


def _fill_quantized(model: DenseModel, stored: dict, parts: Mapping[str, Sequence[str]]) -> None:
    """Put the stored tensors into the rebuilt model's quantised weights, and raise if the two
    don't line up (the installed torchao quantises differently from the one that saved)."""
    own = {k: t for k, t in model.state_dict(keep_vars=True).items() if _is_quantized(t)}
    seen: dict[int, str] = {}
    canonical = {k: t for k, t in own.items() if seen.setdefault(id(t), k) == k}
    if set(canonical) != set(parts):
        diff = sorted(set(canonical) ^ set(parts))
        raise RuntimeError(f"quantised weights differ from the file's: {diff[:3]}")
    for name, tensor in canonical.items():
        want = [p for p in (*tensor.tensor_data_names,
                            *(getattr(tensor, "optional_tensor_data_names", None) or ()))
                if getattr(tensor, p) is not None]
        if sorted(want) != sorted(parts[name]):
            raise RuntimeError(f"{name} is made of {want} here, {list(parts[name])} in the file")
        for p in want:
            new = stored[f"{name}:{p}"]
            old = getattr(tensor, p)
            if new.shape != old.shape:
                raise RuntimeError(f"{name}:{p} has shape {tuple(new.shape)} in the file, "
                                   f"{tuple(old.shape)} here")
            setattr(tensor, p, new.to(old.device))


def _dedup_state(model: torch.nn.Module) -> tuple[dict, dict]:
    """Shared modules (shared backbones, shared per-input heads, shared norms) put one tensor
    under several state-dict names. Each is stored once, under its first name in state-dict
    order; ``aliases`` maps every other name to it. (safetensors' save_model / load_model
    deduplicate too, but could pick different names for shared buffers at save and at load.)"""
    first: dict[int, str] = {}
    tensors, aliases = {}, {}
    for name, t in model.state_dict(keep_vars=True).items():
        if id(t) in first:
            aliases[name] = first[id(t)]
            continue
        first[id(t)] = name
        tensors[name] = t if _is_quantized(t) else t.detach().contiguous().clone()
    return tensors, aliases


def _rebuilt_type(module: torch.nn.Module) -> str:
    """The name of the type a file rebuilds ``module`` as. Multi-GPU training swaps in the
    SyncBatchNorm classes: outside training they compute the same as the type they replace and
    hold the same weights. Exact types, as a class that only shares the name may do anything."""
    from timm.layers import SyncBatchNormAct

    kind = type(module)
    if kind is torch.nn.SyncBatchNorm:
        return "BatchNorm2d"
    if kind is SyncBatchNormAct:
        return "BatchNormAct2d"
    return kind.__qualname__


def _structure(model: torch.nn.Module) -> dict[str, tuple]:
    """Every parameter and buffer's shape, and every module's type and settings
    (``extra_repr``: kernel sizes, activation choices, ...)."""
    out: dict[str, tuple] = {k: ("tensor", tuple(v.shape)) for k, v in model.state_dict().items()}
    for name, module in model.named_modules():
        out[f"module:{name}"] = (_rebuilt_type(module), module.extra_repr())
    return out


def _check_names(model, config: dict, classes, bands) -> None:
    """One class name per head channel (a one-channel head may name both of its classes) and
    one band name per input channel, all strings."""
    for arg, value in (("classes", classes), ("bands", bands)):
        for names in (value.values() if isinstance(value, Mapping) else [value]):
            if names is not None and (isinstance(names, (str, bytes)) or not all(
                    isinstance(n, str) for n in names)):
                raise ValueError(f"{arg} must be lists of names (strings), got {names!r}.")
    if classes is not None:
        by_head = classes if isinstance(classes, Mapping) else {next(iter(model.heads)): classes}
        for head, names in by_head.items():
            n = (config.get("heads") or {}).get(head, {}).get("out_channels",
                                                              config.get("num_classes"))
            if n is not None and len(names) != n and (n, len(names)) != (1, 2):
                raise ValueError(f"classes: head {head!r} has {n} channels, got {len(names)} "
                                 f"names {list(names)}.")
    if bands is not None:
        inputs = config.get("inputs") or {None: config}
        for name, values in (bands.items() if isinstance(bands, Mapping)
                             else [(next(n for n, c in inputs.items()
                                         if c.get("kind", "raster") != "vector"), bands)]):
            n = inputs[name].get("in_channels") or 3
            if len(values) != n:
                label = "the input" if name is None else f"input {name!r}"
                raise ValueError(f"bands: {label} has {n} channels, got {len(values)} names "
                                 f"{list(values)}.")


def _check_rebuildable(model: DenseModel, config: Mapping[str, Any]) -> None:
    """Rebuild from ``config`` (no downloads) and compare the module tree, parameters and
    buffers, so a model changed after create_model (modules added or replaced, even by ones
    without weights such as an activation) fails here, not at load time. Changes that leave
    the modules alone (a replaced ``forward``) can't be seen."""
    from .models import create_model

    recipe = getattr(model, "quantization", None)
    with _keep_rng():
        rebuilt = create_model(**_no_download(dict(config)))
        if model.single_output is not None and rebuilt.single_output is None:
            rebuilt.select_heads([model.single_output])
        elif list(rebuilt.heads) != list(model.heads):
            rebuilt.select_heads(list(model.heads))
        if recipe is not None:
            from .quantize import quantize_weights

            # quantise in the model's dtype, as load() does: torchao's quantised weights record
            # the dtype they dequantise to
            rebuilt.to(dtype=next(model.parameters()).dtype)
            quantize_weights(rebuilt, recipe["mode"], conv=recipe["conv"], inplace=True)
            rebuilt.to(next(model.parameters()).device)
    want, have = _structure(model), _structure(rebuilt)
    if want != have:
        diff = sorted(set(want) ^ set(have)) or sorted(k for k in want if want[k] != have[k])
        raise ValueError(
            f"This model differs from what its create_model arguments build (e.g. {diff[:3]}), "
            f"so it could not be loaded back. Save models as create_model built them "
            f"(select_heads is fine)."
        )


def _local(path: str | Path, token: str | None = None) -> str | Path:
    """``path``, or the cached download of a ``"hf-hub:org/name[@revision]"`` name."""
    from .hub import PREFIX, download

    if isinstance(path, str) and path.startswith(PREFIX):
        return download(path, token=token)
    return path


def read_metadata(path: str | Path, *, token: str | None = None) -> dict:
    """The metadata stored by :func:`save`, without loading the weights. ``path`` may name a
    Hub model, ``"hf-hub:org/name[@revision]"`` (see :mod:`manyseg.hub`; ``token`` for a
    private one, default ``HF_TOKEN`` or the stored login)."""
    from safetensors import safe_open

    path = _local(path, token)
    if _is_zip(path):
        stored = _read_archive_metadata(path)
    else:
        with safe_open(str(path), framework="pt") as f:
            stored = (f.metadata() or {}).get(_KEY)
    if stored is None:
        raise ValueError(f"{path} was not written by manyseg.save (no {_KEY!r} metadata). A "
                         f"program from torch.export.save has none: save it with "
                         f"manyseg.save(ep, path, source=float_model).")
    meta = json.loads(stored)
    fmt = meta.get("format")
    if not isinstance(fmt, int) or fmt < 1:
        raise ValueError(f"{path}: invalid save format {fmt!r}.")
    if fmt < FORMAT:
        raise ValueError(f"{path} uses save format {fmt}, which predates ManySeg 0.1.0 and can't "
                         f"be read. Rebuild the model with create_model and load its weights.")
    if fmt == _LEGACY_QUANTIZED and isinstance(meta.get("quantization"), Mapping):
        meta["format"] = fmt = FORMAT  # a pre-release quantised file: same schema
    if fmt > FORMAT:
        raise ValueError(f"{path} uses save format {fmt}; this ManySeg reads up to {FORMAT}. "
                         f"Upgrade ManySeg.")
    if "quantization" in meta and not isinstance(meta["quantization"], Mapping):
        raise ValueError(f"{path}: invalid quantisation metadata {meta['quantization']!r}.")
    missing = {"create_model", "heads", "activations"} - set(meta)
    if missing:
        raise ValueError(f"{path}: metadata lacks {sorted(missing)}.")
    return meta


def _is_zip(path: str | Path) -> bool:
    """A ``.pt2`` archive is a zip file; a safetensors file starts with its header's length."""
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"


def _read_archive_metadata(path: str | Path) -> str | None:
    """The metadata in a ``.pt2`` archive, read as text: nothing else in it is touched."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        name = next((n for n in z.namelist() if n.endswith(f"/extra/{_KEY}.json")), None)
        return z.read(name).decode() if name is not None else None


# timm options that read a local file or fetch weights: the weights come from the saved file,
# and a file's metadata must not make load() open another path (torch.load on a pickle).
_WEIGHT_SOURCES = ("checkpoint_path", "pretrained_cfg", "pretrained_cfg_overlay")


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
           "float64": torch.float64}


def _no_download(config: dict) -> dict:
    """The create_model arguments with every pretrained flag off and every weight source
    removed: the weights come from the file."""
    config = json.loads(json.dumps(config))
    for cfg in (config, *(config.get("inputs") or {}).values()):
        if not isinstance(cfg, dict):
            continue
        if cfg is config or "pretrained" in cfg:
            cfg["pretrained"] = False
        kwargs = cfg.get("backbone_kwargs")
        if isinstance(kwargs, dict):
            for key in _WEIGHT_SOURCES:
                kwargs.pop(key, None)
    return config


def _restore_weight_sources(model: DenseModel, saved: Mapping[str, Any]) -> None:
    """Undo :func:`_no_download` on ``model.build_config``: put back the saved pretrained tags
    and weight sources (only those), so the loaded model saves again as it was built
    (``normalize="weights"`` works and the file keeps its pretrained tag)."""
    config = getattr(model, "build_config", None)
    if not isinstance(config, dict):
        return
    saved = json.loads(json.dumps(saved))
    pairs = [(config, saved)] + [
        (config["inputs"][n], cfg) for n, cfg in (saved.get("inputs") or {}).items()
        if isinstance(cfg, dict) and isinstance(config.get("inputs"), dict)
        and isinstance(config["inputs"].get(n), dict)]
    for have, want in pairs:
        if "pretrained" in want:
            have["pretrained"] = want["pretrained"]
        kwargs = want.get("backbone_kwargs")
        if isinstance(kwargs, dict) and isinstance(have.get("backbone_kwargs"), dict):
            for key in _WEIGHT_SOURCES:
                if key in kwargs:
                    have["backbone_kwargs"][key] = kwargs[key]


def _is_newer(saved, current: str) -> bool:
    """PEP 440 ordering (``0.1.0.dev0 < 0.1.0rc1 < 0.1.0``); False when unknown."""
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # packaging comes with timm's huggingface_hub; be safe anyway
        return False
    try:
        return Version(str(saved)) > Version(current)
    except InvalidVersion:
        return False


def _version_error(path, meta: dict, err: Exception, *, quantizing: bool = False) -> Exception:
    """The error for a file this ManySeg cannot rebuild, naming the versions involved. The
    model was built once when it was saved, so a failure here is either a newer ManySeg's
    option or a missing migration; ``quantizing`` when it came from re-quantising the rebuilt
    model, which depends on torchao."""
    saved = meta.get("manyseg_version", "unknown")
    recipe = meta.get("quantization")
    if recipe is not None and quantizing and not _is_newer(saved, __version__):
        import torchao

        return ValueError(f"{path} cannot be rebuilt: it was saved with torchao "
                          f"{recipe.get('torchao_version')}, and this torchao "
                          f"({torchao.__version__}) quantises it differently ({err}). Install "
                          f"the torchao it was saved with, or a newer one.")
    if _is_newer(saved, __version__):
        return ValueError(f"{path} was saved by ManySeg {saved}; this is {__version__}, which "
                          f"cannot rebuild it ({err}). It may use options added since: upgrade "
                          f"ManySeg to {saved} or later.")
    return ValueError(f"{path} (saved by ManySeg {saved}, format {meta.get('format')}) cannot be "
                      f"rebuilt by this ManySeg ({__version__}): {err}. If it was saved by an "
                      f"older ManySeg, this is a missing migration (a ManySeg bug): please "
                      f"report it.")


def _match_dtypes(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Give each floating-point parameter and buffer the dtype of its tensor in ``state`` (the
    file), before the values are copied in, so a model saved with mixed dtypes (half precision
    with float32 BatchNorm, say) loads as saved, without a lossy round trip through the model's
    main dtype. Parameters keep their identity (``.data`` is replaced) and a buffer shared by
    several modules stays shared."""
    cast: dict[int, torch.Tensor] = {}
    for prefix, owner in model.named_modules(remove_duplicate=False):
        for kind in ("_parameters", "_buffers"):
            for leaf, t in list(getattr(owner, kind).items()):
                if t is None or _is_quantized(t):
                    continue
                if id(t) in cast:
                    if kind == "_buffers":
                        owner._buffers[leaf] = cast[id(t)]
                    continue
                saved = state.get(f"{prefix}.{leaf}" if prefix else leaf)
                if (saved is None or not (t.is_floating_point() and saved.is_floating_point())
                        or t.dtype == saved.dtype):
                    continue
                if kind == "_parameters":
                    t.data = t.data.to(saved.dtype)
                    cast[id(t)] = t
                else:
                    cast[id(t)] = owner._buffers[leaf] = t.to(saved.dtype)


def load(path: str | Path, map_location: str | torch.device = "cpu", *,
         token: str | None = None) -> Any:
    """Rebuild and load a model saved with :func:`save`, in eval mode, on ``map_location`` and
    with each parameter and buffer in the dtype it was saved in (half precision with float32
    BatchNorm stays so). Its metadata is on ``model.saved_metadata`` and its ``tiling_spec()``
    includes the stored normalisation and activations. ``path`` may name a
    Hub model, ``"hf-hub:org/name[@revision]"``, downloaded once and cached (see
    :mod:`manyseg.hub`; ``token`` for a private one, default ``HF_TOKEN`` or the stored
    login).

    A quantised file needs torchao (``manyseg[quant]``). A ``quantize_weights`` file gives a
    :class:`~manyseg.DenseModel` with int8 weights, like the one saved. A ``quantize_pt2e`` file
    gives a :class:`~manyseg.quantize.QuantizedProgram`: a callable with ``tiling_spec()``, not an
    ``nn.Module``."""
    from safetensors.torch import load_file

    from .models import create_model

    path = _local(path, token)
    meta = read_metadata(path)
    recipe = meta.get("quantization")
    if recipe is not None:
        from .quantize import _load_program, _require_torchao

        _require_torchao(f"{path} holds a quantised model; loading it")
        if recipe.get("route") == "pt2e":
            return _load_program(path, meta, map_location)
        if recipe.get("route") != "weights":
            raise ValueError(f"{path}: unknown quantisation route {recipe.get('route')!r}. "
                             f"Upgrade ManySeg.")
    try:
        with _keep_rng():
            model = create_model(**_no_download(meta["create_model"]))
    except (TypeError, ValueError) as e:
        raise _version_error(path, meta, e) from e
    if meta.get("single_output") is not None and model.single_output is None:
        model.select_heads([meta["single_output"]])
        model.single_output = meta["single_output"]
    elif list(model.heads) != meta["heads"]:
        model.select_heads(meta["heads"])
    dtype = _DTYPES.get(meta.get("dtype", "float32"))
    if dtype is None:
        raise ValueError(f"{path}: unsupported dtype {meta.get('dtype')!r}.")
    model.to(dtype=dtype)
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        stored = f.metadata() or {}
    aliases = json.loads(stored.get(_ALIASES, "{}"))
    state = load_file(str(path), device="cpu")
    try:
        if recipe is not None:
            from .quantize import quantize_weights

            with _keep_rng():
                quantize_weights(model, recipe["mode"], conv=recipe["conv"], inplace=True)
            parts = json.loads(stored.get(_PARTS, "{}"))
            _fill_quantized(model, state, parts)
            quantised = {k for k, t in model.state_dict(keep_vars=True).items()
                         if _is_quantized(t)}
            state = {k: v for k, v in state.items() if k.rpartition(":")[0] not in parts}
            state.update({a: state[n] for a, n in aliases.items() if a not in quantised})
            _match_dtypes(model, state)
            result = model.load_state_dict(state, strict=False)
            bad = [*result.unexpected_keys, *(set(result.missing_keys) - quantised)]
            if bad:
                raise RuntimeError(f"state dict mismatch: {sorted(bad)[:3]}")
        else:
            state.update({alias: state[name] for alias, name in aliases.items()})
            _match_dtypes(model, state)
            model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as e:
        raise _version_error(path, meta, e, quantizing=recipe is not None) from e
    model.saved_metadata = meta
    _restore_weight_sources(model, meta["create_model"])
    if recipe is not None:
        model.quantization = {"mode": recipe["mode"], "conv": recipe["conv"]}
    for rec in meta.get("pretrained_weights") or []:  # load built the backbones without them
        rec = dict(rec)
        model.get_submodule(rec.pop("module")).restored_attribution = rec
    return model.to(device=map_location).eval()
