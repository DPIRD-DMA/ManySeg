"""Save a trained model with what inference needs, and load it back.

``save(model, path, ...)`` writes one ``.safetensors`` file: the weights, plus JSON metadata with
the ``create_model`` arguments the model was built with, its heads, and what inference needs to
know that the weights don't say: how inputs were normalised, each head's activation, class
names and band names. ``load(path)`` rebuilds the model (without downloading pretrained weights)
and loads the weights; its :meth:`~manyseg.DenseModel.tiling_spec` then reports the stored
normalisation and activations, so ManyPred uses them without being told.

Normalisation is stored as a ManyPred normalisation spec (plain data): a list of steps such as
``[{"type": "scale", "value": 10000}, {"type": "zscore", "mean": [...], "std": [...]}]``, or
``{input_name: steps}`` for several inputs.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .models import DenseModel

FORMAT = 2  # 2: in_channels / out_channels / num_features (1: "channels"; migrated on read)
_KEY = "manyseg"
ACTIVATIONS = ("softmax", "sigmoid", "none")


def _plain(value: Any) -> Any:
    """``value`` as JSON-compatible plain data, or raise TypeError."""
    return json.loads(json.dumps(value))


def _weights_normalize(config: Mapping[str, Any]) -> Any:
    """The normalisation the pretrained weights expect, as a ManyPred spec, when the model was
    built with remote-sensing weights tags (``pretrained="ssl4eo_s2_moco"``); else None."""
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
    per_input = {}
    for name, cfg in inputs.items():
        if cfg.get("kind") == "vector":
            continue
        arch = cfg.get("backbone", config.get("backbone"))
        found = steps(arch, cfg.get("pretrained", config.get("pretrained"))) if isinstance(
            arch, str) else None
        if found is None:
            return None  # all or nothing: a partial spec would silently skip inputs
        per_input[name] = found
    return per_input or None


def save(model: DenseModel, path: str | Path, *, normalize: Any = None,
         activations: Mapping[str, str] | None = None,
         classes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
         bands: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
         extra: Mapping[str, Any] | None = None) -> Path:
    """Save ``model`` (built with :func:`manyseg.create_model`) to ``path`` (``.safetensors``).

    Args:
        normalize: How inputs must be normalised for this model, as a ManyPred normalisation
            spec: pass the normalisation you trained with. ``"weights"`` uses what the
            remote-sensing pretrained weights the model was built with expect
            (``pretrained="ssl4eo_s2_moco"``), which is right only if you trained with it too.
            None stores nothing (with a warning): whoever runs the model must then know.
        activations: ``{head: "softmax" | "sigmoid" | "none"}``: how each head's raw output
            becomes a prediction (softmax for one class per pixel or image, sigmoid for
            multi-label, none for raw values). Required for every segmentation and
            classification head, as the head type can't tell softmax from sigmoid; regression
            heads default to "none".
        classes: Class names, per head (``{head: [...]}``) or for a single head.
        bands: Band names, per input (``{input: [...]}``) or for a single input.
        extra: Any other JSON-compatible metadata, stored as is.
    """
    if not isinstance(model, DenseModel):
        raise TypeError(
            f"save takes a manyseg DenseModel, got {type(model).__name__}. Unwrap compiled, "
            f"DataParallel or quantised models first (e.g. model._orig_mod, model.module)."
        )
    config = getattr(model, "build_config", None)
    if config is None:
        raise ValueError(
            "This model has no build_config: save works for models built with "
            "manyseg.create_model from plain arguments (names and numbers, not modules or "
            "callables)."
        )
    heads = list(model.heads)
    acts = {n: "none" for n in heads if getattr(model.heads[n], "kind", None) == "regression"}
    for name, act in dict(activations or {}).items():
        if name not in model.heads:
            raise ValueError(f"activations given for unknown head {name!r}; heads: {heads}.")
        if act not in ACTIVATIONS:
            raise ValueError(f"Head {name!r}: activation must be one of {ACTIVATIONS}, got "
                             f"{act!r}.")
        acts[name] = act
    missing = [n for n in heads if n not in acts]
    if missing:
        raise ValueError(
            f"Say how heads {missing} turn into predictions: activations={{head: 'softmax' "
            f"(one class), 'sigmoid' (multi-label) or 'none' (raw values)}}."
        )
    if normalize == "weights":
        normalize = _weights_normalize(config)
        if normalize is None:
            raise ValueError("normalize='weights', but the model was not built with "
                             "remote-sensing pretrained weights (a weights tag).")
    elif normalize is None:
        warnings.warn(
            "No normalisation stored: whoever runs this model must normalise inputs as in "
            "training. Pass normalize=<the spec you trained with>.",
            stacklevel=2,
        )
    _check_rebuildable(model, config)
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
        "extra": dict(extra or {}),
    }
    try:
        meta = _plain(meta)
    except TypeError as e:
        raise ValueError(f"save metadata must be plain JSON-compatible data: {e}") from None
    from safetensors.torch import save_file

    tensors, aliases = _dedup_state(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={_KEY: json.dumps(meta),
                                            _ALIASES: json.dumps(aliases)})
    return path


_ALIASES = "manyseg_aliases"


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
        tensors[name] = t.detach().contiguous().clone()
    return tensors, aliases


def _structure(model: torch.nn.Module) -> dict[str, tuple]:
    """Every parameter and buffer's shape, and every module's type and settings
    (``extra_repr``: kernel sizes, activation choices, ...)."""
    out: dict[str, tuple] = {k: ("tensor", tuple(v.shape)) for k, v in model.state_dict().items()}
    for name, module in model.named_modules():
        out[f"module:{name}"] = (type(module).__qualname__, module.extra_repr())
    return out


def _check_rebuildable(model: DenseModel, config: Mapping[str, Any]) -> None:
    """Rebuild from ``config`` (no downloads) and compare the module tree, parameters and
    buffers, so a model changed after create_model (modules added or replaced, even by ones
    without weights such as an activation) fails here, not at load time. Changes that leave
    the modules alone (a replaced ``forward``) can't be seen."""
    from .models import create_model

    rebuilt = create_model(**_no_download(dict(config)))
    if model.single_output is not None and rebuilt.single_output is None:
        rebuilt.select_heads([model.single_output])
    elif list(rebuilt.heads) != list(model.heads):
        rebuilt.select_heads(list(model.heads))
    want, have = _structure(model), _structure(rebuilt)
    if want != have:
        diff = sorted(set(want) ^ set(have)) or sorted(k for k in want if want[k] != have[k])
        raise ValueError(
            f"This model differs from what its create_model arguments build (e.g. {diff[:3]}), "
            f"so it could not be loaded back. Save models as create_model built them "
            f"(select_heads is fine)."
        )


def read_metadata(path: str | Path) -> dict:
    """The metadata stored by :func:`save`, without loading the weights."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        stored = (f.metadata() or {}).get(_KEY)
    if stored is None:
        raise ValueError(f"{path} was not written by manyseg.save (no {_KEY!r} metadata).")
    meta = json.loads(stored)
    fmt = meta.get("format")
    if not isinstance(fmt, int) or fmt < 1:
        raise ValueError(f"{path}: invalid save format {fmt!r}.")
    if fmt > FORMAT:
        raise ValueError(f"{path} uses save format {fmt}; this ManySeg reads up to {FORMAT}. "
                         f"Upgrade ManySeg.")
    missing = {"create_model", "heads", "activations"} - set(meta)
    if missing:
        raise ValueError(f"{path}: metadata lacks {sorted(missing)}.")
    if fmt < 2:
        meta["create_model"] = _migrate_v1(meta["create_model"])
    return meta


def _migrate_v1(config: dict) -> dict:
    """Format 1 (before 0.1.0) used ``"channels"`` for input bands, head outputs and raw
    vector widths; format 2 says ``in_channels`` / ``out_channels`` / ``num_features``."""
    config = dict(config)

    def rename(cfg, new):
        cfg = dict(cfg)
        if "channels" in cfg:
            cfg[new] = cfg.pop("channels")
        return cfg

    if config.get("inputs"):
        config["inputs"] = {n: rename(c, "num_features" if c.get("kind") == "vector"
                                      else "in_channels") for n, c in config["inputs"].items()}
    if config.get("heads"):
        config["heads"] = {n: rename(c, "out_channels") for n, c in config["heads"].items()}
    return config


def _no_download(config: dict) -> dict:
    """The create_model arguments with every pretrained flag off (the weights come from the
    file)."""
    config = json.loads(json.dumps(config))
    config["pretrained"] = False
    if config.get("inputs") is not None and config.get("in_channels") == 3:
        config["in_channels"] = None  # files saved before in_channels defaulted to None
    for cfg in (config.get("inputs") or {}).values():
        if "pretrained" in cfg:
            cfg["pretrained"] = False
    return config


def load(path: str | Path, map_location: str | torch.device = "cpu") -> DenseModel:
    """Rebuild and load a model saved with :func:`save`, in eval mode, on ``map_location`` and
    in the dtype it was saved in. Its metadata is on ``model.saved_metadata`` and its
    ``tiling_spec()`` includes the stored normalisation and activations."""
    from safetensors.torch import load_file

    from .models import create_model

    meta = read_metadata(path)
    model = create_model(**_no_download(meta["create_model"]))
    if meta.get("single_output") is not None and model.single_output is None:
        model.select_heads([meta["single_output"]])
        model.single_output = meta["single_output"]
    elif list(model.heads) != meta["heads"]:
        model.select_heads(meta["heads"])
    dtype = getattr(torch, meta.get("dtype", "float32"))
    model.to(dtype=dtype)
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        aliases = json.loads((f.metadata() or {}).get(_ALIASES, "{}"))
    state = load_file(str(path), device="cpu")
    state.update({alias: state[name] for alias, name in aliases.items()})
    model.load_state_dict(state, strict=True)
    model.saved_metadata = meta
    return model.to(device=map_location).eval()
