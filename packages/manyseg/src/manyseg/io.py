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

FORMAT = 1
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
    from safetensors.torch import save_model

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # save_model, not save_file: it handles tied weights (shared backbones, shared heads)
    save_model(model, str(path), metadata={_KEY: json.dumps(meta)})
    return path


def _check_rebuildable(model: DenseModel, config: Mapping[str, Any]) -> None:
    """Rebuild from ``config`` (no downloads) and compare parameters and buffers, so a model
    changed after create_model (modules added or replaced) fails here, not at load time."""
    from .models import create_model

    rebuilt = create_model(**_no_download(dict(config)))
    if model.single_output is not None and rebuilt.single_output is None:
        rebuilt.select_heads([model.single_output])
    elif list(rebuilt.heads) != list(model.heads):
        rebuilt.select_heads(list(model.heads))
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    have = {k: tuple(v.shape) for k, v in rebuilt.state_dict().items()}
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
    return meta


def _no_download(config: dict) -> dict:
    """The create_model arguments with every pretrained flag off (the weights come from the
    file)."""
    config = json.loads(json.dumps(config))
    config["pretrained"] = False
    for cfg in (config.get("inputs") or {}).values():
        if "pretrained" in cfg:
            cfg["pretrained"] = False
    return config


def load(path: str | Path, map_location: str | torch.device = "cpu") -> DenseModel:
    """Rebuild and load a model saved with :func:`save`, in eval mode, on ``map_location`` and
    in the dtype it was saved in. Its metadata is on ``model.saved_metadata`` and its
    ``tiling_spec()`` includes the stored normalisation and activations."""
    from safetensors.torch import load_model

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
    load_model(model, str(path), strict=True, device="cpu")
    model.saved_metadata = meta
    return model.to(device=map_location).eval()
