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
            spec. Default: what the pretrained weights expect, if the model was built with a
            remote-sensing weights tag (``pretrained="ssl4eo_s2_moco"``); else nothing is
            stored and whoever runs the model must know. Pass the spec you trained with.
        activations: ``{head: "softmax" | "sigmoid" | "none"}``: how each head's raw output
            becomes a prediction (softmax for one class per pixel or image, sigmoid for
            multi-label, none for regression). Regression heads default to "none".
        classes: Class names, per head (``{head: [...]}``) or for a single head.
        bands: Band names, per input (``{input: [...]}``) or for a single input.
        extra: Any other JSON-compatible metadata, stored as is.
    """
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
    if normalize is None:
        normalize = _weights_normalize(config)
    meta = {
        "format": FORMAT,
        "manyseg_version": __version__,
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


def read_metadata(path: str | Path) -> dict:
    """The metadata stored by :func:`save`, without loading the weights."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        stored = (f.metadata() or {}).get(_KEY)
    if stored is None:
        raise ValueError(f"{path} was not written by manyseg.save (no {_KEY!r} metadata).")
    meta = json.loads(stored)
    if meta.get("format", 0) > FORMAT:
        raise ValueError(f"{path} uses save format {meta['format']}; this ManySeg reads up to "
                         f"{FORMAT}. Upgrade ManySeg.")
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
    """Rebuild and load a model saved with :func:`save`, in eval mode. Its metadata is on
    ``model.saved_metadata`` and its ``tiling_spec()`` includes the stored normalisation and
    activations."""
    from safetensors.torch import load_model

    from .models import create_model

    meta = read_metadata(path)
    model = create_model(**_no_download(meta["create_model"]))
    if meta.get("single_output") is not None and model.single_output is None:
        model.select_heads([meta["single_output"]])
        model.single_output = meta["single_output"]
    elif list(model.heads) != meta["heads"]:
        model.select_heads(meta["heads"])
    load_model(model, str(path), strict=True, device=str(map_location))
    model.saved_metadata = meta
    return model.eval()
