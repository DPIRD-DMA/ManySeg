"""Check a normalisation spec before :func:`manyseg.save` stores it.

A spec is ManyPred's plain-data format (its ``normalize.py`` documents the steps). ManySeg stores
specs but never runs them and doesn't import ManyPred, so a typo would otherwise surface only
when inference fails. These are the rules of ManyPred's ``validate_spec``, kept in step by
``tests/unit/test_normalize_spec.py``, without callables, which a file can't hold. Two checks are
ManySeg's own, as it knows the model's inputs: every raster input needs a spec, and a metadata
input can't have one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_OPTIONS = {
    "none": set(),
    "scale": {"value"},
    "zscore": {"mean", "std", "scope"},
    "minmax": {"min", "max", "clip"},
    "percentile": {"low", "high", "scope", "clip"},
    "db": {"eps"},
    "clip": {"min", "max"},
}
_SCOPES = ("tile", "scene")


def _flatten(value: Any) -> list:
    """A number or an evenly nested list of them, as a flat list of floats; ValueError if uneven
    (numpy, which ManyPred reads these with, can't make an array of it)."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [float(value)]
    parts = [_flatten(v) for v in value]
    nested = {isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in value}
    if len(nested) > 1 or len({len(p) for p in parts}) > 1:
        raise ValueError("uneven nesting")
    return [x for p in parts for x in p]


def _numbers(value: Any, name: str, *, nonzero: bool = False, positive: bool = False) -> list:
    """A number or a (nested) list of them, as a flat list of finite floats."""
    if hasattr(value, "tolist"):  # numpy arrays and tensors reach the JSON check with a clear error
        value = value.tolist()
    try:
        flat = _flatten(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or a list of numbers, got {value!r}.") from None
    if not flat or not all(math.isfinite(x) for x in flat):
        raise ValueError(f"{name} must be finite numbers, got {value!r}.")
    if nonzero and any(x == 0 for x in flat):
        raise ValueError(f"{name} must not be 0, got {value!r}.")
    if positive and any(x <= 0 for x in flat):
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return flat


def _pairs(lo: list, hi: list, what: str) -> zip:
    """Scalars apply to every band; otherwise one value per band on both sides."""
    if len(lo) != len(hi) and 1 not in (len(lo), len(hi)):
        raise ValueError(f"{what} need the same number of values.")
    n = max(len(lo), len(hi))
    return zip(lo * (n if len(lo) == 1 else 1), hi * (n if len(hi) == 1 else 1))


def _check_step(step: Any) -> None:
    if callable(step):
        raise ValueError(f"A callable normalisation step can't be saved in a file, got {step!r}.")
    if not isinstance(step, Mapping):
        raise ValueError(
            f"A normalisation step must be a dict like {{'type': 'scale', 'value': 255}}, got "
            f"{step!r}. For no normalisation, pass [] or {{'type': 'none'}}.")
    if "type" not in step:
        raise ValueError(f"Normalisation step {dict(step)!r} has no 'type' key.")
    kind = step["type"]
    if not isinstance(kind, str) or kind not in _OPTIONS:
        raise ValueError(f"Unknown normalisation type {kind!r}; use one of {sorted(_OPTIONS)}.")
    extra = set(step) - _OPTIONS[kind] - {"type"}
    if extra:
        raise ValueError(f"{kind!r} step has unexpected keys {sorted(extra)}; allowed: "
                         f"{sorted(_OPTIONS[kind])}.")
    scope = step.get("scope")
    if scope is not None and scope not in _SCOPES:
        raise ValueError(f"{kind!r} step: scope must be one of {_SCOPES}, got {scope!r}.")
    if "clip" in step and kind != "clip" and not isinstance(step["clip"], bool):
        raise ValueError(f"{kind!r} step: clip must be True or False, got {step['clip']!r}.")
    if kind == "scale":
        if "value" not in step:
            raise ValueError("'scale' step needs 'value' (scalar or one per band).")
        _numbers(step["value"], "scale value", nonzero=True)
    elif kind == "zscore":
        has_stats = "mean" in step and "std" in step
        if has_stats == (scope is not None) or ("mean" in step) != ("std" in step):
            raise ValueError("'zscore' step needs dataset mean and std, or scope='tile' / 'scene' "
                             "(not both).")
        if has_stats:
            _numbers(step["mean"], "mean")
            _numbers(step["std"], "std", positive=True)
    elif kind == "minmax":
        for key in ("min", "max"):
            if key not in step:
                raise ValueError(f"'minmax' step needs '{key}'.")
        pairs = _pairs(_numbers(step["min"], "min"), _numbers(step["max"], "max"), "'minmax' min "
                       "and max")
        if not all(a < b for a, b in pairs):
            raise ValueError("'minmax' step needs min < max for every band.")
    elif kind == "percentile":
        low, high = step.get("low", 2), step.get("high", 98)
        try:
            ordered = 0 <= low < high <= 100
        except TypeError:
            ordered = False
        if not ordered:
            raise ValueError(f"'percentile' step needs 0 <= low < high <= 100, got low={low}, "
                             f"high={high}.")
    elif kind == "db":
        eps = step.get("eps", 1e-10)
        if isinstance(eps, str) or not eps > 0:
            raise ValueError(f"'db' step needs eps > 0, got {eps!r}.")
    elif kind == "clip":
        if "min" not in step and "max" not in step:
            raise ValueError("'clip' step needs 'min', 'max' or both.")
        lo = _numbers(step["min"], "clip min") if "min" in step else None
        hi = _numbers(step["max"], "clip max") if "max" in step else None
        if lo is not None and hi is not None and not all(
                a <= b for a, b in _pairs(lo, hi, "'clip' min and max")):
            raise ValueError("'clip' step needs min <= max.")


def check_steps(spec: Any) -> None:
    """One step or a list of steps, in order. Raises ValueError saying what to fix."""
    if spec is None:
        return
    steps = list(spec) if isinstance(spec, Sequence) and not isinstance(spec, str) else [spec]
    per_tile = False
    for step in steps:
        _check_step(step)
        if step.get("scope") == "scene" and per_tile:
            raise ValueError("A scope='scene' step can't follow a scope='tile' step: scene "
                             "statistics are computed once over the whole scene, before tiling. "
                             "Put scene steps first.")
        per_tile |= step.get("scope") == "tile"


def check_normalize(spec: Any, inputs: Sequence[str] | None) -> None:
    """``spec`` for a model's raster ``inputs`` (None: one unnamed input, which takes steps). A
    model with named inputs takes ``{input_name: steps}``, with every raster input present: a
    forgotten one would silently get raw values."""
    if inputs is None:
        check_steps(spec)
        return
    if not isinstance(spec, Mapping) or "type" in spec:
        raise ValueError(f"This model has named inputs {list(inputs)}: pass normalize as "
                         f"{{input_name: steps}}, got {spec!r}.")
    unknown = sorted(set(spec) - set(inputs))
    if unknown:
        raise ValueError(f"normalize given for unknown inputs {unknown}; raster inputs: "
                         f"{list(inputs)}. Metadata inputs take no normalisation.")
    missing = [n for n in inputs if n not in spec]
    if missing:
        raise ValueError(f"normalize doesn't cover inputs {missing}, which would get raw values. "
                         f"If that's intended, say so: {missing[0]!r}: {{'type': 'none'}}.")
    for name, steps in spec.items():
        try:
            check_steps(steps)
        except ValueError as e:
            raise ValueError(f"normalize for input {name!r}: {e}") from None
