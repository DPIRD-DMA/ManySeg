"""Aggregate run JSONs into tables (mean ± std over seeds) and Pareto fronts."""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_runs(paths) -> list[dict]:
    runs = []
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                runs.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return runs


def _ms(vals: list) -> str:
    """mean ± std over seeds; NaN / missing values (e.g. no boundary F1) are left out."""
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def _defaults(cls) -> dict:
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            out[f.name] = f.default_factory()
    return json.loads(json.dumps(out))  # tuples -> lists, as in the records


def _canon(d, defaults: dict | None = None) -> str:
    """Canonical JSON of a config dict. Fields missing from the record (written before the
    field existed) take the dataclass default, and ``None`` values are dropped, so old and new
    records of one configuration compare equal."""
    d = {**(defaults or {}), **(d or {})}
    return json.dumps({k: v for k, v in d.items() if v is not None}, sort_keys=True)


def _task_kwargs(r: dict) -> str:
    """Canonical JSON of a record's ``task_kwargs`` without the entries that restate an
    effective default: the task constructor's default overlaid with the dataset's
    ``TASK_KWARGS`` (what the runner applies to a kwarg left out). So ``{}`` and
    ``{"label_smoothing": 0.0}`` (records before / after the runner wrote the recipe's
    default) are one configuration, while any non-default setting stays distinct."""
    from ..datasets import get_dataset
    from ..tasks import task_defaults

    kw = json.loads(json.dumps(r.get("task_kwargs") or {}))
    try:
        ds = get_dataset(r["dataset"]["name"])
    except ValueError:  # a dataset this version does not know: compare as recorded
        return _canon(kw)
    defaults = {**task_defaults(getattr(ds, "TASK", "semantic")),
                **getattr(ds, "TASK_KWARGS", {})}
    defaults = json.loads(json.dumps(defaults))  # tuples -> lists, as in the records
    return _canon({k: v for k, v in kw.items() if k not in defaults or defaults[k] != v})


def group_key(r: dict) -> tuple:
    """Every configuration field of a train record except the seed: runs with the same key are
    seeds of one configuration and are averaged.

    Fields: dataset name, the full recipe (so overrides such as ``epochs=5`` separate runs even
    though the recipe keeps its name), provider, the full model spec (weights, input mode,
    options, band names, ...), task kwargs (entries equal to the task's effective default left
    out, see ``_task_kwargs``), split scheme, subset and tag. Records written before a field
    existed group as if it held its default.
    """
    from ..providers.base import ModelSpec
    from ..training.recipe import Recipe

    return (r["dataset"]["name"], r["recipe"]["name"], r["provider"], r["spec"]["arch"],
            r["spec"]["encoder"], json.dumps(r["spec"].get("options") or {}, sort_keys=True),
            r.get("tag", ""), _canon(r["recipe"], _defaults(Recipe)),
            _canon(r["spec"], _defaults(ModelSpec)),
            _task_kwargs(r), r.get("split_scheme") or "official", r.get("subset"))


def _config_label(k: tuple) -> str:
    """The key fields the table has no column for, compactly (blank when all are defaults)."""
    spec, rec = json.loads(k[8]), json.loads(k[7])
    parts = [f"{f}={spec[f]}" for f in ("weights", "input_mode") if f in spec]
    if k[10] != "official":
        parts.append(f"split={k[10]}")
    if k[11]:
        parts.append(f"subset={k[11]}")
    if k[9] != "{}":  # non-default task settings only
        parts.append(f"task={k[9]}")
    parts.append(f"epochs={rec.get('epochs')}")
    return " ".join(parts)


def training_table(runs: list[dict]) -> str:
    """Markdown table of train records, one row per configuration (``group_key``): mean ± std
    over seeds of the primary metric on the final validation epoch and on each declared test
    split evaluated (``split: value``, "n/a" when none was)."""
    groups = defaultdict(list)
    for r in runs:
        if r.get("kind") == "train":
            groups[group_key(r)].append(r)
    lines = ["| dataset | recipe | provider | arch | encoder | options | tag | config | n | "
             "params (M) | val primary (final) | val boundary F1 | test primary | "
             "train time (s) | fingerprint |",
             "|" + "---|" * 15]
    for k, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        fps = sorted({r["weights_fingerprint"] for r in rs})
        pm = rs[0].get("primary_metric", "miou")
        # every declared test split (e.g. Sen1Floods11's test and Bolivia), named
        splits = rs[0]["dataset"].get("test_splits") or ["test"]
        tests = "; ".join(f"{sp}: {_ms([r[sp][pm] for r in rs if sp in r])}"
                          for sp in splits if any(sp in r for r in rs)) or "n/a"
        lines.append(
            f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {k[6]} | "
            f"{_config_label(k)} | {len(rs)} | "
            f"{rs[0]['params'] / 1e6:.2f} | {pm}: {_ms([r['final_val'][pm] for r in rs])} | "
            f"{_ms([r['final_val'].get('boundary_f1', float('nan')) for r in rs])} | "
            f"{tests} | "
            f"{statistics.mean(r['train_time_s'] for r in rs):.0f} | {','.join(fps)} |")
    return "\n".join(lines)


def pareto_front(points: list[tuple[float, float, str]]) -> list[str]:
    """points: (cost, quality, label). Lower cost and higher quality are better."""
    front, best = [], float("-inf")
    for _cost, q, label in sorted(points, key=lambda t: (t[0], -t[1])):
        if q > best:
            front.append(label)
            best = q
    return front
