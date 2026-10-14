"""Aggregate run JSONs into tables (mean ± std over seeds) and Pareto fronts."""

from __future__ import annotations

import json
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


def _ms(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def group_key(r: dict) -> tuple:
    s = r["spec"]
    opts = json.dumps(s.get("options", {}), sort_keys=True)
    return (r["dataset"]["name"], r["recipe"]["name"], r["provider"], s["arch"], s["encoder"],
            opts, r.get("tag", ""))


def training_table(runs: list[dict]) -> str:
    groups = defaultdict(list)
    for r in runs:
        if r.get("kind") == "train":
            groups[group_key(r)].append(r)
    lines = ["| dataset | recipe | provider | arch | encoder | options | tag | n | params (M) | "
             "val primary (final) | val boundary F1 | test primary | train time (s) | "
             "fingerprint |",
             "|" + "---|" * 14]
    for k, rs in sorted(groups.items()):
        fps = sorted({r["weights_fingerprint"] for r in rs})
        pm = rs[0].get("primary_metric", "miou")
        tests = [r["test"][pm] for r in rs if "test" in r]
        lines.append(
            f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {k[6]} | {len(rs)} | "
            f"{rs[0]['params'] / 1e6:.2f} | {pm}: {_ms([r['final_val'][pm] for r in rs])} | "
            f"{_ms([r['final_val'].get('boundary_f1', float('nan')) for r in rs])} | "
            f"{_ms(tests)} | "
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
