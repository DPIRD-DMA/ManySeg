"""Summarise experiment runs: mean ± SD per arm on val/test, Welch t vs a reference arm.

Usage: uv run python experiments/analyse_runs.py <runs_dir>... --key arch|tag --ref <arm>
       [--metric miou] [--flops]

Only training records (``"kind": "train"``) are summarised; profile and evaluate records,
which ``manyseg-bench`` writes to the same default ``runs`` directory, and any other JSON are
skipped.
"""

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def ms(v):
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb) if va + vb > 0 else float("nan")


def flops(arch: str, options: dict) -> float | None:
    import torch
    from manyseg import create_model
    from torch.utils.flop_counter import FlopCounterMode

    m = create_model(backbone="resnet34", pretrained=False, decoder=arch, num_classes=3,
                     **options).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as f:
        m(torch.randn(1, 3, 256, 256))
    return f.get_total_flops() / 1e9, sum(p.numel() for p in m.parameters()) / 1e6


def load(dirs) -> list[dict]:
    """Every JSON object in ``dirs/*.json`` (unparsable files and non-object JSON skipped)."""
    out = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.json")):
            try:
                r = json.loads(Path(f).read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(r, dict):
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--key", default="tag")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--flops", action="store_true")
    a = ap.parse_args()
    runs = [r for r in load(a.dirs) if r.get("kind") == "train"]
    if not runs:
        raise SystemExit(f"no training records (kind 'train') in {', '.join(a.dirs)}")
    g = defaultdict(list)
    for r in runs:
        key = r["spec"]["arch"] if a.key == "arch" else r.get(a.key, "")
        g[key].append(r)
    pm = a.metric or runs[0].get("primary_metric", "miou")
    splits = [s for s in runs[0]["dataset"]["test_splits"] if s in runs[0]]
    ref = g[a.ref]
    hdr = ["arm", "n", f"val {pm}"] + [f"{s} {pm}" for s in splits] + \
          ["t vs ref (val)"] + [f"t vs ref ({s})" for s in splits]
    if a.flops:
        hdr += ["GFLOPs@256", "params (M)"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for k, rs in sorted(g.items(), key=lambda kv: -st.mean(r["final_val"][pm] for r in kv[1])):
        val = [r["final_val"][pm] for r in rs]
        cells = [str(k), str(len(rs)), ms(val)]
        tests = {s: [r[s][pm] for r in rs if s in r] for s in splits}
        cells += [ms(tests[s]) if tests[s] else "n/a" for s in splits]
        rv = [r["final_val"][pm] for r in ref]
        cells.append("" if k == a.ref else f"{welch(val, rv):+.1f}")
        for s in splits:
            rt = [r[s][pm] for r in ref if s in r]
            cells.append("" if k == a.ref else f"{welch(tests[s], rt):+.1f}")
        if a.flops:
            fl, p = flops(rs[0]["spec"]["arch"], rs[0]["spec"].get("options", {}))
            cells += [f"{fl:.1f}", f"{p:.2f}"]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
