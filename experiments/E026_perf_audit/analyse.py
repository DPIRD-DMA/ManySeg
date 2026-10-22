"""E026: summarise runs/*.jsonl into markdown tables (current code vs patched)."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).parent / "runs"


def load(name):
    p = R / name
    return [json.loads(line) for line in p.open()] if p.exists() else []


def models():
    rows = [r for r in load("models.jsonl") if r["steps"] >= 10]  # drop the preflight
    # pair each patched run with the current-code run just before it (the grid is A B A B, so
    # slow drift on the shared GPU cancels)
    last_base, pairs = {}, defaultdict(list)
    for r in rows:
        key = (r["model"], r["mode"], r["batch"], r["size"], r["compile"], r["channels_last"],
               ",".join(r["ckpt"]))
        if not r["patches"]:
            last_base[key] = r
        elif key in last_base:
            pairs[(key, ",".join(r["patches"]))].append((last_base[key], r))
    groups = defaultdict(list)
    for r in rows:
        key = (r["model"], r["mode"], r["batch"], r["size"], r["compile"], r["channels_last"],
               ",".join(r["ckpt"]))
        groups[key].append(r)
    out = ["## Patches (train step, bf16, current -> patched; Δ time = median of paired A/B runs)", "",
           "| model | patch | mode | ms current | ms patched | Δ time | GiB current | "
           "GiB patched | Δ peak | pairs |", "|---|---|---|---|---|---|---|---|---|---|"]
    ck, inf = [], []
    for key, rs in groups.items():
        model, mode, batch, size, comp, cl, ckpt = key
        base = [r for r in rs if not r["patches"]]
        by_patch = defaultdict(list)
        for r in rs:
            if r["patches"]:
                by_patch[",".join(r["patches"])].append(r)
        if not base:
            continue
        bm = statistics.median(r["median_ms"] for r in base)
        bg = max(r["peak_gib"] for r in base)
        label = f"{'compile' if comp else 'eager'}{' CL' if cl else ''} b{batch} {size}²"
        for p in by_patch:
            ps = pairs[(key, p)]
            bm = statistics.median(b["median_ms"] for b, _ in ps)
            pm = statistics.median(q["median_ms"] for _, q in ps)
            d = statistics.median(100 * (q["median_ms"] / b["median_ms"] - 1) for b, q in ps)
            bg = max(b["peak_gib"] for b, _ in ps)
            pg = max(q["peak_gib"] for _, q in ps)
            out.append(f"| {model} | {p} | {label} | {bm:.1f} | {pm:.1f} | {d:+.1f}% "
                       f"| {bg:.2f} | {pg:.2f} | {100 * (pg / bg - 1):+.1f}% | {len(ps)} |")
        if mode == "train" and not by_patch and (ckpt or model in ("unet_convnext",)):
            ck.append((model, ckpt or "none", bm, bg, batch, size))
        if mode == "infer":
            inf.append((model, batch, comp or "eager", bm, bg))
    out += ["", "## Gradient checkpointing (train step)", "",
            "| model | checkpointing | ms | peak GiB |", "|---|---|---|---|"]
    out += [f"| {m} b{b} {s}² | {c} | {t:.1f} | {g:.2f} |" for m, c, t, g, b, s in sorted(ck)]
    out += ["", "## Inference (bf16, inference_mode)", "",
            "| model | batch | mode | ms | peak GiB |", "|---|---|---|---|---|"]
    out += [f"| {m} | {b} | {c} | {t:.2f} | {g:.2f} |" for m, b, c, t, g in sorted(inf)]
    return out


def train():
    rows = load("train.jsonl")
    if not rows:
        return []
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["model"], r["size"], r["batch"], r["in_ch"])][",".join(r["variant"])].append(r)
    out = ["", "## Training loop (ms per step incl. loader, median of reps)", "",
           "| model | variant | ms/step | Δ vs base | peak GiB |", "|---|---|---|---|---|"]
    for key, vs in groups.items():
        b = statistics.median(r["ms_per_step"] for r in vs.get("base", []) or [{"ms_per_step": 0}])
        for v, rs in vs.items():
            m = statistics.median(r["ms_per_step"] for r in rs)
            d = f"{100 * (m / b - 1):+.1f}%" if b else ""
            out.append(f"| {key[0]} {key[1]}² b{key[2]} {key[3]}ch | {v} | {m:.1f} | {d} | "
                       f"{max(r['peak_gib'] for r in rs):.2f} |")
    return out


def fetch():
    rows = load("fetch.jsonl")
    if not rows:
        return []
    out = ["", "## CloudSEN12+ cold reads (batch 16, U-Net R34 13-band 512²)", "",
           "| variant | ms/step | waiting for data ms/step | MB/s | images/s |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['variant']} | {r['ms_per_step']:.0f} | "
                   f"{r['wait_for_data_ms_per_step']:.0f} | {r['read_mb_per_s']:.0f} | "
                   f"{r['images_per_s']:.1f} |")
    return out


def post():
    rows = load("post.jsonl")
    if not rows:
        return []
    out = ["", "## Post-processing", "", "| bench | device | input | current s | new s | "
           "speedup | identical / max diff |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["bench"] == "connected_components":
            out.append(f"| CC | {r['device']} | {r['mask']} b{r['batch']} {r['size']}² | "
                       f"{r['current_s']:.3f} | {r['jump_s']:.3f} | {r['speedup']:.1f}x | "
                       f"{r['identical']} |")
        else:
            out.append(f"| EDT | {r['device']} (scipy: cpu) | b{r['batch']} {r['size']}² | "
                       f"{r['torch_s']:.3f} | {r.get('scipy_cpu_s', float('nan')):.3f} | "
                       f"{r.get('speedup', float('nan')):.1f}x | {r.get('max_abs_diff')} |")
    return out


if __name__ == "__main__":
    text = "\n".join(models() + train() + fetch() + post())
    print(text)
    if "--write" in sys.argv:
        (Path(__file__).parent / "results.md").write_text(text + "\n")
