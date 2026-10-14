"""E006 analysis: Pareto front, frontier selection (pre-registered rule) and hypothesis check.

Usage:
  uv run python experiments/E006_decoder_pareto/analyse.py <runs_dir> [--write-frontier F]
      [--final] [--plot out.png]

Selection and the hypothesis use validation metrics only. Test metrics are printed only with
``--final`` (after stage 2), never used to choose anything.
"""

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict

FORCED = ["unet:1", "unet:2"]  # the hypothesis pair always gets 3 seeds
MAX_PER_CELL = 4
ORDER = ["unet", "fastai_unet", "segformer", "fpn", "deeplabv3plus"]  # fixed colour slots
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz reference palette
MARKERS = {"1": "o", "2": "s", "4": "^", "native": "D"}


def config_of(tag):  # "e006-unet-s2" -> "unet:2"; "e006-fpn-snative" -> "fpn:native"
    arch, s = tag[len("e006-"):].rsplit("-s", 1)
    return f"{arch}:{s}"


def load(runs):
    train, prof = defaultdict(list), {}
    for path in sorted(glob.glob(f"{runs}/**/*.json", recursive=True)):
        with open(path) as f:
            r = json.load(f)
        if not r.get("tag", "").startswith("e006-"):
            continue
        cfg = config_of(r["tag"])
        if r.get("kind") == "profile":
            prof[(r["spec"]["encoder"], cfg, tuple(r["img_size"]))] = r
        else:
            train[(r["dataset"]["name"], r["spec"]["encoder"], cfg)].append(r)
    return train, prof


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else float("nan"))


def pareto(rows):
    """Non-dominated (higher val mIoU, lower FLOPs) configs."""
    front = []
    for c, m, f in rows:
        if not any((m2 >= m and f2 <= f) and (m2 > m or f2 < f) for _, m2, f2 in rows):
            front.append(c)
    return front


def select(train):
    """Pre-registered: per (dataset, encoder), FORCED plus seed-0 Pareto configs by descending
    val mIoU, at most MAX_PER_CELL."""
    cells = defaultdict(list)
    for (ds, enc, cfg), runs in train.items():
        r0 = [r for r in runs if r["seed"] == 0]
        if r0:
            cells[(ds, enc)].append((cfg, r0[0]["final_val"]["miou"], r0[0]["flops_per_image"]))
    out = []
    for (ds, enc), rows in sorted(cells.items()):
        miou = {c: m for c, m, _ in rows}
        chosen = [c for c in FORCED if c in miou]
        for c in sorted(pareto(rows), key=lambda c: -miou[c]):
            if c not in chosen and len(chosen) < MAX_PER_CELL:
                chosen.append(c)
        out += [(ds, enc, *c.split(":")) for c in chosen]
    return out


def latency(prof, enc, cfg, size):
    p = prof.get((enc, cfg, size))
    if not p:
        return None, None, None
    inf = {i["batch_size"]: i for i in p["inference"] if not i.get("oom")}
    tr = p.get("train") or {}
    return inf.get(8, {}).get("median_ms"), tr.get("peak_mem_mib"), \
        (p.get("gpu_contention_at_end") or {}).get("contended")


SIZES = {"oxford_pet": (256, 256), "sen1floods11": (512, 512)}


def report(train, prof, final):
    for ds, enc in sorted({(d, e) for d, e, _ in train}):
        print(f"\n## {ds} / {enc}\n")
        print("| config | n | val mIoU | val BF1 | GFLOPs | params M | ms/batch8 | train MiB |"
              + (" test mIoU |" if final else ""))
        print("|---|---|---|---|---|---|---|---|" + ("---|" if final else ""))
        for cfg in sorted({c for d, e, c in train if (d, e) == (ds, enc)},
                          key=lambda c: (ORDER.index(c.split(":")[0]), c)):
            runs = train[(ds, enc, cfg)]
            mi, si = ms([r["final_val"]["miou"] for r in runs])
            mb, sb = ms([r["final_val"]["boundary_f1"] for r in runs])
            lat, mem, cont = latency(prof, enc, cfg, SIZES.get(ds))
            row = (f"| {cfg} | {len(runs)} | {mi:.4f} ± {si:.4f} | {mb:.4f} ± {sb:.4f} | "
                   f"{runs[0]['flops_per_image'] / 1e9:.1f} | {runs[0]['params'] / 1e6:.2f} | "
                   f"{'' if lat is None else f'{lat:.1f}' + (' (c)' if cont else '')} | "
                   f"{'' if mem is None else f'{mem:.0f}'} |")
            if final:
                t = [r["test"]["miou"] for r in runs if "test" in r]
                row += f" {ms(t)[0]:.4f} |" if t else " |"
            print(row)
        hypothesis(train, prof, ds, enc)


def hypothesis(train, prof, ds, enc):
    """U-Net out_stride 2 vs 1: mIoU drop < 1 pooled seed-SD, BF1 drop > 2 pooled seed-SDs,
    and >= 15% lower peak training memory or >= 10% lower batch-8 latency."""
    a, b = train.get((ds, enc, "unet:1"), []), train.get((ds, enc, "unet:2"), [])
    if len(a) < 2 or len(b) < 2:
        print("\nhypothesis: needs >= 2 seeds of unet:1 and unet:2")
        return

    def delta(key):
        x = [r["final_val"][key] for r in a]
        y = [r["final_val"][key] for r in b]
        sd = math.sqrt((st.variance(x) + st.variance(y)) / 2)
        return st.mean(x) - st.mean(y), sd

    dm, sm = delta("miou")
    db, sb = delta("boundary_f1")
    l1, m1, _ = latency(prof, enc, "unet:1", SIZES.get(ds))
    l2, m2, _ = latency(prof, enc, "unet:2", SIZES.get(ds))
    mem_save = 1 - m2 / m1 if m1 and m2 else float("nan")
    lat_save = 1 - l2 / l1 if l1 and l2 else float("nan")
    ok = dm < sm and db > 2 * sb and (mem_save >= 0.15 or lat_save >= 0.10)
    print(f"\nhypothesis (unet s1 -> s2): mIoU drop {dm:+.4f} (SD {sm:.4f}), BF1 drop "
          f"{db:+.4f} (SD {sb:.4f}), train-memory saving {mem_save:.0%}, latency saving "
          f"{lat_save:.0%} -> {'SUPPORTED' if ok else 'not supported'}")


def plot(train, prof, path):
    import matplotlib.pyplot as plt

    cells = sorted({(d, e) for d, e, _ in train})
    fig, axes = plt.subplots(2, len(cells), figsize=(4 * len(cells), 7), squeeze=False)
    for j, (ds, enc) in enumerate(cells):
        for i, xkind in enumerate(("GFLOPs / image", "ms / batch of 8 (bf16)")):
            ax = axes[i][j]
            for (d, e, cfg), runs in train.items():
                if (d, e) != (ds, enc):
                    continue
                arch, s = cfg.split(":")
                x = runs[0]["flops_per_image"] / 1e9 if i == 0 else \
                    latency(prof, enc, cfg, SIZES.get(ds))[0]
                if x is None:
                    continue
                m, sd = ms([r["final_val"]["miou"] for r in runs])
                ax.errorbar(x, m, yerr=0 if math.isnan(sd) else sd, fmt=MARKERS[s], ms=8,
                            color=COLOURS[ORDER.index(arch)], elinewidth=1, capsize=0)
                ax.annotate(cfg, (x, m), textcoords="offset points", xytext=(6, 2),
                            fontsize=7, color="#444444")
            ax.set_xlabel(xkind)
            ax.set_ylabel("val mIoU")
            ax.set_title(f"{ds} / {enc}", fontsize=9)
            ax.grid(alpha=0.25, linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=a)
               for a, c in zip(ORDER, COLOURS)]
    handles += [plt.Line2D([], [], marker=m, ls="", color="#666666", label=f"out_stride {s}")
                for s, m in MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--write-frontier", default=None)
    ap.add_argument("--final", action="store_true", help="also print test metrics")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    train, prof = load(a.runs)
    if a.write_frontier:
        sel = select(train)
        with open(a.write_frontier, "w") as f:
            f.write("# dataset encoder arch out_stride (E006 pre-registered selection)\n")
            f.writelines(" ".join(r) + "\n" for r in sel)
        print(f"wrote {len(sel)} frontier configs to {a.write_frontier}")
        return
    report(train, prof, a.final)
    if a.plot:
        plot(train, prof, a.plot)


if __name__ == "__main__":
    main()
