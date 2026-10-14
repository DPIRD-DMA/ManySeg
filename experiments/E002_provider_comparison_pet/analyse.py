"""Summarise E002: mean ± SD over seeds per (recipe, provider), Welch t vs ManySeg."""

import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
runs = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / "runs" / "*.json")))]
g = defaultdict(list)
for r in runs:
    g[(r["recipe"]["name"], r["provider"])].append(r)


def ms(v):
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb) if va + vb > 0 else float("nan")


print("| recipe | provider | n | params (M) | val mIoU | val boundary F1 | test mIoU | "
      "epoch time (s, contended) | t vs manyseg (val) | t vs manyseg (test) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (rec, prov), rs in sorted(g.items()):
    val = [r["final_val"]["miou"] for r in rs]
    bf = [r["final_val"]["boundary_f1"] for r in rs]
    test = [r["test"]["miou"] for r in rs if "test" in r]
    ep = [st.median(h["epoch_time_s"] for h in r["history"][1:]) for r in rs]
    ref = [r["final_val"]["miou"] for r in g[(rec, "manyseg")]]
    ref_t = [r["test"]["miou"] for r in g[(rec, "manyseg")] if "test" in r]
    t = "" if prov == "manyseg" else f"{welch(val, ref):+.2f}"
    tt = "" if prov == "manyseg" else f"{welch(test, ref_t):+.2f}"
    print(f"| {rec} | {prov} | {len(rs)} | {rs[0]['params'] / 1e6:.2f} | {ms(val)} | {ms(bf)} | "
          f"{ms(test) if test else 'n/a'} | {st.mean(ep):.1f} | {t} | {tt} |")
for prov in ("manyseg", "smp", "fastai"):
    a = [r["final_val"]["miou"] for r in g[("fastai_like", prov)]]
    b = [r["final_val"]["miou"] for r in g[("standard", prov)]]
    if a and b:
        print(f"\nrecipe effect for {prov}: fastai_like - standard = "
              f"{st.mean(a) - st.mean(b):+.4f} (Welch t {welch(a, b):+.1f})")
fps = {r["weights_fingerprint"] for r in runs}
print(f"\nencoder weight fingerprints across all runs: {fps}")
