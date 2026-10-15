"""E018 analysis: per-config accuracy (mean ± SD over available seeds) joined with compiled bf16
latency (profile stage), and the pre-registered Pareto front per dataset.

Front rule: a config is dominated if another config has higher mean val mIoU by more than 1
pooled SD (0.5 points when only one seed exists) AND lower latency. `--candidates` prints the
stage-2 CONFIRM list: configs on, or within 0.5 val mIoU of, either dataset's front, plus the
reference resnet34-unet.
"""

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
WEIGHTS = {"resnet18": "tv_in1k", "resnet34": "tv_in1k", "efficientnet_b0": "ra_in1k",
           "pvt_v2_b0": "in1k"}
runs = defaultdict(list)
for f in glob.glob(str(E / "runs" / "*.json")):
    with open(f) as fh:
        d = json.load(fh)
    runs[(d["dataset"]["name"], d["tag"].removeprefix("e018-"))].append(d)
lat, flops = {}, {}
for f in glob.glob(str(E / "runs" / "profile" / "*.json")):
    with open(f) as fh:
        d = json.load(fh)
    cfg = d["tag"].removeprefix("e018-")
    lat[cfg] = next(i["median_ms"] for i in d["inference"] if i["batch_size"] == 8)
    flops[cfg] = d["flops_per_image"] / 1e9


def stats(ds, cfg, split="final_val"):
    v = [r[split]["miou"] for r in runs[(ds, cfg)]]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else None), len(v)


def front(ds):
    cfgs = [c for d, c in runs if d == ds]
    on = []
    for c in cfgs:
        m, s, _ = stats(ds, c)
        dominated = False
        for o in cfgs:
            if o == c or o not in lat or c not in lat:
                continue
            mo, so, _ = stats(ds, o)
            margin = 0.005 if s is None or so is None else ((s ** 2 + so ** 2) / 2) ** 0.5
            if mo - m > margin and lat[o] < lat[c]:
                dominated = True
                break
        if not dominated:
            on.append(c)
    return cfgs, on


for ds in ("sen1floods11", "oxford_pet"):
    cfgs, on = front(ds)
    print(f"\n### {ds}\n")
    print("| config | n | val mIoU | test mIoU | latency b8 (ms) | GFLOPs | front |")
    print("|---|---|---|---|---|---|---|")
    for c in sorted(cfgs, key=lambda c: lat.get(c, 1e9)):
        m, s, n = stats(ds, c)
        tm, _, _ = stats(ds, c, "test")
        sd = f" ± {s:.4f}" if s is not None else ""
        print(f"| {c} | {n} | {m:.4f}{sd} | {tm:.4f} | {lat.get(c, float('nan')):.1f} | "
              f"{flops.get(c, float('nan')):.1f} | {'**yes**' if c in on else ''} |")

if "--candidates" in sys.argv:
    keep = {"resnet34-unet"}
    for ds in ("sen1floods11", "oxford_pet"):
        cfgs, on = front(ds)
        best = {c: stats(ds, c)[0] for c in on}
        for c in cfgs:
            m = stats(ds, c)[0]
            # within 0.5 val mIoU of a front member that is no faster
            if c in on or any(best[f] - m <= 0.005 and lat.get(f, 1e9) >= lat.get(c, 1e9)
                              for f in on):
                keep.add(c)
    out = []
    for ds in ("sen1floods11", "oxford_pet"):
        for c in sorted(keep):
            enc, dec = c.split("-", 1)
            out.append(f"{ds}:{enc}:{WEIGHTS[enc]}:{dec}")
    print("\nCONFIRM=" + repr(" ".join(out)))
