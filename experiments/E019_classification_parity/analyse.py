"""E019 / E019b analysis: ManySeg vs timm classifiers per dataset x encoder (val/test accuracy,
macro F1), the pre-registered parity rule, and the E019b head-init comparison."""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
runs = defaultdict(list)
for f in glob.glob(str(E / "runs" / "*.json")):
    with open(f) as fh:
        d = json.load(fh)
    arm, enc = d["tag"].removeprefix("e019-").rsplit("-", 1)  # timm | manyseg | manyseg-init02
    runs[(d["dataset"]["name"], enc, arm)].append(d)
ENCODERS = ("resnet50", "convnext_tiny", "pvt_v2_b0")
DATASETS = ("eurosat", "oxford_pet_breeds")


def acc(ds, enc, arm, split="final_val"):
    return [r[split]["accuracy"] for r in runs[(ds, enc, arm)]]


def gap(ds, enc, arm, split):
    a, b = acc(ds, enc, arm, split), acc(ds, enc, "timm", split)
    sd = ((st.stdev(a) ** 2 + st.stdev(b) ** 2) / 2) ** 0.5
    return (st.mean(a) - st.mean(b)) * 100, sd * 100


for arm in ("manyseg", "manyseg-init02"):
    if not any(k[2] == arm for k in runs):
        continue
    print(f"\n### {arm} vs timm\n")
    print("| dataset | encoder | n | val (ManySeg / timm) | val gap | bar | parity | test gap "
          "(pooled SD) | test macro F1 (ManySeg / timm) |")
    print("|---|---|---|---|---|---|---|---|---|")
    vg, tg = [], []
    for ds in DATASETS:
        for enc in ENCODERS:
            if not runs[(ds, enc, arm)]:
                continue
            dv, sv = gap(ds, enc, arm, "final_val")
            dt, s_t = gap(ds, enc, arm, "test")
            bar = 0.3 if sv < 0.15 else 2 * sv
            f1 = [st.mean(r["test"]["macro_f1"] for r in runs[(ds, enc, a)])
                  for a in (arm, "timm")]
            vg.append(dv)
            tg.append(dt)
            print(f"| {ds} | {enc} | {len(runs[(ds, enc, arm)])} | "
                  f"{st.mean(acc(ds, enc, arm)):.4f} / {st.mean(acc(ds, enc, 'timm')):.4f} | "
                  f"{dv:+.2f} | {bar:.2f} | {'yes' if abs(dv) <= bar else 'NO'} | "
                  f"{dt:+.2f} ({s_t:.2f}) | {f1[0]:.4f} / {f1[1]:.4f} |")
    print(f"\nmean gap over {len(vg)} cells: val {st.mean(vg):+.2f}, test {st.mean(tg):+.2f}")
