"""E020 analysis: per-arm val/test metrics (mean ± SD over seeds) and the pre-registered
decisions Q1-Q5 (README)."""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
arms = defaultdict(list)
for f in glob.glob(str(E / "runs" / "*.json")):
    with open(f) as fh:
        d = json.load(fh)
    arms[d["tag"].removeprefix("e020-")].append(d)
CLS = ["timm-s2", "s2", "groups", "s1s2-stack", "s1s2-fusion", "meta", "meta-shuffled", "seg"]
REG = ["timm-frac", "frac"]


def vals(arm, key, split="final_val"):
    return [r[split][key] for r in arms[arm]]


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan"))


print("| arm | n | val mAP macro | test mAP macro | test mAP micro | test F1 macro | params | "
      "train img/s |")
print("|---|---|---|---|---|---|---|---|")
for a in CLS:
    if not arms[a]:
        continue
    (vm, vs), (tm, ts) = ms(vals(a, "map_macro")), ms(vals(a, "map_macro", "test"))
    extra = ""
    if a == "seg":
        extra = f" (seg mIoU test {st.mean(r['test']['seg']['miou'] for r in arms[a]):.4f})"
    print(f"| {a} | {len(arms[a])} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | "
          f"{st.mean(vals(a, 'map_micro', 'test')):.4f} | "
          f"{st.mean(vals(a, 'f1_macro', 'test')):.4f}{extra} | "
          f"{arms[a][0]['params'] / 1e6:.1f} M | "
          f"{st.mean(r['train_images_per_s_steady'] for r in arms[a]):.0f} |")
print("\n| arm | n | val RMSE | test RMSE | test R² mean |")
print("|---|---|---|---|---|")
for a in REG:
    if arms[a]:
        (vm, vs), (tm, ts) = ms(vals(a, "rmse")), ms(vals(a, "rmse", "test"))
        print(f"| {a} | {len(arms[a])} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | "
              f"{st.mean(vals(a, 'r2_mean', 'test')):.4f} |")


def diff(a, b, key):
    (va, sa), (vb, sb) = ms(vals(a, key)), ms(vals(b, key))
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    return va - vb, pooled, st.mean(vals(a, key, "test")) - st.mean(vals(b, key, "test"))


print()
if arms["s2"] and arms["timm-s2"]:
    d, sd, dt = diff("s2", "timm-s2", "map_macro")
    bar = 0.003 if sd < 0.0015 else 2 * sd
    print(f"Q1 cls parity s2 - timm-s2: val {100 * d:+.2f} (bar {100 * bar:.2f}), test "
          f"{100 * dt:+.2f} -> {'PARITY' if abs(d) <= bar else 'NO'}")
if arms["frac"] and arms["timm-frac"]:
    d, sd, dt = diff("frac", "timm-frac", "rmse")
    bar = max(2 * sd, 0.002)
    print(f"Q1 reg parity frac - timm-frac: val RMSE {d:+.4f} (bar {bar:.4f}), test {dt:+.4f} -> "
          f"{'PARITY' if abs(d) <= bar else 'NO'}")
for a, q in (("groups", "Q2"), ("s1s2-stack", "Q3"), ("s1s2-fusion", "Q3"), ("meta", "Q4"),
             ("meta-shuffled", "Q4 control"), ("seg", "Q5")):
    if arms[a] and arms["s2"]:
        d, sd, dt = diff(a, "s2", "map_macro")
        ok = d > 2 * sd and dt > 0
        print(f"{q} {a} - s2: val {100 * d:+.2f} (2 SD {200 * sd:.2f}), test {100 * dt:+.2f} -> "
              f"{'HELPS' if ok else 'no'}")
if arms["meta"] and arms["meta-shuffled"]:
    d, sd, dt = diff("meta", "meta-shuffled", "map_macro")
    print(f"Q4 meta - meta-shuffled: val {100 * d:+.2f} (2 SD {200 * sd:.2f}), test {100 * dt:+.2f}")
