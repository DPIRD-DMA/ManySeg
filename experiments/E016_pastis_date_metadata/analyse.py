"""E016 analysis: per-arm mean ± SD and the pre-registered comparisons (Q1a–d, Q2)."""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
arms = defaultdict(list)
for f in glob.glob(str(E / "runs" / "*.json")):
    d = json.load(open(f))
    arms[d["tag"]].append(d)
CLASSES = json.load(open(next(iter(glob.glob(str(E / "runs" / "*.json"))))))["dataset"].get(
    "class_names") or []
WINTER = [c for c in ("soft_winter_wheat", "winter_barley", "winter_triticale",
                      "winter_durum_wheat", "mixed_cereal") if c in CLASSES]
SUMMER = [c for c in ("corn", "sunflower", "soybeans", "sorghum") if c in CLASSES]


def ms(tag, split="final_val"):
    v = [d[split]["miou"] for d in arms[tag]]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan")), len(v)


def group_iou(tag, names, split="test"):
    idx = [CLASSES.index(c) for c in names]
    return st.mean(st.mean(d[split]["iou"][i] for i in idx) for d in arms[tag])


print("| arm | n | val mIoU | test mIoU | test winter cereals IoU | test summer crops IoU | params |")
print("|---|---|---|---|---|---|---|")
for tag in sorted(arms):
    (vm, vs, n), (tm, ts, _) = ms(tag), ms(tag, "test")
    print(f"| {tag} | {n} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | "
          f"{group_iou(tag, WINTER):.4f} | {group_iou(tag, SUMMER):.4f} | "
          f"{arms[tag][0]['params'] / 1e6:.1f} M |")


def diff(a, b):
    (va, sa, _), (vb, sb, _) = ms(a), ms(b)
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    dv, dt = (va - vb) * 100, (ms(a, "test")[0] - ms(b, "test")[0]) * 100
    return dv, 2 * pooled * 100, dt


print()
for a, b, q in (("e016-film", "e016-none", "Q1a FiLM vs none"),
                ("e016-film-shuffled", "e016-none", "Q1b shuffled vs none (must NOT pass)"),
                ("e016-film", "e016-film-tileshuffled", "Q1c FiLM vs tile-shuffled"),
                ("e016-film-tileshuffled", "e016-none", "Q1c tile-shuffled vs none"),
                ("e016-film", "e016-channels", "Q1d FiLM vs channels"),
                ("e016-groups-shared", "e016-none", "Q2 band groups vs stack")):
    if a in arms and b in arms:
        dv, bar, dt = diff(a, b)
        ok = dv > bar and dt > 0
        print(f"{q}: val {dv:+.2f} (2 pooled SD {bar:.2f}), test {dt:+.2f} -> "
              f"{'PASS' if ok else 'no'}")
        if a.startswith("e016-film") and b == "e016-none":
            print(f"   test winter cereals {100 * (group_iou(a, WINTER) - group_iou(b, WINTER)):+.2f}, "
                  f"summer crops {100 * (group_iou(a, SUMMER) - group_iou(b, SUMMER)):+.2f}")
