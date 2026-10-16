"""E021 analysis: per-arm val/test (mean ± SD over seeds) and the pre-registered decisions
Q1 (time series vs dates as channels) and Q2 (attention vs mean) over both datasets."""

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
    arms[d["tag"].removeprefix("e021-")].append(d)
DATASETS = {"pastis_ts": "miou", "ai4boundaries_s2": "pq"}
ARMS = ("single", "stack", "mean", "attention")


def vals(ds, arm, split="final_val"):
    return [r[split][DATASETS[ds]] for r in arms[f"{ds}-{arm}"]]


def diff(ds, a, b):
    va, vb = vals(ds, a), vals(ds, b)
    sd = ((st.stdev(va) ** 2 + st.stdev(vb) ** 2) / 2) ** 0.5
    dt = st.mean(vals(ds, a, "test")) - st.mean(vals(ds, b, "test"))
    d = st.mean(va) - st.mean(vb)
    return d, sd, dt, d > 2 * sd and dt > 0


for ds, metric in DATASETS.items():
    if not arms[f"{ds}-stack"]:
        continue
    print(f"\n### {ds} ({metric})\n")
    print(f"| arm | n | val {metric} | test {metric} | best-epoch val | train img/s |")
    print("|---|---|---|---|---|---|")
    for a in ARMS:
        rs = arms[f"{ds}-{a}"]
        if rs:
            v, t = vals(ds, a), vals(ds, a, "test")
            best = st.mean(r[f"best_val_{metric}"] for r in rs)
            ips = st.mean(r["train_images_per_s_steady"] for r in rs)
            print(f"| {a} | {len(rs)} | {st.mean(v):.4f} ± {st.stdev(v):.4f} | "
                  f"{st.mean(t):.4f} | {best:.4f} | {ips:.0f} |")
    for a, b in (("mean", "stack"), ("attention", "stack"), ("attention", "mean"),
                 ("stack", "single")):
        d, sd, dt, ok = diff(ds, a, b)
        print(f"{a} - {b}: val {100 * d:+.2f} (2 SD {200 * sd:.2f}), test {100 * dt:+.2f}"
              f"{' -> beats' if ok else ''}")

done = [ds for ds in DATASETS if arms[f"{ds}-stack"] and arms[f"{ds}-attention"]]
if len(done) == 2:
    q1 = any(all(diff(ds, a, "stack")[3] for ds in done) for a in ("mean", "attention"))
    q2 = all(diff(ds, "attention", "mean")[3] for ds in done)
    print(f"\nQ1 time series beats dates-as-channels on both datasets: {'YES' if q1 else 'no'}")
    print(f"Q2 attention beats mean on both datasets: {'YES' if q2 else 'no'}")
else:
    print(f"\n(decisions need both datasets; have {done})")
