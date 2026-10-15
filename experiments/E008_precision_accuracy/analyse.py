"""E008 analysis: paired Δ test mIoU vs fp32 per variant and dataset (mean ± SD over seeds).

PT2E variants come from runs/eval_rr (fixed default quantizer, reduce_range on non-VNNI CPUs);
the first-pass PT2E numbers in runs/eval are reported separately as the bug evidence."""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
train = {json.load(open(f))["run_id"]: json.load(open(f))
         for f in glob.glob(str(E / "runs" / "*.json"))}


def load(sub):
    out = defaultdict(dict)  # source run -> variant -> result
    for f in sorted(glob.glob(str(E / "runs" / sub / "*.json"))):
        d = json.load(open(f))
        for v, r in d["variants"].items():
            out[d["source_run"]][v] = r
    return out


first, fixed = load("eval"), load("eval_rr")
rows = defaultdict(lambda: defaultdict(list))  # (dataset, variant) -> metric lists
for run, variants in first.items():
    ds = train[run]["dataset"]["name"]
    ref = variants["fp32"]["test"]["miou"]
    for v, r in variants.items():
        src = fixed[run].get(v, None) if v.startswith("int8_pt2e") else r
        if src is None or "test" not in src:
            continue
        rows[(ds, v)]["delta"].append((src["test"]["miou"] - ref) * 100)
        rows[(ds, v)]["agree"].append(src["test"]["agreement"] * 100)
        if v.startswith("int8_pt2e") and "test" in r:
            rows[(ds, v + " [first pass, no reduce_range]")]["delta"].append(
                (r["test"]["miou"] - ref) * 100)
            rows[(ds, v + " [first pass, no reduce_range]")]["agree"].append(
                r["test"]["agreement"] * 100)

print("| dataset | variant | n | Δ test mIoU vs fp32 (pts) | worst seed | agreement % |")
print("|---|---|---|---|---|---|")
for (ds, v), r in sorted(rows.items()):
    d, a = r["delta"], r["agree"]
    sd = st.stdev(d) if len(d) > 1 else float("nan")
    print(f"| {ds} | {v} | {len(d)} | {st.mean(d):+.2f} ± {sd:.2f} | {min(d):+.2f} | "
          f"{st.mean(a):.2f} |")
print()
for v in ("bf16", "fp16", "int8_pt2e", "int8_pt2e_compiled", "int8_weight_only"):
    ok = all(rows[(ds, v)]["delta"] and st.mean(rows[(ds, v)]["delta"]) >= -0.3
             and min(rows[(ds, v)]["delta"]) >= -1.0 for ds in ("oxford_pet", "sen1floods11"))
    print(f"{v}: {'SAFE' if ok else 'not safe'} (rule: mean drop <= 0.3 pts on both, no seed > 1)")
