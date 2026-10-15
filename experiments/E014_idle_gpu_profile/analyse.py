"""E014 analysis: latest record per config; refuses contended records (see README)."""

import glob
import json
import statistics
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "runs")
latest = {}
for f in sorted(glob.glob(str(runs / "*.json"))):
    d = json.load(open(f))
    latest[(d["spec"]["encoder"], d["spec"]["arch"], d["precision"], d["compiled"])] = d


def contended(d):
    a = d["env"].get("gpu_contention_at_start") or {}
    b = d.get("gpu_contention_at_end") or {}
    return bool(a.get("other_compute_processes") or b.get("other_compute_processes"))


bad = sorted(k for k, d in latest.items() if contended(d))
if bad:
    print("CONTENDED (excluded):", bad)
latest = {k: d for k, d in latest.items() if not contended(d)}


def inf(d, bs):
    return next(i for i in d["inference"] if i["batch_size"] == bs)


pairs = sorted({(e, a) for e, a, _, _ in latest})
print("| encoder | decoder | prec | eager b1 ms | compiled b1 ms | eager b8 ms | compiled b8 ms "
      "| compile ×(b8) | train eager ms | train compiled ms | compile first call s (inf b1 / train) "
      "| train peak MiB eager / compiled |")
print("|" + " --- |" * 12)
speed = {p: [] for p in ("fp32", "bf16", "fp16")}
tspeed = {p: [] for p in ("fp32", "bf16", "fp16")}
prec_gain = {"bf16": [], "fp16": []}
for e, a in pairs:
    for p in ("fp32", "bf16", "fp16"):
        de, dc = latest.get((e, a, p, False)), latest.get((e, a, p, True))
        if not de or not dc:
            continue
        s = inf(de, 8)["median_ms"] / inf(dc, 8)["median_ms"]
        ts = de["train"]["median_ms"] / dc["train"]["median_ms"]
        speed[p].append(s)
        tspeed[p].append(ts)
        print(f"| {e} | {a} | {p} | {inf(de, 1)['median_ms']:.1f} | {inf(dc, 1)['median_ms']:.1f} "
              f"| {inf(de, 8)['median_ms']:.1f} | {inf(dc, 8)['median_ms']:.1f} | {s:.2f} "
              f"| {de['train']['median_ms']:.0f} | {dc['train']['median_ms']:.0f} "
              f"| {inf(dc, 1)['first_call_ms'] / 1e3:.0f} / {dc['train']['first_call_ms'] / 1e3:.0f} "
              f"| {de['train']['peak_mem_mib']:.0f} / {dc['train']['peak_mem_mib']:.0f} |")
    f32 = latest.get((e, a, "fp32", False))
    for p in ("bf16", "fp16"):
        dp = latest.get((e, a, p, False))
        if f32 and dp:
            prec_gain[p].append(inf(f32, 8)["median_ms"] / inf(dp, 8)["median_ms"])

print()
for p, v in speed.items():
    if v:
        print(f"compile speedup, inference b8, {p}: median {statistics.median(v):.2f}x, "
              f">=1.2x in {sum(x >= 1.2 for x in v)}/{len(v)}; train step median "
              f"{statistics.median(tspeed[p]):.2f}x")
for p, v in prec_gain.items():
    if v:
        print(f"{p} vs fp32 (eager, inference b8): median {statistics.median(v):.2f}x, "
              f">=1.3x in {sum(x >= 1.3 for x in v)}/{len(v)}")
