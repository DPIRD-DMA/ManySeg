"""E007b analysis: arms per dataset (mean ± SD) and the pre-registered comparisons."""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
g = defaultdict(list)
for f in glob.glob(str(E / "runs" / "*.json")):
    d = json.load(open(f))
    arm = d["tag"].split(d["dataset"]["name"] + "-")[1]
    g[(d["dataset"]["name"], arm)].append(d)


def ms(ds, arm, split="final_val"):
    v = [d[split]["miou"] for d in g[(ds, arm)]]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan")), len(v)


print("| dataset | arm | n | val mIoU | test mIoU | train img/s |")
print("|---|---|---|---|---|---|")
for (ds, arm) in sorted(g):
    vm, vs, n = ms(ds, arm)
    tm, ts, _ = ms(ds, arm, "test")
    ips = st.mean(d.get("train_images_per_s_steady") or 0 for d in g[(ds, arm)])
    print(f"| {ds} | {arm} | {n} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | {ips:.0f} |")


def diff(ds, a, b):
    (va, sa, _), (vb, sb, _) = ms(ds, a), ms(ds, b)
    ta, tb = ms(ds, a, "test")[0], ms(ds, b, "test")[0]
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    return (va - vb) * 100, pooled * 100, (ta - tb) * 100


print()
for ds in ("oxford_pet", "sen1floods11"):
    d, p, t = diff(ds, "b2x8-frozen", "b16")
    print(f"{ds}: b2x8-frozen - b16: val {d:+.2f} (1 pooled SD {p:.2f}), test {t:+.2f} -> "
          f"{'within/better' if d >= -p else 'worse'}")
    for a, b, what in (("b2x8", "b2", "accumulation at fixed BN noise"),
                       ("b2x8-frozen", "b2x8", "BN freezing at matched steps"),
                       ("b2", "b16", "batch-2 gap (default)")):
        d, p, t = diff(ds, a, b)
        print(f"  {what}: {a} - {b}: val {d:+.2f} (2 pooled SD {2 * p:.2f}), test {t:+.2f}")
