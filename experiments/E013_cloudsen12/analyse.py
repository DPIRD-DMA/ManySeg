"""E013 analysis: per-arm mean ± SD and the pre-registered comparisons against e013-s2."""

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "runs")
CLASSES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
arms = defaultdict(list)
for f in glob.glob(str(runs / "*.json")):
    d = json.load(open(f))
    if d.get("kind") != "train":
        continue
    arms[d["tag"]].append(d)


def m(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else float("nan"))


def stat(tag, split, key="miou", cls=None):
    v = [d[split]["iou"][cls] if cls is not None else d[split][key] for d in arms[tag]]
    return m(v)


print("| arm | n | val mIoU | test mIoU | test thin IoU | test shadow IoU | val bF1 | params | img/s |")
print("|---|---|---|---|---|---|---|---|---|")
for tag in sorted(arms):
    ds = arms[tag]
    vm, vs = stat(tag, "final_val")
    tm, ts = stat(tag, "test")
    th = stat(tag, "test", cls=2)[0]
    sh = stat(tag, "test", cls=3)[0]
    bf = st.mean(d["final_val"]["boundary_f1"] for d in ds)
    ips = st.mean(d.get("train_images_per_s_steady") or 0 for d in ds)
    print(f"| {tag} | {len(ds)} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | {th:.4f} | "
          f"{sh:.4f} | {bf:.3f} | {ds[0]['params'] / 1e6:.1f} M | {ips:.0f} |")

base = "e013-s2"


def compare(tag, ref=base):
    if tag not in arms or ref not in arms:
        return
    (va, sa), (vb, sb) = stat(tag, "final_val"), stat(ref, "final_val")
    (ta, _), (tb, _) = stat(tag, "test"), stat(ref, "test")
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    d, dt = (va - vb) * 100, (ta - tb) * 100
    ok = d > 2 * pooled * 100 and dt > 0
    sh = (stat(tag, "test", cls=3)[0] - stat(ref, "test", cls=3)[0]) * 100
    th = (stat(tag, "test", cls=2)[0] - stat(ref, "test", cls=2)[0]) * 100
    print(f"{tag} vs {ref}: val {d:+.2f} pts (2 pooled SD {2 * pooled * 100:.2f}), test {dt:+.2f}, "
          f"test shadow IoU {sh:+.2f}, thin {th:+.2f} -> {'PASS' if ok else 'no'}")


print()
for tag in ("e013-aux-early", "e013-aux-mid", "e013-meta-film", "e013-meta-channels",
            "e013-meta-film-nogeo", "e013-unet-cross"):
    compare(tag)
compare("e013-meta-film", "e013-meta-channels")
compare("e013-meta-film-nogeo", "e013-meta-film")
print("\ncode versions:", {d["env"]["code_version"] for ds in arms.values() for d in ds})
