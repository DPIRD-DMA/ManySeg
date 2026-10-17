"""E024 analysis: per-arm val/test (mean ± SD over seeds) and the pre-registered decision:
a remote-sensing weight is recommended if it beats `imagenet` by more than 2 pooled SD on val
on at least 2 of the 3 datasets (test agreeing on those) and is not worse by more than 2
pooled SD on the third."""

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
    arms[d["tag"].removeprefix("e024-")].append(d)
DATASETS = ("sen1floods11_s2", "bigearthnet", "pastis")
ARMS = ("imagenet", "ssl4eo_moco", "ssl4eo_dino", "satlas")


def missing_bands(run) -> str:
    """Bands of the weights the dataset lacks (filled with the pretraining mean). Runs before
    the runner recorded model_notes: derived from the recorded band_names and the registry."""
    from manyseg.weights import get_weights

    spec = run["spec"]
    if not spec.get("weights") or spec["weights"] in ("tv_in1k",) or not spec.get("band_names"):
        return ""
    w = get_weights(spec["encoder"], spec["weights"])
    return ", ".join(b for b in w.bands if b not in spec["band_names"]) or "-"


def metric(ds):
    return arms[f"{ds}-imagenet"][0]["primary_metric"]


def vals(ds, arm, split="final_val"):
    return [r[split][metric(ds)] for r in arms[f"{ds}-{arm}"]]


def diff(ds, a, b="imagenet"):
    va, vb = vals(ds, a), vals(ds, b)
    sd = ((st.stdev(va) ** 2 + st.stdev(vb) ** 2) / 2) ** 0.5
    d = st.mean(va) - st.mean(vb)
    dt = st.mean(vals(ds, a, "test")) - st.mean(vals(ds, b, "test"))
    return d, sd, dt


done = [ds for ds in DATASETS if all(len(arms[f"{ds}-{a}"]) == 3 for a in ARMS)]
for ds in DATASETS:
    if not arms[f"{ds}-imagenet"]:
        continue
    m = metric(ds)
    print(f"\n### {ds} ({m})\n")
    print(f"| arm | n | val {m} | test {m} | missing bands | train img/s |")
    print("|---|---|---|---|---|---|")
    for a in ARMS:
        rs = arms[f"{ds}-{a}"]
        if not rs:
            continue
        v, t = vals(ds, a), vals(ds, a, "test")
        sd = f"{st.stdev(v):.4f}" if len(v) > 1 else "-"
        missing = missing_bands(rs[0])
        ips = st.mean(r["train_images_per_s_steady"] for r in rs)
        print(f"| {a} | {len(rs)} | {st.mean(v):.4f} ± {sd} | {st.mean(t):.4f} | {missing} | "
              f"{ips:.0f} |")
    if ds in done:
        for a in ARMS[1:]:
            d, sd, dt = diff(ds, a)
            print(f"{a} - imagenet: val {100 * d:+.2f} (2 SD {200 * sd:.2f}), "
                  f"test {100 * dt:+.2f}")

if len(done) == len(DATASETS):
    print()
    for a in ARMS[1:]:
        wins = [ds for ds in DATASETS
                if diff(ds, a)[0] > 2 * diff(ds, a)[1] and diff(ds, a)[2] > 0]
        losses = [ds for ds in DATASETS if diff(ds, a)[0] < -2 * diff(ds, a)[1]]
        ok = len(wins) >= 2 and not losses
        print(f"{a}: beats imagenet on {wins or 'none'}, worse on {losses or 'none'} -> "
              f"{'RECOMMENDED' if ok else 'not recommended'}")
else:
    print(f"\n(decision needs all three datasets; complete: {done})")
