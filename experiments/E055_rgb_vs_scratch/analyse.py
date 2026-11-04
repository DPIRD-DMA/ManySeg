"""E055 analysis: the registered decisions of README.md. Part A compares scratch arms with E024's
imagenet runs and CloudSEN12 (part C) with its own (pooled SD); part B compares arms paired by seed (95% CI, t, df 4)."""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).parent
T_95_DF4 = 2.776
SCRATCH = ("scratch", "scratch_fair")
DATASETS = ("sen1floods11_s2", "bigearthnet", "pastis", "cloudsen12")


def load(pattern, prefix):
    out = defaultdict(list)
    for f in glob.glob(str(pattern)):
        d = json.loads(Path(f).read_text())
        if d["tag"].startswith(prefix):
            out[d["tag"].removeprefix(prefix)].append(d)
    return out


def score(run, split):
    return run[split][run["primary_metric"]]


def water(run, split):
    return run[split]["iou"][1] if "iou" in run.get(split, {}) else float("nan")


def verdict(helps, hurts):
    return "help" if all(helps) else "hurt" if any(hurts) else "level"


def part_a():
    runs = load(E / "runs_a" / "*.json", "e055a-")
    runs.update(load(E.parent / "E024_rs_pretrained_weights" / "runs" / "*.json", "e024-"))
    runs.update(load(E / "runs_c" / "*.json", "e055c-"))  # CloudSEN12 has its own imagenet arm
    print("## Parts A and C\n")
    print("| dataset | arm | n | val | test | val diff vs imagenet (2 SD) | test diff |")
    print("|---|---|---|---|---|---|---|")
    calls = {}
    for ds in DATASETS:
        base = runs[f"{ds}-imagenet"]
        if not base:
            continue
        bv, bt = [score(r, "final_val") for r in base], [score(r, "test") for r in base]
        print(f"| {ds} | imagenet | {len(base)} | {st.mean(bv):.4f} ± {st.stdev(bv):.4f} | "
              f"{st.mean(bt):.4f} | | |")
        helps, hurts, complete = [], [], True
        for arm in SCRATCH:
            rs = runs[f"{ds}-{arm}"]
            if len(rs) < 3:
                complete = False
                continue
            v, t = [score(r, "final_val") for r in rs], [score(r, "test") for r in rs]
            sd = ((st.stdev(v) ** 2 + st.stdev(bv) ** 2) / 2) ** 0.5
            d, dt = st.mean(bv) - st.mean(v), st.mean(bt) - st.mean(t)  # + = RGB weights ahead
            helps.append(d > 2 * sd and dt > 0)
            hurts.append(d < -2 * sd and dt < 0)
            print(f"| {ds} | {arm} | {len(rs)} | {st.mean(v):.4f} ± {st.stdev(v):.4f} | "
                  f"{st.mean(t):.4f} | {100 * d:+.2f} ({200 * sd:.2f}) | {100 * dt:+.2f} |")
        calls[ds] = verdict(helps, hurts) if complete else "pending"
    print("\nRGB weights, per dataset:", calls)
    return calls


def part_b():
    runs = load(E / "runs_b" / "*.json", "e055b-")
    print("\n## Part B\n")
    print("| chips | arm | n | val mIoU | test mIoU | Bolivia mIoU | test water IoU |")
    print("|---|---|---|---|---|---|---|")
    calls = {}
    for n in (12, 25, 63, 252):
        by_arm = {a: {r["seed"]: r for r in runs[f"n{n}-{a}"]} for a in ("imagenet", *SCRATCH)}
        for a, rs in by_arm.items():
            if not rs:
                continue
            rs = list(rs.values())
            v = [score(r, "final_val") for r in rs]
            print(f"| {n} | {a} | {len(rs)} | {st.mean(v):.4f} ± "
                  f"{st.stdev(v) if len(v) > 1 else 0:.4f} | "
                  f"{st.mean(score(r, 'test') for r in rs):.4f} | "
                  f"{st.mean(score(r, 'bolivia') for r in rs):.4f} | "
                  f"{st.mean(water(r, 'test') for r in rs):.4f} |")
        helps, hurts, complete = [], [], True
        for arm in SCRATCH:
            seeds = sorted(set(by_arm["imagenet"]) & set(by_arm[arm]))
            if len(seeds) < 5:
                complete = False
                continue
            base = by_arm["imagenet"]
            d = [score(base[s], "final_val") - score(by_arm[arm][s], "final_val") for s in seeds]
            dt = st.mean(score(base[s], "test") - score(by_arm[arm][s], "test") for s in seeds)
            half = T_95_DF4 * st.stdev(d) / len(d) ** 0.5
            helps.append(st.mean(d) - half > 0 and dt > 0)
            hurts.append(st.mean(d) + half < 0 and dt < 0)
            print(f"n{n} imagenet - {arm}: val {100 * st.mean(d):+.2f} ± {100 * half:.2f}, "
                  f"test {100 * dt:+.2f}")
        calls[n] = verdict(helps, hurts) if complete else "pending"
    print("\nRGB weights, per size:", calls)
    return calls


if __name__ == "__main__":
    part_a()
    part_b()
