"""E040 runner: tune heat / field / both on Innaloo per level, test with B0 on the same draws.

    E028_DATA=... uv run python run40.py tune|test|decide [--procs 6] [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import e040 as F

E, B = F.E, F.B
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BEST028 = Path(os.environ.get("E028_BEST", "$WORKSPACE/ManySeg-e028/experiments/E028_vector_oracle/runs/best.json"))
TUNE_AREA, TEST_AREAS = "innaloo", ("murdoch", "northam")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
P = lambda keys, vals: [dict(zip(keys, v), snap_r=4.0) for v in itertools.product(*vals)]  # noqa: E731
GRIDS = {
    "heat": P(("thr_c", "big_dev", "agree", "merge_deg"), ((0.3, 0.5), (4.0, 8.0), (0.4, 0.6), (8.0, 15.0))),
    "field": P(("window", "big_dev", "agree", "merge_deg"), ((5, 9, 15), (4.0, 8.0), (0.4, 0.6), (8.0, 15.0))),
    "both": P(("thr_c", "window", "big_dev", "agree"), ((0.3, 0.5), (9, 15), (4.0, 8.0), (0.4, 0.6))),
}
for g in GRIDS["both"]:
    g["merge_deg"] = 8.0
_T: dict = {}


def tiles(area):
    if area not in _T:
        _T[area] = E.building_tiles(area, E.load_area(Path(os.environ["E028_DATA"]) / f"{area}.json"))
    return _T[area]


def decode(arm, a, p, p0):
    full = dict(p, b0=p0, kernel=3, simplify=p0["simplify"])
    if arm == "heat":  # same 3x3 strongest-pixel sampling as the field arms (Codex E040 #4)
        return B.decode_b3r(dict(a, direction=F.strongest_3x3(a["direction"])), full)
    return F.decode_b3f(a, dict(full, mode=arm, thr_c=p.get("thr_c", 0.3)))


def work(job):
    import torch
    torch.set_num_threads(1)
    t = tiles(job["area"])[job["tile"]]
    lv = job["level"]
    p0 = json.loads(BEST028.read_text())[f"L{lv}"]["B0"]["params"]
    a = F.corrupt_b3f(F.encode_b3f(t), t, E.LEVELS[lv], job["seed"])
    rows = []
    E.CACHE.clear()  # once per job: B0's instance labels depend only on p0 and are shared (speed fix)
    for arm, plist in job["arms"].items():
        for k, p in enumerate(plist):
            t0 = time.perf_counter()
            pred = E.decode_b0(a, p0) if arm == "B0" else decode(arm, a, p, p0)
            m = E.polygon_metrics(t, pred)
            rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm=arm, pidx=k, params=p,
                             s=time.perf_counter() - t0, score=m["ciou"] * m["det_f1"], **m))
    return rows


def run(jobs, out, procs):
    t0 = time.time()
    with open(out, "w") as f, Pool(procs, maxtasksperchild=8) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "test", "decide"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    sfx = "" if args.limit is None else f"_limit{args.limit}"
    if args.phase == "tune":
        n = len(tiles(TUNE_AREA))
        jobs = [dict(area=TUNE_AREA, tile=i, level=lv, seed=s, arms=GRIDS)
                for lv in range(4) for i in range(min(n, args.limit or n)) for s in TUNE_SEEDS]
        run(jobs, RUNS / f"tune{sfx}.jsonl", args.procs)
        acc = defaultdict(list)
        for r in map(json.loads, open(RUNS / f"tune{sfx}.jsonl")):
            acc[(r["arm"], r["level"], r["pidx"])].append(r["score"])
        best = {arm: {f"L{lv}": GRIDS[arm][max(range(len(GRIDS[arm])), key=lambda k: np.mean(acc[(arm, lv, k)]))]
                      for lv in range(4)} for arm in GRIDS}
        (RUNS / f"best{sfx}.json").write_text(json.dumps(best, indent=1))
        print(json.dumps(best, indent=1))
    elif args.phase == "test":
        best = json.loads((RUNS / f"best{sfx}.json").read_text())
        jobs = []
        for area in TEST_AREAS:
            n = len(tiles(area))
            for lv in range(4):
                arms = {"B0": [{}], **{arm: [best[arm][f"L{lv}"]] for arm in GRIDS}}
                jobs += [dict(area=area, tile=i, level=lv, seed=s, arms=arms)
                         for i in range(min(n, args.limit or n)) for s in TEST_SEEDS]
        run(jobs, RUNS / f"test{sfx}.jsonl", args.procs)
    else:
        rows = [json.loads(l) for l in open(RUNS / f"test{sfx}.jsonl")]
        m = defaultdict(list)
        for r in rows:
            m[(r["arm"], r["level"])].append(r["score"])
            m[(r["arm"], r["level"], r["area"])].append(r["score"])
        out = {f"{arm}_L{lv}": {"score": float(np.mean(m[(arm, lv)])),
                                **{a: float(np.mean(m[(arm, lv, a)])) for a in TEST_AREAS}}
               for arm in ("B0", "heat", "field", "both") for lv in range(4)}
        for k, v in out.items():
            print(k, {a: round(b, 4) for a, b in v.items()})
        s = lambda arm, lv: out[f"{arm}_L{lv}"]["score"]  # noqa: E731
        dec = {}
        for arm in ("field", "both"):
            dec[f"{arm}_beats_heat"] = all(s(arm, lv) >= s("heat", lv) + 0.01 for lv in (1, 2)) and s(arm, 0) >= s("heat", 0) - 0.01
        dec["heatmap_redundant"] = all(s("field", lv) >= s("heat", lv) - 0.005 for lv in (0, 1, 2))
        out["decision"] = dec
        print(json.dumps(dec, indent=1))
        (RUNS / f"decision{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
