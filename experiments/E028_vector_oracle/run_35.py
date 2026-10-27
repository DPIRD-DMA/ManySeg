"""E035 driver: tune B3 on innaloo, test B0 / B2 / B3 on murdoch + northam (E028 setup).

B0 and B2 frozen at E028's settings (--best); B3's instance step uses E028's tuned B0 settings
for that level, and its own parameters are tuned (16 settings) by C-IoU x detection F1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_35"
TUNE_AREA, TEST_AREAS = "innaloo", ("murdoch", "northam")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
GRID = [dict(zip(k, v)) for k in [("thr_c", "dev", "agree", "snap_r")]
        for v in itertools.product((0.2, 0.4), (1.5, 3.0), (0.6, 0.8), (3.0, 6.0))]
_T: dict = {}
BEST: dict = {}


def _init(best):
    global BEST
    BEST = best


def tiles(area):
    import e028 as E

    if area not in _T:
        _T[area] = E.building_tiles(area, E.load_area(Path(os.environ["E028_DATA"]) / f"{area}.json"))
    return _T[area]


def work(job):
    import e028 as E
    import e035 as B

    t = tiles(job["area"])[job["tile"]]
    cfg = E.LEVELS[job["level"]]
    a3 = B.corrupt_b3(B.encode_b3(t), t, cfg, job["seed"])  # B0 channels + corner + direction
    a2 = E.corrupt(E.encode(t, "B2"), t, cfg, job["seed"])
    a2["mask"] = a3["extent"].copy()  # one corrupted footprint for all arms
    E.CACHE.clear()
    lv = f"L{job['level']}"
    p0, p2 = BEST[lv]["B0"]["params"], BEST[lv]["B2"]["params"]
    rows = []
    for arm, plist in job["arms"].items():
        for pi, p in enumerate(plist):
            t0 = time.perf_counter()
            if arm == "B0":
                pred = E.decode_b0(a3, p0)
            elif arm == "B2":
                pred = E.decode_b2(a2, p2)
            else:
                pred = B.decode_b3(a3, dict(p, b0=p0, kernel=3, simplify=p0["simplify"]))
            m = E.polygon_metrics(t, pred)
            rows.append(dict(area=t.area, tile=t.index, level=job["level"], seed=job["seed"], arm=arm, pidx=pi,
                             params=p, decode_s=time.perf_counter() - t0, score=m["ciou"] * m["det_f1"], **m))
    return rows


def run(jobs, out, procs, best):
    t0 = time.time()
    with out.open("a") as f, Pool(procs, initializer=_init, initargs=(best,)) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"{k}/{len(jobs)} jobs, {el / 60:.1f} min, eta {el / k * (len(jobs) - k) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "test"])
    ap.add_argument("--best", type=Path, required=True)
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--limit-tiles", type=int, default=None)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    best = json.loads(args.best.read_text())
    jobs = []
    if args.phase == "tune":
        n = len(tiles(TUNE_AREA))
        for ti in range(min(n, args.limit_tiles or n)):
            for lv in range(4):
                for s in TUNE_SEEDS:
                    jobs.append(dict(area=TUNE_AREA, tile=ti, level=lv, seed=s, arms={"B3": GRID}))
        out = RUNS / "tune.jsonl"
        run(jobs, out, args.n, best)
        acc: dict = {}
        for r in map(json.loads, out.open()):
            acc.setdefault((r["level"], r["pidx"]), []).append(r["score"])
        sel = {}
        for (lv, pi), v in acc.items():
            if f"L{lv}" not in sel or np.mean(v) > sel[f"L{lv}"]["score"]:
                sel[f"L{lv}"] = {"score": float(np.mean(v)), "params": GRID[pi]}
        (RUNS / "best_35.json").write_text(json.dumps(sel, indent=1))
    else:
        sel = json.loads((RUNS / "best_35.json").read_text())
        for area in TEST_AREAS:
            n = len(tiles(area))
            for ti in range(min(n, args.limit_tiles or n)):
                for lv in range(4):
                    arms = {"B0": [{}], "B2": [{}], "B3": [sel[f"L{lv}"]["params"]]}
                    for s in TEST_SEEDS:
                        jobs.append(dict(area=area, tile=ti, level=lv, seed=s, arms=arms))
        run(jobs, RUNS / "test.jsonl", args.n, best)


if __name__ == "__main__":
    main()
