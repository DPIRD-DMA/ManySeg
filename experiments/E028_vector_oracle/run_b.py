"""E028b driver: tune cov for B2F / B2M on innaloo, test B0, B2, B2F, B2M on murdoch + northam.

    export E028_DATA="$MANYSEG_BENCH_DATA/osm_e028"
    uv run python run_b.py tune && uv run python run_b.py test
B0 / B2 parameters come from E028's frozen runs/best.json (pass --best).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_b"
TUNE_AREA, TEST_AREAS = "innaloo", ("murdoch", "northam")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
COVS = (0.7, 0.8, 0.9, 0.95)
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
    import e028b as B

    t = tiles(job["area"])[job["tile"]]
    cfg = E.LEVELS[job["level"]]
    a0 = E.corrupt(E.encode(t, "B0"), t, cfg, job["seed"])
    a2 = E.corrupt(E.encode(t, "B2"), t, cfg, job["seed"])
    E.CACHE.clear()
    lv = f"L{job['level']}"
    p0, p2 = BEST[lv]["B0"]["params"], BEST[lv]["B2"]["params"]
    rows = []
    for arm, plist in job["arms"].items():
        for pi, p in enumerate(plist):
            t0 = time.perf_counter()
            if arm == "B0":
                pred = E.decode_b0(a0, p0)
            elif arm == "B2":
                pred = E.decode_b2(a2, p2)
            else:
                pred = (B.decode_b2f if arm == "B2F" else B.decode_b2m)(a2, a0, {"b0": p0, "b2": p2, "cov": p["cov"]})
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
    ap.add_argument("--best", type=Path, required=True, help="E028 runs/best.json")
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
                    jobs.append(dict(area=TUNE_AREA, tile=ti, level=lv, seed=s,
                                     arms={a: [{"cov": c} for c in COVS] for a in ("B2F", "B2M")}))
        out = RUNS / "tune.jsonl"
        run(jobs, out, args.n, best)
        acc: dict = {}
        for r in map(json.loads, out.open()):
            acc.setdefault((r["level"], r["arm"], r["params"]["cov"]), []).append(r["score"])
        sel: dict = {}
        for (lv, arm, cov), v in acc.items():
            cur = sel.setdefault(f"L{lv}", {}).get(arm)
            if cur is None or np.mean(v) > cur["score"]:
                sel[f"L{lv}"][arm] = {"score": float(np.mean(v)), "cov": cov}
        (RUNS / "best_b.json").write_text(json.dumps(sel, indent=1))
    else:
        sel = json.loads((RUNS / "best_b.json").read_text())
        for area in TEST_AREAS:
            n = len(tiles(area))
            for ti in range(min(n, args.limit_tiles or n)):
                for lv in range(4):
                    arms = {"B0": [{}], "B2": [{}]} | {a: [{"cov": sel[f"L{lv}"][a]["cov"]}] for a in ("B2F", "B2M")}
                    for s in TEST_SEEDS:
                        jobs.append(dict(area=area, tile=ti, level=lv, seed=s, arms=arms))
        run(jobs, RUNS / "test.jsonl", args.n, best)


if __name__ == "__main__":
    main()
