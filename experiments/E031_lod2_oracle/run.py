"""E031 driver: tune on delft, test frozen settings on leidscherijn and staphorst.

    export E031_DATA="$MANYSEG_BENCH_DATA/raw/3dbag"
    uv run python run.py tune && uv run python run.py test
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
RUNS = HERE / "runs"
TUNE_AREA, TEST_AREAS = "delft", ("leidscherijn", "staphorst")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


GRIDS = {
    "A0": grid(smooth=[0.5, 1.5], tg=[0.03, 0.06, 0.12], tz=[0.5], min_px=[30, 100], simplify=[1.0, 2.0]),
    "A1": grid(vertex_threshold=[0.1, 0.2, 0.3], link_radius=[6.0, 10.0], gap=[0.0, 25.0], simplify=[1.0, 1.5]),
    "LoD1": [{}],
}
_TILES: dict = {}


def tiles(area):
    import e031 as E

    if area not in _TILES:
        _TILES[area] = E.make_tiles(area, E.load_faces(Path(os.environ["E031_DATA"]) / area))
    return _TILES[area]


def work(job):
    import e031 as E

    t = tiles(job["area"])[job["tile"]]
    a = E.corrupt(E.encode(t), t, E.LEVELS[job["level"]], job["seed"])
    rows = []
    for arm, plist in job["arms"].items():
        for pi, p in enumerate(plist):
            t0 = time.perf_counter()
            faces, planes = E.DECODERS[arm](a, p)
            dt = time.perf_counter() - t0
            m = E.metrics(t, faces, planes)
            rows.append(dict(area=t.area, tile=t.index, level=job["level"], seed=job["seed"], arm=arm,
                             pidx=pi, params=p, decode_s=dt, score=E.score(m), **m))
    return rows


def run(jobs, out: Path, procs: int):
    t0 = time.time()
    with out.open("a") as f, Pool(procs) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"{k}/{len(jobs)} jobs, {el / 60:.1f} min, eta {el / k * (len(jobs) - k) / 60:.1f} min", flush=True)


def select(rows) -> dict:
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["level"], r["arm"], r["pidx"]), [r["params"], []])[1].append(r["score"])
    best: dict = {}
    for (lv, arm, pidx), (params, vals) in acc.items():
        cur = best.setdefault(f"L{lv}", {}).get(arm)
        s = float(np.mean(vals))
        if cur is None or s > cur["score"]:
            best[f"L{lv}"][arm] = {"score": s, "params": params, "pidx": pidx}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "test"])
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--limit-tiles", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    tune_path, best_path = RUNS / f"tune{args.tag}.jsonl", RUNS / f"best{args.tag}.json"
    jobs = []
    if args.phase == "tune":
        n = len(tiles(TUNE_AREA))
        for ti in range(min(n, args.limit_tiles or n)):
            for lv in range(4):
                for s in TUNE_SEEDS:
                    jobs.append(dict(area=TUNE_AREA, tile=ti, level=lv, seed=s,
                                     arms={k: GRIDS[k] for k in ("A0", "A1")}))
        run(jobs, tune_path, args.n)
        best_path.write_text(json.dumps(select([json.loads(x) for x in tune_path.open()]), indent=1))
    else:
        best = json.loads(best_path.read_text())
        for area in TEST_AREAS:
            n = len(tiles(area))
            for ti in range(min(n, args.limit_tiles or n)):
                for lv in range(4):
                    arms = {k: [best[f"L{lv}"][k]["params"]] for k in ("A0", "A1")} | {"LoD1": [{}]}
                    for s in TEST_SEEDS:
                        jobs.append(dict(area=area, tile=ti, level=lv, seed=s, arms=arms))
        run(jobs, RUNS / f"test{args.tag}.jsonl", args.n)


if __name__ == "__main__":
    main()
