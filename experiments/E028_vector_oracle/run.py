"""E028 driver: tune decoders on innaloo, then evaluate frozen settings on murdoch and northam.

    export E028_DATA="$MANYSEG_BENCH_DATA/osm_e028"
    uv run python experiments/E028_vector_oracle/run.py tune1   # R0, R2, B0, B2
    uv run python experiments/E028_vector_oracle/run.py tune2   # R1, B1 (build on R0 / B0)
    uv run python experiments/E028_vector_oracle/run.py test
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TUNE_AREA, TEST_AREAS = "innaloo", ("murdoch", "northam")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
PRIMARY = {"roads": "apls", "buildings": "ciou"}
TASK = {"R0": "roads", "R1": "roads", "R2": "roads", "B0": "buildings", "B1": "buildings", "B2": "buildings"}


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


GRIDS = {
    "R0": grid(thr=[0.3, 0.5], spur=[10, 30], gap=[0, 25, 50], simplify=[1, 2], min_len=[20]),
    "R2": grid(thr_v=[0.2, 0.4], thr_e=[0.3, 0.6], link_r=[4, 10], gap=[0, 25, 50],
               mutual=[False], kernel=[5]),
    "B0": grid(thr=[0.4, 0.6], thr_b=[0.3, 0.5], thr_s=[0.15, 0.3], simplify=[0.5, 1, 2]),
    "B2": grid(thr_v=[0.2, 0.4], link_r=[4, 10], gap=[0, 25, 50], simplify=[0.5, 1.5],
               thr_e=[0.5], mutual=[False], kernel=[3], face_thr=[0.5]),
}


def grid_r1(r0: dict) -> list[dict]:
    return [dict(thr=r0["thr"], simplify=r0["simplify"], min_len=r0["min_len"], kernel=5, **g)
            for g in grid(thr_j=[0.2, 0.4], merge_r=[6, 12], spur=[10, 30], gap=[0, 25, 50])]


def grid_b1(b0: dict) -> list[dict]:
    return [dict(b0, **g) for g in grid(thr_c=[0.2, 0.4], snap_r=[3, 6], kernel=[3, 5])]


_TILES: dict = {}


def tiles(area: str, task: str):
    import e028 as E

    key = (area, task)
    if key not in _TILES:
        d = E.load_area(Path(os.environ["E028_DATA"]) / f"{area}.json")
        _TILES[key] = E.road_tiles(area, d) if task == "roads" else E.building_tiles(area, d)
    return _TILES[key]


def work(job: dict) -> list[dict]:
    """One (area, task, tile, condition, seed): encode + corrupt each rep once, decode each
    parameter set."""
    import torch

    import e028 as E

    torch.set_num_threads(1)
    t = tiles(job["area"], job["task"])[job["tile"]]
    cfg = job["cfg"]
    rows = []
    for rep, plist in job["reps"].items():
        tg = E.encode(t, rep)
        a = E.corrupt(tg, t, cfg, job["seed"])
        E.CACHE.clear()
        for pi, p in enumerate(plist):
            t0 = time.perf_counter()
            pred = E.DECODERS[rep](a, p)
            dt = time.perf_counter() - t0
            m = E.evaluate(t, rep, pred)
            rows.append(dict(area=t.area, task=t.task, tile=t.index, cond=job["cond"], seed=job["seed"],
                             rep=rep, pidx=pi, params=p, decode_s=dt, **tg.info, **m))
    return rows


def conditions(test: bool) -> dict[str, dict]:
    import e028 as E

    out = {f"L{k}": v for k, v in E.LEVELS.items()}
    if test:
        out.update({f"only_{f}": E.factor_only(f) for f in E.FACTORS})
    return out


def run_jobs(jobs: list[dict], out: Path, procs: int):
    t0 = time.time()
    with out.open("a") as f, Pool(procs) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"{k}/{len(jobs)} jobs, {el / 60:.1f} min, eta {el / k * (len(jobs) - k) / 60:.1f} min",
                      flush=True)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def select(rows: list[dict]) -> dict:
    """Best parameter index per (rep, cond) by mean primary metric over tiles and seeds."""
    import numpy as np

    acc: dict = {}
    for r in rows:
        k = (r["rep"], r["cond"], r["pidx"])
        acc.setdefault(k, [json.dumps(r["params"]), []])[1].append(r[PRIMARY[r["task"]]])
    best: dict = {}
    for (rep, cond, pidx), (params, vals) in acc.items():
        score = float(np.mean(vals))
        cur = best.setdefault(cond, {}).get(rep)
        if cur is None or score > cur["score"]:
            best[cond][rep] = {"score": score, "params": json.loads(params), "pidx": pidx, "n": len(vals)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune1", "tune2", "test", "select"])
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--limit-tiles", type=int, default=None, help="smoke test: first N tiles")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    tag = args.tag
    tune_path, best_path = RUNS / f"tune{tag}.jsonl", RUNS / f"best{tag}.json"
    jobs = []
    if args.phase in ("tune1", "tune2"):
        if args.phase == "tune1":
            reps_by_task = {"roads": {r: GRIDS[r] for r in ("R0", "R2")},
                            "buildings": {r: GRIDS[r] for r in ("B0", "B2")}}
        else:
            best = select(load(tune_path))
        for task in ("roads", "buildings"):
            n = len(tiles(TUNE_AREA, task))
            for ti in range(min(n, args.limit_tiles or n)):
                for cond, cfg in conditions(False).items():
                    if args.phase == "tune1":
                        reps = reps_by_task[task]
                    elif task == "roads":
                        reps = {"R1": grid_r1(best[cond]["R0"]["params"])}
                    else:
                        reps = {"B1": grid_b1(best[cond]["B0"]["params"])}
                    for s in TUNE_SEEDS:
                        jobs.append(dict(area=TUNE_AREA, task=task, tile=ti, cond=cond, cfg=cfg, seed=s, reps=reps))
        run_jobs(jobs, tune_path, args.n)
        if args.phase == "tune2":
            best_path.write_text(json.dumps(select(load(tune_path)), indent=1))
    elif args.phase == "select":
        best_path.write_text(json.dumps(select(load(tune_path)), indent=1))
    else:
        best = json.loads(best_path.read_text())
        for area in TEST_AREAS:
            for task in ("roads", "buildings"):
                n = len(tiles(area, task))
                for ti in range(min(n, args.limit_tiles or n)):
                    for cond, cfg in conditions(True).items():
                        src = best[cond] if cond in best else best["L2"]  # factor-only: level-2 params
                        reps = {r: [src[r]["params"]] for r in src if TASK[r] == task}
                        for s in TEST_SEEDS:
                            jobs.append(dict(area=area, task=task, tile=ti, cond=cond, cfg=cfg, seed=s, reps=reps))
        run_jobs(jobs, RUNS / f"test{tag}.jsonl", args.n)


if __name__ == "__main__":
    main()
