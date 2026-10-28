"""E041 runner: tune R5 on Innaloo per level, then test R0, R2, R4 (E039 tuned) and R5 on the same draws.

    E028_DATA=... E039_BEST=<E039 runs/best.json> uv run python run41.py tune|test|decide [--procs 6] [--limit N]
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

import e039 as R
import e041 as R5

E = R.E
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BEST028 = Path("$WORKSPACE/ManySeg-e028/experiments/E028_vector_oracle/runs/best.json")
TUNE_AREA, TEST_AREAS = "innaloo", ("murdoch", "northam")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
GRID = [dict(thr_k=t, spacing=sp, thr_e=te, edge_src=es, thr_n=0.3, r_cross=8.0)
        for t, sp, te, es in itertools.product((0.2, 0.4), (10.0, 14.0), (0.2, 0.35), ("line", "mask"))]
_T: dict = {}


def tiles(area):
    if area not in _T:
        _T[area] = E.road_tiles(area, E.load_area(Path(os.environ["E028_DATA"]) / f"{area}.json"))
    return _T[area]


def best028():
    return json.loads(BEST028.read_text())


def work(job):
    import torch
    torch.set_num_threads(1)
    t = tiles(job["area"])[job["tile"]]
    lv = job["level"]
    cfg = E.LEVELS[lv]
    b = best028()[f"L{lv}"]
    gap = b["R0"]["params"]["gap"]
    rows = []
    tg5 = R5.encode_r5(t)
    a5 = R5.corrupt_r5(tg5, t, cfg, job["seed"])
    for k, p in enumerate(job["r5"]):
        E.CACHE.clear()
        t0 = time.perf_counter()
        pred = R5.decode_r5(a5, dict(p, gap=gap))
        rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm="R5", pidx=k, params=p,
                         s=time.perf_counter() - t0, **E.road_metrics(t, pred)))
    if job.get("baselines"):
        for rep in ("R0", "R2"):
            a = E.corrupt(E.encode(t, rep), t, cfg, job["seed"])
            E.CACHE.clear()
            pred = E.DECODERS[rep](a, b[rep]["params"])
            rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm=rep, pidx=0,
                             params=b[rep]["params"], s=0.0, **E.road_metrics(t, pred)))
        if job.get("r4"):
            a4 = R.corrupt_r4(R.encode_r4(t), t, cfg, job["seed"])
            E.CACHE.clear()
            pred = R.decode_r4(a4, dict(job["r4"], r0=b["R0"]["params"], gap=gap))
            rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm="R4", pidx=0,
                             params=job["r4"], s=0.0, **E.road_metrics(t, pred)))
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


def load(p):
    return [json.loads(l) for l in open(p)]


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
        jobs = [dict(area=TUNE_AREA, tile=i, level=lv, seed=s, r5=GRID)
                for lv in range(4) for i in range(min(n, args.limit or n)) for s in TUNE_SEEDS]
        run(jobs, RUNS / f"tune{sfx}.jsonl", args.procs)
        acc = defaultdict(list)
        for r in load(RUNS / f"tune{sfx}.jsonl"):
            acc[(r["level"], r["pidx"])].append(r["apls"])
        best = {f"L{lv}": GRID[max(range(len(GRID)), key=lambda k: np.mean(acc[(lv, k)]))] for lv in range(4)}
        (RUNS / f"best{sfx}.json").write_text(json.dumps(best, indent=1))
        print(json.dumps(best, indent=1))
    elif args.phase == "test":
        best = json.loads((RUNS / f"best{sfx}.json").read_text())
        b39 = json.loads(Path(os.environ["E039_BEST"]).read_text())
        jobs = []
        for area in TEST_AREAS:
            n = len(tiles(area))
            jobs += [dict(area=area, tile=i, level=lv, seed=s, r5=[best[f"L{lv}"]], r4=b39[f"L{lv}"], baselines=True)
                     for lv in range(4) for i in range(min(n, args.limit or n)) for s in TEST_SEEDS]
        run(jobs, RUNS / f"test{sfx}.jsonl", args.procs)
    else:
        rows = load(RUNS / f"test{sfx}.jsonl")
        m = defaultdict(list)
        cr = defaultdict(lambda: [0, 0])
        for r in rows:
            for k in ("apls", "junction_f1"):
                m[(r["arm"], r["level"], k)].append(r[k])
                m[(r["arm"], r["level"], k, r["area"])].append(r[k])
            cr[(r["arm"], r["level"])][0] += r["cross_ok"]
            cr[(r["arm"], r["level"])][1] += r["cross_n"]
        out = {}
        for arm in ("R0", "R2", "R4", "R5"):
            for lv in range(4):
                o = {k: float(np.mean(m[(arm, lv, k)])) for k in ("apls", "junction_f1")}
                o.update({f"{k}_{a}": float(np.mean(m[(arm, lv, k, a)])) for k in ("apls",) for a in TEST_AREAS})
                ok, n = cr[(arm, lv)]
                o["crossings_kept"] = ok / max(n, 1)
                out[f"{arm}_L{lv}"] = o
                print(arm, f"L{lv}", {k: round(v, 3) for k, v in o.items()})
        g = lambda arm, lv, k: out[f"{arm}_L{lv}"][k]  # noqa: E731
        rule1 = all(g("R5", lv, "junction_f1") >= g("R0", lv, "junction_f1") + 0.10 for lv in (0, 1)) and \
            g("R5", 0, "crossings_kept") >= 0.5
        rule2 = all(g("R5", lv, "apls") >= g("R0", lv, "apls") - 0.01 for lv in (1, 2))
        out["decision"] = {"topology_gain": rule1, "robustness_kept": rule2, "viable": rule1 and rule2}
        print(json.dumps(out["decision"], indent=1))
        (RUNS / f"decision{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
