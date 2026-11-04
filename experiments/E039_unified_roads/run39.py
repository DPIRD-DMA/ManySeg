"""E039 runner: tune R4 on Innaloo per level, then test R0, R2 and R4 on the same corrupted draws.

    E028_DATA=... uv run python run39.py tune|test|decide [--procs 6] [--limit N]
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

E = R.E
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BEST028 = Path(os.environ.get("E028_BEST", "best.json"))
TUNE_AREA, TEST_AREAS = "innaloo", ("murdoch", "northam")
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
GRID = [dict(thr_k=t, zr=z, pair_deg=150.0, thr_line=0.3, mode=m)
        for t, z, m in itertools.product((0.2, 0.4), (8.0, 12.0), ("centre", "surface", "hybrid"))]
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
    rows = []
    a4 = R.corrupt_r4(R.encode_r4(t), t, cfg, job["seed"])
    E.CACHE.clear()
    for k, p in enumerate(job["r4"]):
        t0 = time.perf_counter()
        pred = R.decode_r4(a4, dict(p, r0=b["R0"]["params"], gap=b["R0"]["params"]["gap"]))
        rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm="R4", pidx=k, params=p,
                         s=time.perf_counter() - t0, **E.road_metrics(t, pred)))
    if job.get("baselines"):
        for rep in ("R0", "R2"):
            a = E.corrupt(E.encode(t, rep), t, cfg, job["seed"])
            E.CACHE.clear()
            pred = E.DECODERS[rep](a, b[rep]["params"])
            rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm=rep, pidx=0,
                             params=b[rep]["params"], s=0.0, **E.road_metrics(t, pred)))
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
        jobs = [dict(area=TUNE_AREA, tile=i, level=lv, seed=s, r4=GRID)
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
        jobs = []
        for area in TEST_AREAS:
            n = len(tiles(area))
            jobs += [dict(area=area, tile=i, level=lv, seed=s, r4=[best[f"L{lv}"]], baselines=True)
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
        for arm in ("R0", "R2", "R4"):
            for lv in range(4):
                o = {k: float(np.mean(m[(arm, lv, k)])) for k in ("apls", "junction_f1")}
                o.update({f"{k}_{a}": float(np.mean(m[(arm, lv, k, a)])) for k in ("apls",) for a in TEST_AREAS})
                ok, n = cr[(arm, lv)]
                o["crossings_kept"] = ok / max(n, 1)
                out[f"{arm}_L{lv}"] = o
                print(arm, f"L{lv}", {k: round(v, 3) for k, v in o.items()})
        g = lambda arm, lv, k: out[f"{arm}_L{lv}"][k]  # noqa: E731
        rule1 = all(g("R4", lv, "junction_f1") >= g("R0", lv, "junction_f1") + 0.10 for lv in (0, 1)) and \
            g("R4", 0, "crossings_kept") >= 0.5
        rule2 = all(g("R4", lv, "apls") >= g("R0", lv, "apls") - 0.01 for lv in (1, 2))
        out["decision"] = {"topology_gain": rule1, "robustness_kept": rule2, "viable": rule1 and rule2}
        print(json.dumps(out["decision"], indent=1))
        (RUNS / f"decision{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
