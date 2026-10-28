"""E045 driver (README). Phases: tune0 (B0 grid), tune3 (B3 grid on tuned B0), test.

    E028_DATA unused; tiles come from tiles.pkl (``python run45.py tiles`` builds it from FTW).
    python run45.py {tiles|tune0|tune3|test} [-n procs] [--limit-tiles N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TUNE, TESTS = ("netherlands", "val"), (("netherlands", "test"), ("denmark", "test"))
TUNE_SEEDS, TEST_SEEDS = (0, 1), (0, 1, 2, 3, 4)
GRID0 = [dict(zip(("thr", "thr_b", "thr_s", "simplify"), v))
         for v in itertools.product((0.4, 0.6), (0.3, 0.5), (0.15, 0.3), (0.5, 1, 2))]
GRID3 = [dict(zip(("thr_c", "big_dev", "agree", "merge_deg"), v), snap_r=4.0)
         for v in itertools.product((0.3, 0.5), (4.0, 8.0), (0.4, 0.6), (8.0, 15.0))]
_T: dict = {}


def tiles(key):
    if not _T:
        _T.update(pickle.loads((RUNS / "tiles.pkl").read_bytes()))
    return _T[key]


def work(job):
    import e028 as E
    import e035 as B
    import e045
    from manyseg.vectors import polygons_from_labels
    from shapely.geometry import Polygon

    t = tiles(tuple(job["key"]))[job["tile"]]
    a = B.corrupt_b3(B.encode_b3(t), t, E.LEVELS[job["level"]], job["seed"])
    rows = []
    for arm, p, pi in job["arms"]:
        t0 = time.perf_counter()
        if arm == "B0":  # exact pixel-edge outline + Douglas-Peucker, as the library (Codex E045 #2)
            lab = E._labels_b0(a, p)
            pred = [Polygon(q.exterior) for q in polygons_from_labels(lab, simplify=p["simplify"], min_area=1)]
        else:
            pred = B.decode_b3r(a, dict(p, b0=p["b0"], kernel=3, simplify=p["b0"]["simplify"]))
        dt = time.perf_counter() - t0
        m = E.polygon_metrics(t, pred)
        bp_tp, bp_n, bg_tp, bg_n = e045.boundary_counts(t.polygons, pred)
        irr = [g for g in t.polygons if e045.irregular(g)]
        mi = E.polygon_metrics(E.Tile(t.area, t.task, t.index, 0, 0, t.gsd, polygons=irr), pred) if irr else None
        rows.append(dict(key=job["key"], tile=t.index, level=job["level"], seed=job["seed"], arm=arm, pidx=pi,
                         decode_s=dt, bp_tp=bp_tp, bp_n=bp_n, bg_tp=bg_tp, bg_n=bg_n,
                         tp=m["tp"], n_gt=m["n_gt"], n_pred=m["n_pred"], ciou=m["ciou"], polis=m["polis"] / t.gsd,
                         n_ratio=m["n_ratio"], irr_n=len(irr), irr_tp=mi["tp"] if mi else 0))
    return rows


def run(jobs, out, procs):
    t0 = time.time()
    with out.open("w") as f, Pool(procs) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 50 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"{k}/{len(jobs)} jobs, {el / 60:.1f} min, eta {el / k * (len(jobs) - k) / 60:.1f} min", flush=True)


def bf(rows):
    """Pooled boundary F over rows."""
    tp_p, n_p = sum(r["bp_tp"] for r in rows), sum(r["bp_n"] for r in rows)
    tp_g, n_g = sum(r["bg_tp"] for r in rows), sum(r["bg_n"] for r in rows)
    p, r = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
    return 2 * p * r / max(p + r, 1e-9)


def det_f1(rows):
    tp = sum(r["tp"] for r in rows)
    return 2 * tp / max(sum(r["n_gt"] + r["n_pred"] for r in rows), 1)


def select(path, grid):
    acc: dict = {}
    for r in map(json.loads, path.open()):
        acc.setdefault((r["level"], r["pidx"]), []).append(r)
    sel = {}
    for (lv, pi), rows in acc.items():
        s = bf(rows)
        if f"L{lv}" not in sel or s > sel[f"L{lv}"]["bf"]:
            sel[f"L{lv}"] = {"bf": s, "det_f1": det_f1(rows), "params": grid[pi]}
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tiles", "tune0", "tune3", "test"])
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--limit-tiles", type=int, default=None)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    if args.phase == "tiles":
        import e045

        d = {k: e045.field_tiles(*k) for k in (TUNE, *TESTS)}
        (RUNS / "tiles.pkl").write_bytes(pickle.dumps(d))
        print({k: len(v) for k, v in d.items()})
        return
    lim = args.limit_tiles
    jobs = []
    if args.phase == "tune0":
        for ti in range(min(len(tiles(TUNE)), lim or 10**9)):
            for lv in range(4):
                for s in TUNE_SEEDS:
                    jobs.append(dict(key=TUNE, tile=ti, level=lv, seed=s, arms=[("B0", g, i) for i, g in enumerate(GRID0)]))
        run(jobs, RUNS / "tune0.jsonl", args.n)
        (RUNS / "best0.json").write_text(json.dumps(select(RUNS / "tune0.jsonl", GRID0), indent=1))
    elif args.phase == "tune3":
        b0 = json.loads((RUNS / "best0.json").read_text())
        for ti in range(min(len(tiles(TUNE)), lim or 10**9)):
            for lv in range(4):
                for s in TUNE_SEEDS:
                    arms = [("B3", dict(g, b0=b0[f"L{lv}"]["params"]), i) for i, g in enumerate(GRID3)]
                    jobs.append(dict(key=TUNE, tile=ti, level=lv, seed=s, arms=arms))
        run(jobs, RUNS / "tune3.jsonl", args.n)
        (RUNS / "best3.json").write_text(json.dumps(select(RUNS / "tune3.jsonl", GRID3), indent=1))
    else:
        b0 = json.loads((RUNS / "best0.json").read_text())
        b3 = json.loads((RUNS / "best3.json").read_text())
        for key in TESTS:
            for ti in range(min(len(tiles(key)), lim or 10**9)):
                for lv in range(4):
                    p0 = b0[f"L{lv}"]["params"]
                    arms = [("B0", p0, 0), ("B3", dict(b3[f"L{lv}"]["params"], b0=p0), 0)]
                    for s in TEST_SEEDS:
                        jobs.append(dict(key=key, tile=ti, level=lv, seed=s, arms=arms))
        run(jobs, RUNS / "test.jsonl", args.n)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE.parent / "E028_vector_oracle"))
    sys.path.insert(0, str(HERE))
    main()
