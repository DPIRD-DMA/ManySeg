"""E037 A1 summary + pre-registered L0 kill rules.  uv run python analyze_a1.py [runs/a1.jsonl]"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "runs" / "a1.jsonl")
rows = [json.loads(l) for l in open(path)]
by = defaultdict(list)
for r in rows:
    src = "cityscale" if r["area"].startswith("cityscale") else r["area"] if r["key"] != "roads" else "osm"
    if r["key"] == "buildings":
        src = "whu" if r["area"] == "whu" else "osm"
    by[(r["key"], r["arm"])].append(dict(r, src=src))

out = {}


def frac(a):
    a = np.asarray(a, bool)
    return float(a.mean()) if len(a) else float("nan")


print("== roads (mean over tiles; strata pooled over junctions)")
for (k, arm), rs in sorted(by.items()):
    if k != "roads":
        continue
    ang = np.concatenate([r["j_ang"] for r in rs]) if rs else np.array([])
    hit = np.concatenate([r["j_hit_deg"] for r in rs]).astype(bool)
    cr_ok, cr_n = sum(r["cross_ok"] for r in rs), sum(r["cross_n"] for r in rs)
    m = {"apls": np.mean([r["apls"] for r in rs]), "junction_f1": np.mean([r["junction_f1"] for r in rs]),
         "apls_osm": np.mean([r["apls"] for r in rs if r["src"] == "osm"]),
         "apls_cityscale": np.mean([r["apls"] for r in rs if r["src"] == "cityscale"]),
         "junction_lost_or_wrong": 1 - frac(hit), "crossings_kept": f"{cr_ok}/{cr_n}",
         "crossings_kept_frac": cr_ok / max(cr_n, 1)}
    for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 181)):
        s = (ang >= lo) & (ang < hi)
        m[f"jfail_{lo}-{hi}"] = (round(1 - frac(hit[s]), 4), int(s.sum()))
    if "collisions" in rs[0]:
        m["collisions"] = int(sum(r["collisions"] for r in rs))
    out[("roads", arm)] = m
    print(arm, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in m.items()})

print("\n== buildings")
for (k, arm), rs in sorted(by.items()):
    if k != "buildings":
        continue
    per = np.concatenate([r["b_per"] for r in rs])
    sep = np.concatenate([r["b_sep"] for r in rs])
    hit = np.concatenate([r["b_hit"] for r in rs]).astype(bool)
    m = {"ciou_x_f1": np.mean([r["ciou"] * r["det_f1"] for r in rs]),
         "ciou_x_f1_osm": np.mean([r["ciou"] * r["det_f1"] for r in rs if r["src"] == "osm"]),
         "ciou_x_f1_whu": np.mean([r["ciou"] * r["det_f1"] for r in rs if r["src"] == "whu"]),
         "det_f1": np.mean([r["det_f1"] for r in rs]), "ciou": np.mean([r["ciou"] for r in rs]),
         "ring_not_closed": 1 - frac(hit)}
    q = np.quantile(per, [1 / 3, 2 / 3])
    for name, s in (("per_short", per < q[0]), ("per_mid", (per >= q[0]) & (per < q[1])), ("per_long", per >= q[1]),
                    ("sep<2px", sep < 2), ("sep>=2px", sep >= 2)):
        m[f"open_{name}"] = (round(1 - frac(hit[s]), 4), int(s.sum()))
    if "collisions" in rs[0]:
        m["collisions"] = int(sum(r["collisions"] for r in rs))
    out[("buildings", arm)] = m
    print(arm, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in m.items()})

print("\n== roofs")
for (k, arm), rs in sorted(by.items()):
    if k != "roofs":
        continue
    tp, ng = sum(r["tp"] for r in rs), sum(r["n_gt"] for r in rs)
    m = {"face_f1": np.mean([r["face_f1"] for r in rs]), "h_rmse": np.mean([r["h_rmse"] for r in rs]),
         "ciou": np.mean([r["ciou"] for r in rs]), "face_not_closed": 1 - tp / max(ng, 1)}
    if "collisions" in rs[0]:
        m["collisions"] = int(sum(r["collisions"] for r in rs))
    out[("roofs", arm)] = m
    print(arm, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in m.items()})

print("\n== L0 kill rules (README A1)")
ref = {"roads": ("R2", "apls"), "buildings": ("B2", "ciou_x_f1"), "roofs": ("A1", "face_f1")}
for key, arms in (("roads", ("P8", "P8r", "R3", "P8g", "P8rg", "R3g")), ("buildings", ("P8", "P8r", "P8g", "P8rg")), ("roofs", ("P8", "P8r", "P8g", "P8rg"))):
    rarm, metric = ref[key]
    if (key, rarm) not in out:
        continue
    for arm in arms:
        if (key, arm) not in out:
            continue
        m = out[(key, arm)]
        why = []
        if key == "roads" and m["junction_f1"] < 0.995:  # registered form (Codex E037 #3)
            why.append(f"rule 1: junction F1 {m['junction_f1']:.4f} < 0.995")
        if key == "buildings" and m["ring_not_closed"] > 0.01:
            why.append(f"rule 1: rings not closed {m['ring_not_closed']:.4f} > 0.01")
        if key == "roofs" and m["face_not_closed"] > 0.01:
            why.append(f"rule 1: faces not closed {m['face_not_closed']:.4f} > 0.01")
        gap = out[(key, rarm)][metric] - m[metric]
        if gap > 0.02:
            why.append(f"rule 2: {metric} {m[metric]:.4f} is {gap:.4f} below {rarm}")
        strata = []
        if key == "roads":
            n_all = sum(v[1] for kk, v in m.items() if kk.startswith("jfail_"))
            f, n = m["jfail_0-15"]
            strata.append(("junction angle < 15", f, n / max(n_all, 1)))
            strata.append(("crossings", 1 - m["crossings_kept_frac"], 1.0))  # crossings are a product feature
        if key == "buildings":
            f, n = m["open_sep<2px"]
            n_all = n + m["open_sep>=2px"][1]
            strata.append(("separation < 2 px", f, n / max(n_all, 1)))
        for name, f, share in strata:
            if f > 0.05 and share > 0.01:
                why.append(f"rule 3: {name} failure {f:.3f} (share {share:.3f})")
        print(f"{key:9s} {arm:4s}", "KILL: " + "; ".join(why) if why else "survives L0")
