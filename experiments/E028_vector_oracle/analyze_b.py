"""E028b analysis: mean ± SD over seeds of tile means; the registered B2F rule."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

runs = Path(sys.argv[1])
rows = [json.loads(x) for x in (runs / "test.jsonl").open()]
AREAS, ARMS, LV = ("murdoch", "northam"), ("B0", "B2", "B2F", "B2M"), (0, 1, 2, 3)
acc = defaultdict(lambda: defaultdict(list))
for r in rows:
    for m in ("score", "ciou", "det_f1", "polis", "n_ratio"):
        v = r[m]
        if v == v:  # not NaN
            acc[(r["area"], r["arm"], r["level"], m)][r["seed"]].append(v)
S = {k: np.array([np.mean(v) for _, v in sorted(d.items())]) for k, d in acc.items()}
lines = []
for m, name in (("score", "C-IoU × detection F1"), ("ciou", "C-IoU"), ("det_f1", "detection F1"), ("polis", "PoLiS (px)"), ("n_ratio", "N-ratio")):
    lines += [f"\n**{name}**\n", "| area | arm | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|---|"]
    for a in AREAS:
        for arm in ARMS:
            lines.append(f"| {a} | {arm} | " + " | ".join(f"{S[(a, arm, l, m)].mean():.3f} ± {S[(a, arm, l, m)].std(ddof=1):.3f}" for l in LV) + " |")
def beats(l):
    for a in AREAS:
        x, y = S[(a, "B2F", l, "score")], S[(a, "B2", l, "score")]
        d, sd = x.mean() - y.mean(), np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        if not (d > 2 * sd and d >= 0.02):
            return False
    return True
def viable(l):
    return all(S[(a, "B2F", l, "ciou")].mean() >= S[(a, "B2", l, "ciou")].mean() - 0.02 and
               S[(a, "B2F", l, "det_f1")].mean() >= S[(a, "B0", l, "det_f1")].mean() - 0.02 for a in AREAS)
b = [f"L{l}" for l in (1, 2) if beats(l)]
v = {f"L{l}": viable(l) for l in (0, 1, 2)}
lines += ["\n### Decision (E028b rule)\n", f"* B2F beats B2 at: {b or 'neither L1 nor L2'}; viable per level: {v}.",
          "* → " + ("**B2F becomes the default graph footprint decoder.**" if b and all(v.values()) else "**B2F is not adopted.**")]
text = "\n".join(lines)
(runs / "results_b.md").write_text(text)
print(text)
