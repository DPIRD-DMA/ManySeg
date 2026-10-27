"""E031 analysis: per-area tables (mean ± SD over 5 noise seeds of tile-pooled values) and the
pre-registered decision.

    uv run python analyze.py [runs_dir]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AREAS = ("leidscherijn", "staphorst")
ARMS = ("A0", "A1", "LoD1")
LEVELS = (0, 1, 2, 3)


def pooled(rows):
    """Per (area, arm, level, seed): face F1 from summed counts, mean of per-tile metrics."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["area"], r["arm"], r["level"], r["seed"])].append(r)
    out = {}
    for k, rs in acc.items():
        tp, ng, npd = (sum(r[x] for r in rs) for x in ("tp", "n_gt", "n_pred"))
        ctp, cg, cp = (sum(r[x] for r in rs) for x in ("corner_tp", "corner_gt", "corner_pred"))
        nn = sum(r["n_normal"] for r in rs)
        out[k] = {"face_f1": 2 * tp / max(1, ng + npd), "corner_f1": 2 * ctp / max(1, cg + cp),
                  "h_rmse": float(np.mean([r["h_rmse"] for r in rs])),
                  "h_mae": float(np.mean([r["h_mae"] for r in rs])),
                  "normal_err_deg": float(sum(r["normal_err_deg"] * r["n_normal"] for r in rs if r["n_normal"]) / max(1, nn)),
                  "ciou": float(sum(r["ciou"] * r["tp"] for r in rs) / max(1, tp))}
    return out


def series(p, area, arm, lv, m):
    return np.array([v[m] for (a, ar, l, s), v in sorted(p.items()) if (a, ar, l) == (area, arm, lv)])


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "runs"
    rows = [json.loads(x) for x in (runs / "test.jsonl").open()]
    p = pooled(rows)
    lines = []
    for m, fmt in (("face_f1", ".3f"), ("h_rmse", ".2f"), ("h_mae", ".2f"), ("corner_f1", ".3f"),
                   ("normal_err_deg", ".1f"), ("ciou", ".3f")):
        lines += [f"\n**{m}**\n", "| area | arm | " + " | ".join(f"L{l}" for l in LEVELS) + " |",
                  "|---|---|" + "---|" * len(LEVELS)]
        for area in AREAS:
            for arm in ARMS:
                cells = []
                for lv in LEVELS:
                    x = series(p, area, arm, lv, m)
                    cells.append(f"{x.mean():{fmt}} ± {x.std(ddof=1):{fmt}}" if len(x) > 1 else "")
                lines.append(f"| {area} | {arm} | " + " | ".join(cells) + " |")

    def beats(lv):
        for area in AREAS:
            a, b = series(p, area, "A1", lv, "face_f1"), series(p, area, "A0", lv, "face_f1")
            d, sd = a.mean() - b.mean(), np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            dr = series(p, area, "A1", lv, "h_rmse").mean() - series(p, area, "A0", lv, "h_rmse").mean()
            if not (d > 2 * sd and d >= 0.02 and dr <= 0.05):
                return False
        return True

    def viable(lv):
        return all(series(p, area, "A1", lv, "face_f1").mean() >= series(p, area, "A0", lv, "face_f1").mean() - 0.02
                   and series(p, area, "A1", lv, "h_rmse").mean() <= series(p, area, "A0", lv, "h_rmse").mean() + 0.05
                   for area in AREAS)

    b = [f"L{l}" for l in (1, 2) if beats(l)]
    v = {f"L{l}": viable(l) for l in (0, 1, 2)}
    ok = all(v.values())
    lines += ["\n### Decision (pre-registered rule, applied by `analyze.py`)\n",
              f"* A1 beats A0 at: {b or 'neither L1 nor L2'}; viable per level: {v}.",
              "* → " + ("**a trained LoD2 experiment with roof-graph heads is worth running.**" if b and ok else
                        "**viable but never beats A0: LoD2 uses plane segmentation; ManySeg's part is the footprint.**"
                        if ok else "**not viable: plane segmentation for roofs; no graph tensor for roofs.**")]
    text = "\n".join(lines)
    (runs / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
