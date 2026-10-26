"""E030 analysis: mean ± SD over seeds, and the pre-registered decision.

    uv run python experiments/E030_whu_polygons/analyze.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ("ciou", "det_f1", "ciou_x_f1", "miou", "polis_m", "n_ratio", "decode_s_per_tile")


def main():
    runs = {}
    for f in sorted((HERE / "runs").glob("B?_s[0-9]_eval.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["arm"], []).append(r)
    lines = ["| arm | seeds | " + " | ".join(METRICS) + " | extent IoU |",
             "|---|---|" + "---|" * (len(METRICS) + 1)]
    stats = {}
    for arm, rs in sorted(runs.items()):
        vals = {m: np.array([r["test"][m] for r in rs]) for m in METRICS}
        vals["extent_iou"] = np.array([r["test_extent_iou"] for r in rs])
        stats[arm] = vals
        cells = [f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"
                 for v in vals.values()]
        lines.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + " |")
    lines.append("")
    for arm, rs in sorted(runs.items()):
        for r in rs:
            lines.append(f"* {r['run']}: decoder {r['params']}; val C-IoU×F1 {r['val']['ciou_x_f1']:.3f}")
    if {"B0", "B2"} <= set(stats) and min(len(runs["B0"]), len(runs["B2"])) >= 2:
        a, b = stats["B2"], stats["B0"]
        diff = a["ciou"].mean() - b["ciou"].mean()
        pooled = np.sqrt((a["ciou"].var(ddof=1) + b["ciou"].var(ddof=1)) / 2)
        f1_gap = a["det_f1"].mean() - b["det_f1"].mean()
        win = diff > 2 * pooled and diff >= 0.02 and f1_gap >= -0.02
        lines += ["", f"C-IoU B2 − B0 = {diff:+.4f} (2 pooled SD = {2 * pooled:.4f}); "
                      f"detection F1 B2 − B0 = {f1_gap:+.4f}",
                  f"**Decision: {'B2 wins' if win else 'B2 does not win'}** (pre-registered rule)."]
    text = "\n".join(lines)
    (HERE / "runs" / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
