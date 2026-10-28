"""E045 analysis: pooled per (test set, level, seed), mean over seeds, the README rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run45 import TESTS, bf, det_f1

RUNS = Path(__file__).resolve().parent / "runs"


def main():
    rows = [json.loads(x) for x in (RUNS / "test.jsonl").open()]
    S: dict = {}
    for r in rows:
        S.setdefault((tuple(r["key"]), r["arm"], r["level"], r["seed"]), []).append(r)
    stat = {}
    for (key, arm, lv, s), rs in S.items():
        tp = sum(r["tp"] for r in rs)
        pol = [r["polis"] for r in rs if r["tp"] and r["polis"] == r["polis"]]
        stat[(key, arm, lv, s)] = dict(bf=bf(rs), det_f1=det_f1(rs), polis=float(np.mean(pol)) if pol else float("nan"),
                                       irr_recall=sum(r["irr_tp"] for r in rs) / max(sum(r["irr_n"] for r in rs), 1),
                                       ciou=sum(r["ciou"] * r["tp"] for r in rs) / max(tp, 1))
    seeds = sorted({k[3] for k in stat})

    def m(key, arm, lv, met):
        v = np.array([stat[(key, arm, lv, s)][met] for s in seeds])
        return v.mean(), v.std(ddof=1)

    lines = ["| test set | metric | arm | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|---|---|"]
    for key in TESTS:
        for met in ("bf", "det_f1", "ciou", "polis", "irr_recall"):
            for arm in ("B0", "B3"):
                cells = [f"{m(key, arm, lv, met)[0]:.3f} ± {m(key, arm, lv, met)[1]:.3f}" for lv in range(4)]
                lines.append(f"| {key[0]} {key[1]} | {met} | {arm} | " + " | ".join(cells) + " |")
    transfers = all(m(k, "B3", lv, "bf")[0] >= m(k, "B0", lv, "bf")[0] - 0.01
                    and m(k, "B3", lv, "det_f1")[0] >= m(k, "B0", lv, "det_f1")[0] - 0.03
                    for k in TESTS for lv in (1, 2))
    preferred = transfers and all(m(k, "B3", lv, "bf")[0] >= m(k, "B0", lv, "bf")[0] + 0.01 for k in TESTS for lv in (1, 2))
    lines += ["", f"B3 transfers to fields: **{transfers}**; B3 preferred for fields: **{preferred}**"]
    text = "\n".join(lines)
    (RUNS / "results45.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
