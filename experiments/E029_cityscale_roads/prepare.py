"""Convert City-Scale (Sat2Graph's 20 cities, from the RNGDet++ Tsinghua Cloud share) into
pickle-free files: one ``region_<id>.npz`` per region with

* ``xy`` (N, 2) float32 node coordinates, ``(x, y)`` pixel centres (``col + 0.5, row + 0.5``);
* ``edges`` (E, 2) int32;
* ``overpass``, ``complicated``, ``parallel`` (K, 2) float32 annotated points, same convention;

plus ``split.json`` (144 train / 9 valid / 27 test). Images stay as ``region_<id>_sat.png``.

The source graphs are Python pickles of plain dicts ``{(row, col): [(row, col), ...]}``. They
are read with an unpickler that refuses every global, so no code from the download can run.
Verified visually on region 9 (keys are (row, col)).

    python -I prepare.py "<raw>/cityscale/data" "<out>"
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        adj = _NoGlobals(f, encoding="latin1").load()
    if not isinstance(adj, dict):
        raise TypeError(f"{path}: expected a dict, got {type(adj).__name__}")
    keys = set(adj)
    for nbs in adj.values():
        keys.update(map(tuple, nbs))
    index = {k: i for i, k in enumerate(sorted(keys))}
    xy = np.array([(c + 0.5, r + 0.5) for r, c in sorted(keys)], np.float32).reshape(-1, 2)
    edges = {tuple(sorted((index[tuple(a)], index[tuple(b)]))) for a, nbs in adj.items() for b in nbs
             if tuple(a) != tuple(b)}
    return xy, np.array(sorted(edges), np.int32).reshape(-1, 2)


def points(rc) -> np.ndarray:
    a = np.asarray(rc, np.float32).reshape(-1, 2)
    return np.stack([a[:, 1] + 0.5, a[:, 0] + 0.5], 1)


def main(src: str, out: str) -> None:
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((src / "data_split.json").read_text())
    (out / "split.json").write_text(json.dumps(split))
    n = 0
    for rid in sorted({i for v in split.values() for i in v}):
        xy, edges = load_graph(src / "20cities" / f"region_{rid}_graph_gt.pickle")
        sp = json.loads((src / "20cities" / f"region_{rid}_refine_gt_graph_samplepoints.json").read_text())
        np.savez_compressed(out / f"region_{rid}.npz", xy=xy, edges=edges,
                            overpass=points(sp.get("overpass", [])),
                            complicated=points(sp.get("complicated_intersections", [])),
                            parallel=points(sp.get("parallel_road", [])))
        n += 1
    print(f"{n} regions -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
