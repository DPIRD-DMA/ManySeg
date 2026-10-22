"""E027: E026's model_bench.py with round-2 patches: ``--patch2 name[,name]`` (patches2.py)."""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "E026_perf_audit"))
import patches2  # noqa: E402

argv, names = [], []
it = iter(sys.argv[1:])
for a in it:
    if a == "--patch2":
        names = next(it).split(",")
    else:
        argv.append(a)
applied2 = patches2.apply(names)
sys.argv = [sys.argv[0], *argv]
import model_bench  # noqa: E402
import patches  # noqa: E402  (E026's; its apply() result is what the record lists)

_apply = patches.apply
patches.apply = lambda n: _apply(n) + applied2

model_bench.main()
