"""E062 seed-0 screen (pre-registered, val only): pick the FiLM arm with the highest val mIoU;
pass if it beats aerial by more than 0.01. Writes runs/screen.json; exit 0 = pass, 3 = stop."""
import json
import sys
from pathlib import Path

runs = Path(__file__).parent / "runs"
arms = ("aerial", "s2enc", "film_med", "film_ts", "film_ctx")
val = {a: json.loads((runs / f"{a}_s0" / "result.json").read_text())["val"]["miou"] for a in arms}
pick = max(("film_med", "film_ts", "film_ctx"), key=val.get)
gain = val[pick] - val["aerial"]
res = {"val": val, "pick": pick, "gain_over_aerial": gain, "pass": gain > 0.01}
(runs / "screen.json").write_text(json.dumps(res, indent=2))
print("SCREEN", json.dumps(res), flush=True)
sys.exit(0 if res["pass"] else 3)
