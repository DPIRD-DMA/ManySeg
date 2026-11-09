"""E061 seed-0 screen (pre-registered, val only): pick ctx20 or ctx40 by val mIoU; pass if it
beats ctx10 by more than 0.01 val mIoU. Writes runs/screen.json; exit 0 = pass, 3 = stop."""
import json
import sys
from pathlib import Path

runs = Path(__file__).parent / "runs"
val = {a: json.loads((runs / f"{a}_s0" / "result.json").read_text())["val"]["miou"]
       for a in ("aerial", "ctx10", "ctx20", "ctx40")}
pick = max(("ctx20", "ctx40"), key=val.get)
gain = val[pick] - val["ctx10"]
res = {"val": val, "pick": pick, "gain_over_ctx10": gain, "pass": gain > 0.01}
(runs / "screen.json").write_text(json.dumps(res, indent=2))
print("SCREEN", json.dumps(res), flush=True)
sys.exit(0 if res["pass"] else 3)
