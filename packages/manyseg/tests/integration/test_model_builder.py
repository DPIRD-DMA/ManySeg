"""The model builder page (docs/builder) promises that every configuration it offers either shows
an error or produces code that runs with this ManySeg and returns the shapes it shows. This runs
the page's own logic (model_builder_cases.js, under node) over ~90 configurations and executes
the generated code: valid ones must run with the predicted output shapes, refused ones must also
be rejected by ManySeg. Needs node and a repo checkout (docs/builder); skipped otherwise."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
BUILDER = ROOT / "docs" / "builder"
NODE = shutil.which("node")


def _cases() -> tuple[list[dict], str]:
    """(cases, error): a broken core.js or node failure becomes one failing test, not a
    collection error that hides the rest of the module."""
    if NODE is None or not (BUILDER / "core.js").exists():
        return [], ""
    try:
        out = subprocess.run([NODE, str(Path(__file__).with_name("model_builder_cases.js")),
                              str(BUILDER)], capture_output=True, text=True, timeout=120)
        if out.returncode:
            return [], f"node exited {out.returncode}:\n{out.stderr[-4000:]}"
        return json.loads(out.stdout), ""
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        return [], f"{type(e).__name__}: {e}"


CASES, CASES_ERROR = _cases()
pytestmark = pytest.mark.slow


def _run(code: str) -> dict:
    ns: dict = {}
    with torch.no_grad():
        exec(compile(code, "<model builder>", "exec"), ns)  # noqa: S102 - the page's own output
    return ns


def test_cases_generated():
    if not CASES and not CASES_ERROR:
        pytest.skip("needs node and docs/builder (a repo checkout)")
    assert not CASES_ERROR, CASES_ERROR
    assert len(CASES) > 50


@pytest.mark.skipif(not CASES, reason="needs node and docs/builder (a repo checkout)")
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_generated_code(case):
    # describe() runs too: the page tells users to call it
    code = case["code"].replace("print(model.describe(x))", "model.describe(x)")
    if case["expectError"]:
        assert case["errors"], "the page should refuse this configuration"
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            _run(code)
        return
    assert not case["errors"], case["errors"]
    ns = _run(code)
    out, model = ns["out"], ns["model"]
    got = ({k: list(v.shape) for k, v in out.items()} if isinstance(out, dict)
           else {case["shapes"][0]["name"]: list(out.shape)})
    assert got == {s["name"]: s["shape"] for s in case["shapes"]}
    n = sum(p.numel() for p in model.parameters())
    assert abs(case["estimate"] / n - 1) < 0.05, (case["estimate"], n)


@pytest.mark.skipif(not (ROOT / "scripts" / "build_model_builder.py").exists(),
                    reason="needs a repo checkout")
def test_data_is_up_to_date():
    """docs/builder/data.js matches what the installed ManySeg generates."""
    spec = importlib.util.spec_from_file_location("build_model_builder",
                                                  ROOT / "scripts" / "build_model_builder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OUT.read_text() == mod.render(mod.build()), \
        "docs/builder/data.js is stale: run scripts/build_model_builder.py"
