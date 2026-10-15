"""Every script in the repository's ``examples/`` runs to completion on CPU.

Each example runs in its own subprocess (as a user would run it), on CPU, offline, with its
temporary files under ``tmp_path``. Slow tier: about 5-10 s per example.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))


def test_examples_found():
    if not EXAMPLES.is_dir():
        pytest.skip("not a repository checkout (no examples/)")
    assert len(SCRIPTS) >= 7, f"no examples found in {EXAMPLES}"


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_example_does_not_import_bench(script):
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(n.split(".")[0] == "manyseg_bench" for n in names), script.name


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_example_runs(script, tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(tmp_path),
           "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": os.environ.get("MANYSEG_TEST_THREADS", "4")}
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{script.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
