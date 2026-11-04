"""The notebooks in the repository's ``notebooks/``: saved outputs on every commit, a run from
scratch in the heavy tier.

The notebooks are written for readers, so they hold no test switches. The run edits a copy of the
source instead (``SMOKE_EDITS``: a few chips, one epoch, no pretrained weights) and works in a
temporary folder, so the saved outputs stay as they are. Notebooks 1 and 2 download a few chips of
Sen1Floods11 and skip if its bucket can't be reached.
"""

import ast
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[4] / "notebooks"
FILES = sorted(NOTEBOOKS.glob("*.ipynb"))
BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson"
MAX_MB = 3  # saved figures are committed: keep their resolution down
LOCAL_PATH = re.compile(r"/home/|/media/|/Users/|C:\\\\Users")
TEST_SWITCH = re.compile(r"MANYSEG_|SMOKE|os\.environ")
_FLOOD_EDITS = {"N_TRAIN, N_TEST, EPOCHS = 120, 40, 40": "N_TRAIN, N_TEST, EPOCHS = 4, 2, 1",
                "batch_size=8": "batch_size=2", "pretrained=True": "pretrained=False"}
SMOKE_EDITS = {  # source text to replace in each notebook for the run from scratch
    "01_flood_mapping": {**_FLOOD_EDITS, 'backbone="resnet34"': 'backbone="resnet18"'},
    "02_radar_and_multiple_outputs": _FLOOD_EDITS,
}


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(path):
    return [c for c in _cells(path) if c["cell_type"] == "code"]


def test_notebooks_found():
    if not NOTEBOOKS.is_dir():
        pytest.skip("not a repository checkout (no notebooks/)")
    assert len(FILES) >= 3, f"no notebooks found in {NOTEBOOKS}"
    assert set(SMOKE_EDITS) <= {p.stem for p in FILES}


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_notebook_is_saved_with_outputs(path):
    code = _code(path)
    images = sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"])
    assert images >= 3, f"{path.name} has {images} saved figures"
    assert not [o for c in code for o in c["outputs"] if o["output_type"] == "error"], path.name


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_notebook_is_small_and_has_no_local_paths(path):
    assert path.stat().st_size < MAX_MB * 1e6, f"{path.name} is over {MAX_MB} MB"
    text = "\n".join(
        "".join(o.get("text", [])) for c in _code(path) for o in c["outputs"])
    assert not LOCAL_PATH.search(text), f"{path.name} prints a local path"
    assert not LOCAL_PATH.search("\n".join("".join(c["source"]) for c in _cells(path))), path.name


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_notebook_does_not_import_bench(path):
    for cell in _code(path):
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "manyseg_bench" for n in names), path.name


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_notebook_has_no_test_switches(path):
    assert not TEST_SWITCH.search("\n".join("".join(c["source"]) for c in _cells(path))), path.name


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_smoke_edits_match_the_source(path):
    # checked on every commit: a stale edit would otherwise turn the nightly run into a full one
    source = "\n".join("".join(c["source"]) for c in _code(path))
    for old in SMOKE_EDITS.get(path.stem, {}):
        assert source.count(old) == 1, f"{path.name}: {old!r}"


@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_notebook_runs(path, tmp_path, monkeypatch):
    # CI installs these: there, a missing one is a broken install and not a reason to skip
    if os.environ.get("MANYSEG_REQUIRE_TRAINERS"):
        import nbclient
        import nbformat
    else:
        nbclient = pytest.importorskip("nbclient")
        nbformat = pytest.importorskip("nbformat")
        pytest.importorskip("ipykernel")
        pytest.importorskip("matplotlib")
    if "urlopen" in path.read_text(encoding="utf-8"):
        try:
            urllib.request.urlopen(BUCKET, timeout=20).close()
        except urllib.error.URLError as e:
            pytest.skip(f"Sen1Floods11 bucket not reachable: {e}")
    for key, value in {"CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "agg",
                       "PYDEVD_DISABLE_FILE_VALIDATION": "1",
                       "OMP_NUM_THREADS": os.environ.get("MANYSEG_TEST_THREADS", "4")}.items():
        monkeypatch.setenv(key, value)
    nb = nbformat.read(path, as_version=4)
    for old, new in SMOKE_EDITS.get(path.stem, {}).items():
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(old, new)
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
