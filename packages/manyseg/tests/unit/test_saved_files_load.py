"""Files saved by earlier ManySeg versions keep loading, to the same outputs.

``tests/data/golden_format<N>.safetensors`` was written by ``tests/data/make_golden.py`` with
the ManySeg of its save format, and is committed. A change that renames a config key, a module
or a parameter fails here: add a migration in ``manyseg.io`` (and bump ``FORMAT``), never
regenerate the file.
"""

from pathlib import Path

import manyseg
import pytest
import torch
from safetensors.torch import load_file

DATA = Path(__file__).resolve().parents[1] / "data"
GOLDEN = sorted(DATA.glob("golden_format*[0-9].safetensors"))
QUANTIZED = sorted(DATA.glob("golden_quantized_format*[0-9].safetensors"))


def test_golden_files_exist():
    assert GOLDEN, f"no golden files in {DATA}"
    formats = {int(p.stem.removeprefix("golden_format")) for p in GOLDEN}
    assert manyseg.io.FORMAT in formats, "add a golden file for the current save format"


@pytest.mark.parametrize("path", GOLDEN, ids=[p.stem for p in GOLDEN])
def test_golden_file_loads_to_the_same_outputs(path):
    model = manyseg.load(path)
    io = load_file(str(path.with_name(path.stem + "_io.safetensors")))
    with torch.no_grad():
        out = model({"img": io["in_img"], "doy": io["in_doy"]})
    want = {k.removeprefix("out_"): v for k, v in io.items() if k.startswith("out_")}
    assert set(out) == set(want)
    for k in want:  # tolerance: other torch / CPU kernels, not other weights
        torch.testing.assert_close(out[k], want[k], atol=1e-4, rtol=1e-4, msg=k)
    meta = model.saved_metadata
    assert meta["activations"] == {"seg": "softmax", "density": "none", "label": "softmax"}
    assert meta["normalize"] == [{"type": "scale", "value": 10000}]
    assert meta["classes"]["label"] == ["no", "yes"]
    spec = model.tiling_spec()
    assert spec["outputs"]["seg"]["activation"] == "softmax"


def test_file_from_a_newer_manyseg_fails_with_a_version_hint(tmp_path):
    """A newer ManySeg may add config keys within the same save format; an older one must say
    which versions are involved instead of a bare create_model error."""
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = GOLDEN[-1]
    with safe_open(str(src), framework="pt") as f:
        stored = dict(f.metadata())
    meta = json.loads(stored["manyseg"])
    meta["manyseg_version"] = "99.0.0"
    meta["create_model"]["heads"]["seg"]["future_option"] = 1
    bad = tmp_path / "newer.safetensors"
    save_file(load_file(str(src)), str(bad), metadata={**stored, "manyseg": json.dumps(meta)})
    with pytest.raises(ValueError, match=r"saved by ManySeg 99\.0\.0.*upgrade"):
        manyseg.load(bad)


@pytest.mark.parametrize("saved, current, newer", [
    ("99.0.0", "0.1.0", True), ("0.1.0", "0.1.0.dev0", True), ("0.1.0rc1", "0.1.0", False),
    ("0.1.0.dev10", "0.1.0", False), ("0.1.1", "0.1.0+local", True), ("0.1.0", "0.1.0", False),
    ("unknown", "0.1.0", False), (None, "0.1.0", False)])
def test_version_ordering_is_pep440(saved, current, newer):
    assert manyseg.io._is_newer(saved, current) is newer


def test_rebuild_failure_names_versions_for_older_files(tmp_path):
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = GOLDEN[-1]
    with safe_open(str(src), framework="pt") as f:
        stored = dict(f.metadata())
    meta = json.loads(stored["manyseg"])
    meta["manyseg_version"] = "0.0.1"
    meta["create_model"]["heads"]["seg"]["renamed_option"] = 1  # a missing migration
    bad = tmp_path / "older.safetensors"
    save_file(load_file(str(src)), str(bad), metadata={**stored, "manyseg": json.dumps(meta)})
    with pytest.raises(ValueError, match=r"saved by ManySeg 0\.0\.1.*missing migration"):
        manyseg.load(bad)


@pytest.mark.parametrize("path", QUANTIZED, ids=[p.stem for p in QUANTIZED])
def test_golden_quantized_file_loads_to_the_same_outputs(path):
    pytest.importorskip("torchao")
    model = manyseg.load(path)
    io = load_file(str(path.with_name(path.stem + "_io.safetensors")))
    with torch.no_grad():
        out = model({"img": io["in_img"], "doy": io["in_doy"]})
    want = {k.removeprefix("out_"): v for k, v in io.items() if k.startswith("out_")}
    assert set(out) == set(want)
    for k in want:
        torch.testing.assert_close(out[k], want[k], atol=1e-4, rtol=1e-4, msg=k)
    assert model.saved_metadata["quantization"]["mode"] == "int8_dynamic"


def test_golden_quantized_file_exists():
    assert QUANTIZED, "add a golden file for the quantised save format"
