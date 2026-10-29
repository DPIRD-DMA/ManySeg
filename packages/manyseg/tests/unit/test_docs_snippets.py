"""Code shown in docs/api_design.md that is not covered elsewhere runs as written."""

import torch


def test_lower_level_composition():
    from manyseg.backbones import create_backbone
    from manyseg.decoders import UNetDecoder
    from manyseg.heads import PixelHead
    from manyseg.models import DenseModel

    bb = create_backbone("resnet34", pretrained=False, in_channels=4)
    dec = UNetDecoder(bb.feature_specs)
    model = DenseModel(bb, dec, {"mask": PixelHead(dec.out_spec, 3)}, single_output="mask")
    assert model.eval()(torch.randn(1, 4, 97, 131)).shape == (1, 3, 97, 131)


# ---------------------------------------------------------------- vector outputs guide

def _docs_block(marker: str) -> str:
    """The fenced Python block of docs/api_design.md that contains ``marker``."""
    import re
    from pathlib import Path

    import pytest

    doc = Path(__file__).resolve().parents[4] / "docs" / "api_design.md"
    if not doc.is_file():
        pytest.skip("not a repository checkout (no docs/)")
    blocks = [b for b in re.findall(r"```python\n(.*?)```", doc.read_text(), re.S) if marker in b]
    assert len(blocks) == 1, f"{len(blocks)} blocks contain {marker!r}"
    return blocks[0]


def test_vector_scene_snippet(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pytest

    shapely = pytest.importorskip("shapely")
    pytest.importorskip("rasterio")
    pytest.importorskip("skimage")
    from manyseg.vectors import footprint_targets

    code = _docs_block("records = decode_scene(pred, dec")
    H, W, x0, y0 = 300, 280, 391_000.0, 6_466_000.0
    transform = (0.5, 0.0, x0, 0.0, -0.5, y0)
    boxes = [(20, 30, 60, 70), (120, 100, 170, 150), (200, 220, 260, 290)]  # pixel boxes
    buildings = [shapely.box(x0 + a / 2, y0 - d / 2, x0 + c / 2, y0 - b / 2)
                 for a, b, c, d in boxes]  # metres
    t = footprint_targets([shapely.box(*b) for b in boxes], (H, W))
    pred = {k: t[k] for k in ("extent", "boundary", "distance", "corner", "direction")}
    monkeypatch.chdir(tmp_path)
    ns = {"buildings": buildings, "transform": transform, "H": H, "W": W, "crs": "EPSG:28350",
          "pred": pred}
    exec(compile(code, "api_design.md", "exec"), ns)
    assert len(ns["polys"]) == 3 and ns["targets"]["corner"].shape == (H, W)
    assert len(ns["records"]) == 3
    fc = json.loads((tmp_path / "buildings.geojson").read_text())
    got = sorted(shapely.geometry.shape(f["geometry"]).bounds for f in fc["features"])
    assert np.allclose(got, sorted(b.bounds for b in buildings), atol=0.5)  # metres


def test_vector_manypred_snippet():
    """The ManyPred snippet compiles; its keywords exist where a ManyPred with block
    post-processing is installed (it is not released yet)."""
    import ast
    import inspect

    code = _docs_block("BlockPostprocess(")
    tree = ast.parse(code)
    calls = {n.func.id: {k.arg for k in n.keywords} for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"merge", "block_postprocess", "patch_size", "overlap"} <= calls["predict_files"]
    try:
        import manypred
        from manypred import BlockPostprocess, predict_files
    except ImportError:
        return
    del manypred
    for fn in (BlockPostprocess, predict_files):
        params = inspect.signature(fn).parameters
        assert calls[fn.__name__] <= set(params), (fn.__name__, calls[fn.__name__] - set(params))
