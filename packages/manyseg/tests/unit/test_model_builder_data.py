"""The model builder page's option lists (docs/builder/data.js) match the installed ManySeg, so
the page never offers an option the library lacks or misses one it has. Fast: reads the file
only. The full check (generated code runs) is tests/integration/test_model_builder.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from manyseg.conditioning import FIELD_TYPES
from manyseg.decoders import DECODERS
from manyseg.fusion.ops import FUSION_OPS
from manyseg.presets import PRESETS
from manyseg.registry import reserved_names
from manyseg.temporal import AGGREGATIONS
from manyseg.weights import WEIGHTS

DATA_JS = Path(__file__).resolve().parents[4] / "docs" / "builder" / "data.js"
pytestmark = pytest.mark.skipif(not DATA_JS.exists(), reason="needs a repo checkout")


@pytest.fixture(scope="module")
def data() -> dict:
    text = DATA_JS.read_text()
    start = text.index("const DATA = ") + len("const DATA = ")
    return json.loads(text[start:].rstrip().rstrip(";"))


def test_registries(data):
    assert data["decoders"] == sorted(DECODERS)
    assert data["fusion_ops"] == sorted(FUSION_OPS)
    assert data["field_types"] == sorted(FIELD_TYPES)
    assert data["temporal"] == list(AGGREGATIONS)
    assert data["reserved_names"] == sorted(n for n in reserved_names() if not n.startswith("_"))


def test_weights(data):
    assert set(data["weights"]) == set(WEIGHTS)
    for key, w in data["weights"].items():
        assert (w["model"], w["tag"], w["bands"]) == (WEIGHTS[key].model, WEIGHTS[key].tag,
                                                      len(WEIGHTS[key].bands))


def test_presets(data):
    assert {k: v["kwargs"] for k, v in data["presets"].items()} == \
        {k: p.kwargs for k, p in PRESETS.items()}


def test_backbones_cover_defaults(data):
    names = {b["name"] for b in data["backbones"]}
    assert "resnet34" in names  # create_model's default
    assert {p.kwargs["backbone"] for p in PRESETS.values()} <= names
    assert {w.model for w in WEIGHTS.values()} <= names
