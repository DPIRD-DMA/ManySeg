"""Named model presets (manyseg.presets): measured Pareto-front configs from E018."""

import pytest
import torch
from manyseg import create_model
from manyseg.presets import PRESETS, preset


def test_preset_names_are_the_e018_front():
    assert set(PRESETS) == {"fast", "compact", "balanced", "accurate"}


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_builds_and_runs(name):
    model = create_model(**preset(name), pretrained=False, in_channels=4, num_classes=3).eval()
    with torch.no_grad():
        y = model(torch.randn(1, 4, 64, 96))
    assert y.shape == (1, 3, 64, 96)


def test_preset_returns_a_fresh_copy():
    a = preset("balanced")
    a["decoder_kwargs"]["channels"][0] = 1
    a["backbone"] = "x"
    assert preset("balanced")["decoder_kwargs"]["channels"][0] == 128
    assert preset("balanced")["backbone"] == "pvt_v2_b0.in1k"


def test_preset_is_overridable_by_the_caller():
    kw = preset("accurate") | {"decoder_kwargs": {"channels": [64, 32, 16, 8, 8]}}
    model = create_model(**kw, pretrained=False, num_classes=2)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_unknown_preset_lists_the_options():
    with pytest.raises(ValueError, match="balanced"):
        preset("nope")


def test_presets_carry_their_measurements():
    for name, p in PRESETS.items():
        assert {"latency_ms", "val_miou"} <= set(p.measured), name
