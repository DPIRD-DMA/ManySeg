"""The construction-time probe that finds which feature levels a decoder reads (FiLM targets)."""

import pytest
import torch
from manyseg import FeatureSpec, create_model
from manyseg.conditioning import build as build_mod
from manyseg.conditioning.inject import used_levels
from manyseg.decoders import DECODERS
from torch import nn

SPECS = (FeatureSpec(8, 4), FeatureSpec(16, 8), FeatureSpec(32, 16))


def cond_model():
    return create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"in_channels": 3}, "meta": {"kind": "vector", "num_features": 3}})


def test_probe_leaves_the_global_rng_untouched():
    dec = DECODERS["fpn"](SPECS)
    state = torch.random.get_rng_state()
    used_levels(dec, SPECS)
    assert torch.equal(state, torch.random.get_rng_state())


def test_create_model_rng_and_weights_do_not_depend_on_probing(monkeypatch):
    torch.manual_seed(0)
    a = cond_model()
    state_a = torch.random.get_rng_state()
    levels = list(a.conditioner.film["decoder"].levels)
    monkeypatch.setattr(build_mod, "used_levels", lambda *args, **kw: levels)
    torch.manual_seed(0)
    b = cond_model()
    assert torch.equal(state_a, torch.random.get_rng_state())
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


def test_conditioned_build_has_the_unconditioned_raster_weights():
    torch.manual_seed(0)
    cond = cond_model()
    torch.manual_seed(0)
    plain = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    sd = cond.state_dict()
    for k, v in plain.state_dict().items():
        assert torch.equal(v, sd[k]), k


def test_probe_preserves_mixed_train_eval_modes():
    dec = DECODERS["unet"](SPECS).train()
    child = next(iter(dec.children()))
    child.eval()
    before = {n: m.training for n, m in dec.named_modules()}
    assert len(set(before.values())) == 2  # really mixed
    used_levels(dec, SPECS)
    assert {n: m.training for n, m in dec.named_modules()} == before


class _Counting(nn.Module):
    """Mutates a buffer on every call (like a running statistic)."""

    def __init__(self, specs):
        super().__init__()
        self.conv = nn.Conv2d(specs[-1].channels, 4, 1)
        self.register_buffer("calls", torch.zeros(()))
        self.out_spec = FeatureSpec(4, specs[-1].stride)

    def forward(self, feats):
        self.calls += 1
        return self.conv(feats[-1])


def test_probe_restores_buffers():
    dec = _Counting(SPECS)
    assert used_levels(dec, SPECS) == [2]
    assert dec.calls == 0


class _ReadsLevel0WhenNonSquare(nn.Module):
    """A read set that depends on the input shape (unsupported in general; the probe's union over
    square, odd and non-square sizes catches this case)."""

    def __init__(self, specs):
        super().__init__()
        self.conv = nn.Conv2d(specs[2].channels, 4, 1)
        self.out_spec = FeatureSpec(4, specs[2].stride)

    def forward(self, feats):
        y = self.conv(feats[2])
        if feats[0].shape[-1] != feats[0].shape[-2]:
            y = y + feats[0].mean()
        return y


def test_probe_takes_the_union_over_several_sizes():
    assert used_levels(_ReadsLevel0WhenNonSquare(SPECS), SPECS) == [0, 2]


class _Declared(nn.Module):
    read_levels = (1, 2)

    def __init__(self, specs):
        super().__init__()
        self.out_spec = FeatureSpec(4, 8)

    def forward(self, feats):
        raise AssertionError("a declared read set must not be probed")


def test_declared_read_levels_skip_the_probe():
    assert used_levels(_Declared(SPECS), SPECS) == [1, 2]
    bad = _Declared(SPECS)
    bad.read_levels = (0, 3)
    with pytest.raises(ValueError, match="read_levels"):
        used_levels(bad, SPECS)
