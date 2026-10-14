"""Metadata field encoders and the metadata encoder (vector inputs)."""

import math

import pytest
import torch
from manyseg.conditioning import FIELD_TYPES, MetadataEncoder, build_field

FIELDS = {
    "doy": {"type": "cyclic", "period": 365.25},
    "sun_elev": {"type": "continuous", "mean": 40.0, "std": 10.0},
    "latlon": {"type": "coords"},
    "sensor": {"type": "categorical", "num": 3},
}


def enc(**kw):
    return MetadataEncoder(fields=FIELDS, dim=16, **kw).eval()


def values(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([
        torch.rand(n, generator=g) * 365, 40 + 10 * torch.randn(n, generator=g),
        torch.rand(n, generator=g) * 180 - 90, torch.rand(n, generator=g) * 360 - 180,
        torch.randint(0, 3, (n,), generator=g).float()], dim=1)


def test_field_types_registered():
    assert {"cyclic", "continuous", "coords", "categorical"} <= set(FIELD_TYPES)


def test_unknown_field_type_is_clear():
    with pytest.raises(ValueError, match="cyclic"):
        build_field({"type": "angle"})


def test_cyclic_is_continuous_across_the_period_boundary():
    f = build_field({"type": "cyclic", "period": 365.25})
    v = torch.tensor([[0.0], [365.25], [365.2], [0.05], [182.6]])
    e, valid = f(v)
    assert e.shape == (5, 2) and valid.all()
    torch.testing.assert_close(e[0], e[1], atol=1e-5, rtol=0)
    assert (e[2] - e[3]).norm() < 0.01  # neighbours across the wrap stay neighbours
    assert (e[0] - e[4]).norm() > 1.9   # half a period apart is maximally far


@pytest.mark.parametrize("frequencies", [1, 2, 4])
def test_coords_are_continuous_across_the_dateline_and_at_the_poles(frequencies):
    f = build_field({"type": "coords", "frequencies": frequencies})
    e, _ = f(torch.tensor([[10.0, 179.999], [10.0, -179.999], [90.0, 0.0], [90.0, 120.0],
                           [-90.0, -45.0], [-90.0, 170.0]]))
    assert (e[0] - e[1]).norm() < 1e-2
    # every longitude is the same point at a pole, at every frequency
    assert (e[2] - e[3]).norm() < 1e-5 and (e[4] - e[5]).norm() < 1e-5
    torch.testing.assert_close(e[:, :3].norm(dim=1), torch.ones(6))  # unit-sphere part


def test_coords_extra_frequencies_widen_the_encoding():
    assert build_field({"type": "coords", "frequencies": 3}).width == 3 + 6 * 2


def test_continuous_normalisation():
    f = build_field({"type": "continuous", "mean": 40.0, "std": 10.0})
    e, _ = f(torch.tensor([[40.0], [55.0], [20.0]]))
    torch.testing.assert_close(e[:, 0], torch.tensor([0.0, 1.5, -2.0]))


def test_categorical_in_range_and_out_of_range():
    f = build_field({"type": "categorical", "num": 3, "dim": 4})
    e, valid = f(torch.tensor([[0.0], [1.0], [2.0], [3.0], [-1.0]]))
    assert e.shape == (5, 4)
    assert valid.tolist() == [True, True, True, False, False]
    assert not torch.allclose(e[0], e[1]) and not torch.allclose(e[1], e[2])


def test_categorical_validates_the_raw_value_on_both_boundaries():
    """Range and integrality are checked before rounding: -0.4 must not become id 0 and 2.6
    must not be rejected as id 3 for the wrong reason (it is not an integer at all)."""
    f = build_field({"type": "categorical", "num": 3})
    v = torch.tensor([[-0.4], [-1e-3], [0.0], [1.4], [2.0], [2.6], [2.0 + 1e-3], [3.0]])
    _, valid = f(v)
    assert valid.tolist() == [False, False, True, False, True, False, False, False]


def test_encoder_widths_and_output_shape():
    m = enc()
    assert m.columns == 1 + 1 + 2 + 1
    assert m.width == 2 + 1 + 3 + 8
    assert m(values()).shape == (4, 16)
    assert m.encode_fields(values()).shape == (4, m.width)


def test_plain_channels_vector():
    m = MetadataEncoder(channels=5, dim=8).eval()
    assert m.columns == m.width == 5 and len(m.field_names) == 5
    x = torch.randn(3, 5)
    torch.testing.assert_close(m.encode_fields(x), x)


def test_fields_xor_channels():
    with pytest.raises(ValueError, match="exactly one"):
        MetadataEncoder(fields=FIELDS, channels=3)
    with pytest.raises(ValueError, match="exactly one"):
        MetadataEncoder()


def test_dict_input_is_rejected_with_a_hint():
    m = enc()
    with pytest.raises(TypeError, match="pack"):
        m({"doy": torch.zeros(2)})


def test_wrong_width_is_clear():
    with pytest.raises(ValueError, match="5 columns"):
        enc()(torch.zeros(2, 4))


def test_pack_matches_column_order():
    m = enc()
    v = values()
    packed = m.pack({"doy": v[:, 0], "sun_elev": v[:, 1], "latlon": v[:, 2:4],
                     "sensor": v[:, 4]})
    torch.testing.assert_close(packed, v)
    with pytest.raises(KeyError, match="sensor"):
        m.pack({"doy": v[:, 0], "sun_elev": v[:, 1], "latlon": v[:, 2:4]})


def test_all_missing_ignores_values():
    m = enc()
    none = torch.zeros(4, 4)
    a, b = m(values(seed=0), none), m(values(seed=1), none)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all()


def test_nan_means_missing():
    m = enc()
    v = values()
    nan = torch.full_like(v, math.nan)
    torch.testing.assert_close(m(nan), m(v, torch.zeros(4, 4)))


def test_partial_mask_only_hides_masked_fields():
    m = enc()
    v1, v2 = values(seed=0), values(seed=1)
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]]).expand(4, 4)  # sun_elev missing
    mixed = v1.clone()
    mixed[:, 1] = v2[:, 1]
    torch.testing.assert_close(m(v1, mask), m(mixed, mask))  # masked field has no effect
    mixed = v1.clone()
    mixed[:, 0] = v2[:, 0]
    assert not torch.allclose(m(v1, mask), m(mixed, mask))  # unmasked field does


def test_missing_uses_learned_unknown_embeddings():
    m = enc()
    with torch.no_grad():
        m.unknown["sun_elev"].fill_(3.0)
    e = m.encode_fields(values(), torch.tensor([[1.0, 0.0, 1.0, 1.0]]).expand(4, 4))
    assert (e[:, 2] == 3.0).all()  # sun_elev occupies column 2 of the encoding


def test_out_of_range_category_is_unknown():
    m = enc()
    v = values()
    bad = v.clone()
    bad[:, 4] = 7
    mask = torch.ones(4, 4)
    mask[:, 3] = 0
    torch.testing.assert_close(m(bad), m(v, mask))


def test_field_dropout_only_in_training():
    m = enc(dropout=1.0)
    v = values()
    torch.testing.assert_close(m.train()(v), m.eval()(v, torch.zeros(4, 4)))
    assert not torch.allclose(m.eval()(v), m.eval()(v, torch.zeros(4, 4)))


def test_gradients_reach_mlp_embeddings_and_unknowns():
    m = enc().train()
    mask = torch.ones(4, 4)
    mask[:2] = 0
    m(values(), mask).square().sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
