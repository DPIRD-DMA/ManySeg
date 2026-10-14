"""Numerical contracts for pretrained stem-weight adaptation to arbitrary input channels."""

import pytest
import torch
import torch.nn.functional as F

from manyseg.adapters import CHANNEL_ADAPTERS, adapt_conv_weight

WEIGHT_ADAPTERS = ["repeat", "mean", "rgb_plus_random", "rgb_plus_zero"]


@pytest.fixture
def rgb_weight():
    return torch.randn(8, 3, 3, 3)


@pytest.mark.parametrize("method", WEIGHT_ADAPTERS)
@pytest.mark.parametrize("c", [1, 2, 3, 4, 7, 13])
def test_output_shape(rgb_weight, method, c):
    if method.startswith("rgb_plus") and c < 3:
        pytest.skip("rgb_plus_* is defined for C >= 3 (see test_rgb_plus_requires_at_least_three)")
    w = adapt_conv_weight(rgb_weight, c, method)
    assert w.shape == (8, c, 3, 3)
    assert w.dtype == rgb_weight.dtype


@pytest.mark.parametrize("method", WEIGHT_ADAPTERS)
def test_three_channels_is_identity(rgb_weight, method):
    torch.testing.assert_close(adapt_conv_weight(rgb_weight, 3, method), rgb_weight)


@pytest.mark.parametrize(
    "method,c",
    [("mean", c) for c in (1, 2, 4, 13)] + [("repeat", c) for c in (1, 6, 12)],
)
def test_constant_input_response_preserved(rgb_weight, method, c):
    """A channel-constant input gives the pretrained response.

    Holds for every C with "mean", but for "repeat" (timm's rule) only when C == 1 or C % 3 == 0:
    for C=4 the R filter is counted twice. That is a measurable bias of the timm default.
    """
    w = adapt_conv_weight(rgb_weight, c, method)
    plane = torch.randn(1, 1, 9, 9)
    ref = F.conv2d(plane.expand(1, 3, 9, 9), rgb_weight)
    out = F.conv2d(plane.expand(1, c, 9, 9), w)
    torch.testing.assert_close(out, ref)


def test_repeat_single_channel_sums_rgb_filters(rgb_weight):
    torch.testing.assert_close(
        adapt_conv_weight(rgb_weight, 1, "repeat"), rgb_weight.sum(1, keepdim=True)
    )


@pytest.mark.parametrize("c", [2, 4, 13])
def test_repeat_cycles_rgb_filters(rgb_weight, c):
    w = adapt_conv_weight(rgb_weight, c, "repeat")
    for i in range(c):
        torch.testing.assert_close(w[:, i], rgb_weight[:, i % 3] * 3 / c)


@pytest.mark.parametrize("method", ["rgb_plus_random", "rgb_plus_zero"])
@pytest.mark.parametrize("c", [4, 13])
def test_rgb_filters_kept_exactly(rgb_weight, method, c):
    w = adapt_conv_weight(rgb_weight, c, method)
    torch.testing.assert_close(w[:, :3], rgb_weight)


def test_rgb_plus_zero_ignores_extra_channels(rgb_weight):
    w = adapt_conv_weight(rgb_weight, 6, "rgb_plus_zero")
    x = torch.randn(2, 6, 9, 9)
    torch.testing.assert_close(F.conv2d(x, w), F.conv2d(x[:, :3], rgb_weight))


def test_rgb_plus_random_extra_channels_are_nonzero_and_smaller(rgb_weight):
    w = adapt_conv_weight(rgb_weight, 6, "rgb_plus_random")
    extra = w[:, 3:]
    assert extra.abs().sum() > 0
    assert extra.std() < rgb_weight.std()


@pytest.mark.parametrize("method", ["rgb_plus_random", "rgb_plus_zero"])
def test_rgb_plus_requires_at_least_three(rgb_weight, method):
    with pytest.raises(ValueError, match="at least 3"):
        adapt_conv_weight(rgb_weight, 2, method)


def test_unknown_method_lists_options(rgb_weight):
    with pytest.raises(ValueError, match="repeat"):
        adapt_conv_weight(rgb_weight, 4, "nope")


def test_registry_lists_all_methods():
    assert set(WEIGHT_ADAPTERS + ["projection"]) <= set(CHANNEL_ADAPTERS)


def test_does_not_mutate_input(rgb_weight):
    before = rgb_weight.clone()
    adapt_conv_weight(rgb_weight, 5, "repeat")
    torch.testing.assert_close(rgb_weight, before)
