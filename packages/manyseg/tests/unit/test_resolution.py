"""Target downsampling, prediction upsampling and coordinate mapping for scaled heads."""

import pytest
import torch
import torch.nn.functional as F
from manyseg.utils import (
    downsample_labels,
    downsample_mean,
    downsample_sum,
    scale_factor,
    scaled_size,
    to_input_coords,
    to_output_coords,
    upsample_output,
)


@pytest.mark.parametrize("s,f", [(1, 1), (0.5, 2), (0.25, 4), (0.125, 8), (1 / 8, 8)])
def test_scale_factor(s, f):
    assert scale_factor(s) == f


@pytest.mark.parametrize("bad", [0, 0.3, 2, 1 / 16, True, "1/2", None])
def test_scale_factor_rejects(bad):
    with pytest.raises(ValueError, match="scale"):
        scale_factor(bad)


def test_scaled_size_is_ceil():
    assert scaled_size(65, 97, 0.25) == (17, 25)
    assert scaled_size(64, 96, 0.5) == (32, 48)
    assert scaled_size(1, 1, 0.125) == (1, 1)


# --- labels --------------------------------------------------------------------------------

def test_mode_downsampling_majority_per_block():
    y = torch.tensor([[[0, 0, 1, 1],
                       [0, 1, 1, 1],
                       [2, 2, 3, 0],
                       [2, 0, 3, 3]]])
    assert downsample_labels(y, 0.5).tolist() == [[[0, 1], [2, 3]]]


def test_mode_downsampling_respects_ignore_index():
    y = torch.tensor([[[255, 255, 1, 255],
                       [255, 2, 255, 255],
                       [255, 255, 3, 3],
                       [255, 255, 3, 255]]])
    # ignored pixels never vote; a fully ignored block stays ignored
    out = downsample_labels(y, 0.5, ignore_index=255)
    assert out.tolist() == [[[2, 1], [255, 3]]]
    y[0, :2, :2] = 255
    assert downsample_labels(y, 0.5, ignore_index=255)[0, 0, 0] == 255


def test_mode_with_num_classes_matches_inferred_range():
    y = torch.randint(0, 5, (2, 37, 53))
    y[0, :9, :9] = 255
    for s in (0.5, 0.25, 0.125):
        torch.testing.assert_close(downsample_labels(y, s, 255, num_classes=5),
                                   downsample_labels(y, s, 255))


def test_mode_ties_break_to_the_lowest_class():
    y = torch.tensor([[[3, 1], [1, 3]]])
    assert downsample_labels(y, 0.5).item() == 1


def test_mode_downsampling_odd_sizes_partial_blocks():
    y = torch.ones(2, 5, 7, dtype=torch.long)
    y[:, 4, :] = 2  # last row: a partial block containing only class 2
    out = downsample_labels(y, 0.25)
    assert out.shape == (2, 2, 2)
    assert out[:, 1].eq(2).all() and out[:, 0].eq(1).all()


def test_mode_downsampling_negative_ignore_index():
    y = torch.tensor([[[-1, -1], [-1, 0]]])
    assert downsample_labels(y, 0.5, ignore_index=-1).item() == 0
    assert downsample_labels(torch.full((1, 2, 2), -1), 0.5, ignore_index=-1).item() == -1


def test_nearest_downsampling_samples_the_block_centre():
    y = torch.arange(64).view(1, 8, 8)
    out = downsample_labels(y, 0.25, mode="nearest")
    assert out.tolist() == [[[y[0, 2, 2].item(), y[0, 2, 6].item()],
                             [y[0, 6, 2].item(), y[0, 6, 6].item()]]]
    # partial last block: clamped to the last in-image pixel
    assert downsample_labels(torch.arange(5).view(1, 1, 5), 0.25, mode="nearest").tolist() \
        == [[[2, 4]]]


def test_scale_one_is_identity():
    y = torch.randint(0, 3, (2, 7, 9))
    assert torch.equal(downsample_labels(y, 1), y)
    x = torch.randn(2, 3, 7, 9)
    assert torch.equal(downsample_mean(x, 1), x) and torch.equal(downsample_sum(x, 1), x)


# --- continuous and density targets ----------------------------------------------------------

def test_area_mean_exact_and_partial_blocks():
    x = torch.arange(20.0).view(1, 1, 4, 5)
    out = downsample_mean(x, 0.5)
    assert out.shape == (1, 1, 2, 3)
    torch.testing.assert_close(out[..., :2], F.avg_pool2d(x[..., :4], 2))
    # partial last column: mean over the in-image pixels only
    torch.testing.assert_close(out[0, 0, :, 2], torch.tensor([(4 + 9) / 2, (14 + 19) / 2]))


def test_area_mean_with_validity_mask():
    x = torch.tensor([[[[1.0, 3.0], [100.0, 5.0]]]])
    valid = torch.tensor([[[[True, True], [False, True]]]])
    torch.testing.assert_close(downsample_mean(x, 0.5, valid=valid), torch.tensor([[[[3.0]]]]))
    assert downsample_mean(x, 0.5, valid=torch.zeros_like(valid)).item() == 0.0


def test_sum_preserving_density():
    d = torch.rand(2, 1, 37, 53)
    for s in (0.5, 0.25, 0.125):
        out = downsample_sum(d, s)
        assert out.shape[-2:] == scaled_size(37, 53, s)
        torch.testing.assert_close(out.sum(dim=(1, 2, 3)), d.sum(dim=(1, 2, 3)))


def test_downsampling_on_device(device):
    y = torch.randint(0, 4, (2, 33, 47), device=device)
    assert downsample_labels(y, 0.25).device.type == device.type
    assert downsample_mean(y[:, None].float(), 0.25).device.type == device.type


# --- predictions back to input resolution ----------------------------------------------------

def test_upsample_output_shapes_and_crop():
    y = torch.randn(2, 3, 17, 25)
    assert upsample_output(y, (65, 97), 0.25).shape == (2, 3, 65, 97)


def test_upsample_nearest_round_trips_with_area_mean():
    x = torch.randn(1, 2, 8, 12)
    up = upsample_output(x, (32, 48), 0.25, mode="nearest")
    torch.testing.assert_close(downsample_mean(up, 0.25), x)


def test_upsample_nearest_integer_labels():
    lab = torch.tensor([[[1, 2], [3, 4]]])
    out = upsample_output(lab, (3, 4), 0.5, mode="nearest")
    assert out.dtype == lab.dtype and out.tolist() == [[[1, 1, 2, 2], [1, 1, 2, 2],
                                                        [3, 3, 4, 4]]]


def test_bilinear_upsample_of_a_constant_is_constant():
    out = upsample_output(torch.full((1, 1, 5, 7), 3.0), (37, 53), 0.125)
    torch.testing.assert_close(out, torch.full((1, 1, 37, 53), 3.0))


# --- coordinates -------------------------------------------------------------------------------

def test_coordinate_maps_are_inverse_and_pixel_centred():
    p = torch.tensor([[0.0, 0.0], [3.0, 5.0]])
    q = to_input_coords(p, 0.25)
    # output pixel 0 covers input pixels 0..3: centre 1.5
    torch.testing.assert_close(q, torch.tensor([[1.5, 1.5], [13.5, 21.5]]))
    torch.testing.assert_close(to_output_coords(q, 0.25), p)
    assert torch.equal(to_input_coords(p, 1), p)
