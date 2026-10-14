import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader
from manyseg_bench.providers import ModelSpec, get_provider


def data(n=6, h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    # 8x8 blocks: per-pixel noise would make every pixel a boundary, where bilinear (image)
    # and nearest (mask) resampling legitimately disagree
    blocks = rng.integers(0, 2, (n, h // 8, w // 8)).astype(np.uint8)
    masks = blocks.repeat(8, axis=1).repeat(8, axis=2)
    # input "a" encodes the mask in channel 0, input "b" encodes it in channel 1
    a = rng.normal(size=(n, h, w, 2)).astype(np.float16)
    b = rng.normal(size=(n, h, w, 3)).astype(np.float16)
    a[..., 0] = masks * 10
    b[..., 1] = masks * 10
    return {"a": a, "b": b}, masks


def loader(stack, train=True, seed=0):
    imgs, masks = data()
    return GPULoader(imgs, masks, 3, {"a": [0, 0], "b": [0, 0, 0]}, {"a": [1, 1], "b": [1, 1, 1]},
                     torch.device("cpu"), train, seed=seed, stack=stack, vflip=True)


def test_stack_concatenates_in_key_order():
    x, _ = next(iter(loader(stack=True, train=False)))
    assert x.shape == (3, 5, 32, 32)


def test_dict_mode_returns_named_inputs():
    x, _ = next(iter(loader(stack=False, train=False)))
    assert set(x) == {"a", "b"} and x["a"].shape[1] == 2 and x["b"].shape[1] == 3


def test_augmentation_is_shared_across_inputs_and_mask():
    for x, y in loader(stack=False, train=True, seed=2):
        assert ((x["a"][:, 0] > 5) == (y > 0)).float().mean() > 0.97
        assert ((x["b"][:, 1] > 5) == (y > 0)).float().mean() > 0.97


def test_nan_inputs_are_imputed_to_normalised_mean():
    imgs, masks = data()
    imgs["a"][0, 0, 0, 1] = np.nan
    ld = GPULoader(imgs, masks, 6, {"a": [0, 0], "b": [0, 0, 0]}, {"a": [1, 1], "b": [1, 1, 1]},
                   torch.device("cpu"), False)
    x, _ = next(iter(ld))
    assert torch.isfinite(x).all()


def test_dict_inputs_require_capable_provider():
    assert get_provider("manyseg").supports_dict_inputs
    assert not get_provider("smp").supports_dict_inputs


@pytest.mark.parametrize("mode", ["stack", "dict"])
def test_manyseg_provider_multi_input(mode):
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=2, in_channels=5,
                     inputs={"a": 2, "b": 3}, input_mode=mode)
    model = get_provider("manyseg").build(spec).model.eval()
    x, _ = next(iter(loader(stack=mode == "stack", train=False)))
    assert model(x).shape == (3, 2, 32, 32)
