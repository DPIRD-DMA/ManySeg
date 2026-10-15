"""Every provider must honour the benchmark contract (random weights; no network)."""

import pytest
import torch
from manyseg_bench.providers import PROVIDERS, ModelSpec, get_provider

PIXEL = sorted(set(PROVIDERS) - {"timm"})  # timm builds image-level models only


@pytest.mark.parametrize("name", PIXEL)
@pytest.mark.parametrize("hw", [(64, 64), (96, 128)])
def test_contract_shapes(name, hw):
    spec = ModelSpec("unet", "resnet18", weights=None, in_channels=3, num_classes=4)
    built = get_provider(name).build(spec, img_size=hw)
    model = built.model.eval()
    out = model(torch.randn(2, 3, *hw))
    assert type(out) is torch.Tensor
    assert out.shape == (2, 4, *hw)


@pytest.mark.parametrize("name", PIXEL)
def test_param_split_partitions_model(name):
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=2)
    b = get_provider(name).build(spec, img_size=(64, 64))
    ids = [id(p) for p in b.encoder_params + b.decoder_params]
    assert len(ids) == len(set(ids)) == len(list(b.model.parameters()))
    assert len(b.encoder_params) > 0 and len(b.decoder_params) > 0


@pytest.mark.parametrize("name", PIXEL)
def test_version_and_fingerprint(name):
    p = get_provider(name)
    assert p.version()
    b = p.build(ModelSpec("unet", "resnet18", weights=None), img_size=(64, 64))
    assert isinstance(b.weights_fingerprint, str)


def test_unknown_provider():
    with pytest.raises(ValueError, match="smp"):
        get_provider("nope")


@pytest.mark.network
def test_pretrained_weights_identical_across_providers():
    """Fairness precondition: tv_in1k resolves to the same encoder weights in every library."""
    spec = ModelSpec("unet", "resnet34", weights="tv_in1k", num_classes=3)
    fps = {n: get_provider(n).build(spec, img_size=(64, 64)).weights_fingerprint
           for n in PIXEL}
    assert len(set(fps.values())) == 1, fps


@pytest.mark.network
@pytest.mark.parametrize("encoder,weights", [("resnet50", "tv_in1k"), ("pvt_v2_b0", "in1k")])
def test_pretrained_weights_identical_for_image_level_providers(encoder, weights):
    """Classification parity precondition: ManySeg and timm load the same encoder weights,
    also for a non-RGB input (both adapt the stem from the same pretrained kernel)."""
    for c in (3, 13):
        spec = ModelSpec("none", encoder, weights=weights, in_channels=c, num_classes=10,
                         output="image")
        fps = {n: get_provider(n).build(spec).weights_fingerprint for n in ("manyseg", "timm")}
        assert len(set(fps.values())) == 1, (c, fps)
