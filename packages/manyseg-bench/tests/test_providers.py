"""Every provider must honour the benchmark contract (random weights; no network)."""

import pytest
import torch

from manyseg_bench.providers import PROVIDERS, ModelSpec, get_provider


@pytest.mark.parametrize("name", sorted(PROVIDERS))
@pytest.mark.parametrize("hw", [(64, 64), (96, 128)])
def test_contract_shapes(name, hw):
    spec = ModelSpec("unet", "resnet18", weights=None, in_channels=3, num_classes=4)
    built = get_provider(name).build(spec, img_size=hw)
    model = built.model.eval()
    out = model(torch.randn(2, 3, *hw))
    assert type(out) is torch.Tensor
    assert out.shape == (2, 4, *hw)


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_param_split_partitions_model(name):
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=2)
    b = get_provider(name).build(spec, img_size=(64, 64))
    ids = [id(p) for p in b.encoder_params + b.decoder_params]
    assert len(ids) == len(set(ids)) == len(list(b.model.parameters()))
    assert len(b.encoder_params) > 0 and len(b.decoder_params) > 0


@pytest.mark.parametrize("name", sorted(PROVIDERS))
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
           for n in PROVIDERS}
    assert len(set(fps.values())) == 1, fps
