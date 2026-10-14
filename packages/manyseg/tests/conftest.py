import pytest
import torch


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


DEVICES = _available_devices()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request) -> torch.device:
    """Every available device. Unavailable devices are skipped, never silently replaced."""
    name = request.param
    if name not in DEVICES:
        pytest.skip(f"{name} not available")
    return torch.device(name)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
