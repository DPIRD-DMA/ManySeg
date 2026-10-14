import os

import pytest
import torch


def _usable(name: str) -> bool:
    """A device counts only if it can actually allocate (GitHub's macOS VMs report MPS as
    available but every allocation fails)."""
    try:
        torch.ones(8, device=name).sum().item()
        return True
    except RuntimeError:
        return False


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available() and _usable("cuda"):
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() \
            and _usable("mps"):
        devs.append("mps")
    return devs


DEVICES = _available_devices()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request) -> torch.device:
    """Every available device. Unavailable devices are skipped, never silently replaced."""
    name = request.param
    if name not in DEVICES:
        pytest.skip(f"{name} not available or not usable on this machine")
    return torch.device(name)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Isolate torch.compile caches between tests.

    Many tests compile *different* models through the same ``DenseModel.forward`` code object;
    without a reset, Dynamo's per-code-object cache fills across tests and hits its recompile
    limit. A real program compiles one model, so this is a test-isolation concern only.
    """
    yield
    import sys

    if "torch._dynamo" in sys.modules:
        sys.modules["torch._dynamo"].reset()


# Cap intra-op threads per test process: pytest-xdist workers each default to one thread per
# core, which oversubscribes the (shared) machine badly.
torch.set_num_threads(int(os.environ.get("MANYSEG_TEST_THREADS", "4")))
