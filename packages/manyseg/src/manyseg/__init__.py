"""ManySeg: composable dense-prediction models for PyTorch."""

__version__ = "0.1.0.dev0"

from .features import FeatureSpec  # noqa: E402
from .hub import push_to_hub  # noqa: E402
from .io import load, read_metadata, save  # noqa: E402
from .models import DenseModel, create_model  # noqa: E402

# Submodules reachable as attributes after ``import manyseg`` without importing them up front
# (``manyseg.quantize`` must not import torchao before it is used).
_LAZY_SUBMODULES = ("describe", "export", "quantize")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DenseModel", "FeatureSpec", "create_model", "load", "push_to_hub", "read_metadata",
           "save", "__version__"]
