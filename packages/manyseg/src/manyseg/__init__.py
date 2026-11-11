"""ManySeg: composable dense-prediction models for PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("manyseg")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

from .features import FeatureSpec  # noqa: E402
from .hub import push_to_hub  # noqa: E402
from .io import load, read_metadata, save  # noqa: E402
from .models import DenseModel, create_model  # noqa: E402

__all__ = ["DenseModel", "FeatureSpec", "create_model", "load", "push_to_hub", "read_metadata",
           "save", "__version__"]
