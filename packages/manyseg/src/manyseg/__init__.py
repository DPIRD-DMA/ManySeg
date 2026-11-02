"""ManySeg: composable dense-prediction models for PyTorch."""

__version__ = "0.1.0.dev0"

from .features import FeatureSpec  # noqa: E402
from .hub import push_to_hub  # noqa: E402
from .io import load, read_metadata, save  # noqa: E402
from .models import DenseModel, create_model  # noqa: E402

__all__ = ["DenseModel", "FeatureSpec", "create_model", "load", "push_to_hub", "read_metadata",
           "save", "__version__"]
