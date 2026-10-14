"""ManySeg: composable dense-prediction models for PyTorch."""

from .features import FeatureSpec
from .models import DenseModel, create_model

__version__ = "0.1.0.dev0"
__all__ = ["DenseModel", "FeatureSpec", "create_model", "__version__"]
