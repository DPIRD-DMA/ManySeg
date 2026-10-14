"""Non-raster conditioning inputs (metadata vectors): field encoders, the metadata encoder
and the modules that inject the embedding into a dense model. See ``docs/api_design.md``."""

from .build import MODES
from .encoder import MetadataEncoder
from .fields import FIELD_TYPES, build_field
from .inject import Conditioner, DecoderFiLM, FiLM

__all__ = ["FIELD_TYPES", "MODES", "Conditioner", "DecoderFiLM", "FiLM", "MetadataEncoder",
           "build_field"]
