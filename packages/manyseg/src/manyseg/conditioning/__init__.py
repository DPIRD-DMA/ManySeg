"""Non-raster conditioning inputs (metadata vectors): field encoders, the metadata encoder
and the modules that inject the embedding into a dense model."""

from .encoder import MetadataEncoder
from .fields import FIELD_TYPES, build_field

__all__ = ["FIELD_TYPES", "MetadataEncoder", "build_field"]
