"""Data-only metadata contract shared by the website and domain repositories.

Included in the core distribution; importing it does not load scoring modules.
"""

from .contract import FIELD_SPECS, LIST_PATHS, MetadataError, load, validate, dump
from .policy import editability, protected_changes

__all__ = [
    "FIELD_SPECS",
    "LIST_PATHS",
    "MetadataError",
    "load",
    "validate",
    "dump",
    "editability",
    "protected_changes",
]
