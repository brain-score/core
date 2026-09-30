"""Data-only metadata contract shared by the website and domain repositories."""

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
