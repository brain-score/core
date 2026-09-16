"""Public registration for external stimulus domains and channel payloads."""

from . import io_catalog
from .io_catalog import CatalogEntry
from .streaming import StreamEvent


def register_channel(entry: CatalogEntry, *, columns=(), replace=False):
    """Register a schema and optional stimulus columns without editing core.

    Existing registrations require explicit replacement. The legacy catalog
    register() function retains its historical replacement behavior.
    """
    from .streaming import parse_channel
    from .brainscore_model import BrainScoreModel
    _, address = parse_channel(entry.name)
    if address is not None:
        raise ValueError("Register a channel family, not an addressed instance")
    columns = tuple(columns)
    if io_catalog.has(entry.name) and not replace:
        raise ValueError(f'Channel {entry.name!r} already registered')
    if columns and entry.direction not in (io_catalog.INPUT, io_catalog.BOTH):
        raise ValueError('Only input channels can own stimulus columns')
    for column in columns:
        if not isinstance(column, str) or not column:
            raise ValueError('Stimulus columns must be nonempty strings')
        existing = {**BrainScoreModel.COLUMN_TO_MODALITY, **io_catalog.stimulus_columns()}.get(column)
        if existing is not None and existing != entry.name and not replace:
            raise ValueError(f'Column {column!r} already belongs to {existing!r}')
    io_catalog.register(entry)
    io_catalog._STIMULUS_COLUMNS.update({column: entry.name for column in columns})
    return entry


__all__ = ['CatalogEntry', 'StreamEvent', 'register_channel']
