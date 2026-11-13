"""Public registration for external stimulus domains and channel payloads."""

from brainscore_core import io_catalog
from brainscore_core.io_catalog import CatalogEntry
from brainscore_core.streaming import StreamEvent


def register_channel(entry: CatalogEntry, *, columns=(), replace=False):
    """Register a channel family and optional stimulus columns without editing core.

    This is the registration function for code outside core. It adds the entry to
    the channel catalog and binds each column to the family, so that a stimulus
    table sends that column on the channel. A family or column that already
    exists needs ``replace=True``.
    """
    from brainscore_core.streaming import parse_channel
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
        existing = {**io_catalog.BUILTIN_STIMULUS_COLUMNS, **io_catalog.stimulus_columns()}.get(column)
        if existing is not None and existing != entry.name and not replace:
            raise ValueError(f'Column {column!r} already belongs to {existing!r}')
    io_catalog.register(entry, replace=replace)
    io_catalog._STIMULUS_COLUMNS.update({column: entry.name for column in columns})
    return entry


__all__ = ['CatalogEntry', 'StreamEvent', 'register_channel']


# Keep existing pickle references and public type names stable.
for _definition in (
    register_channel,
):
    _definition.__module__ = 'brainscore_core.extensions'
del _definition
