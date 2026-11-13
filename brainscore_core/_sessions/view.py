"""Project a stimulus table onto the streams a subject declares.

Contract Rules v3, rule 5.5: the benchmark sends every stream its human
participants received; a stream the subject cannot take in never enters it.
Every shipped subject (``BrainScoreModel`` and the domain adapters) calls
``Session.declare_inputs`` with its ``in_channels``, so ignoring an input is
implicit: no stub, no placeholder, and no change to the experiment. This
module holds the table projection that method and ``start_task`` share.
"""

from ..streaming import channel_declared


def declared_only(stimulus_set, channels):
    """``stimulus_set`` without the stimulus columns whose channel is not in ``channels``.

    A column's channel comes from the catalog's column table
    (``io_catalog.column_channel``); a column that feeds no channel stays.

    Returns the same object when nothing has to be dropped, so identity (and
    the activation-cache key) survives. Never mutates the input table; a
    projection is a new table that keeps the stimulus set's identifier and
    paths. ``None`` passes through.
    """
    if stimulus_set is None:
        return None
    from brainscore_core.io_catalog import column_channel
    from brainscore_core._sessions.helpers import preserve_stimulus_set_type
    declared = frozenset(channels)

    omitted = []
    for column in list(getattr(stimulus_set, 'columns', [])):
        channel = column_channel(column)
        if channel is not None and not channel_declared(channel, declared):
            omitted.append(column)
    if not omitted:
        return stimulus_set
    return preserve_stimulus_set_type(stimulus_set, stimulus_set.drop(columns=omitted))


def _declared_task(task, stimulus_set, channels):
    """``task`` without the ``present`` entries of input streams not in ``channels``.

    An entry's channel is the channel of its column when the column is in the
    catalog's column table (the one ``declared_only`` drops), else the channel
    of its kind (``responses.PART_CHANNELS``). Returns the same object when nothing is dropped, so a full match is
    unchanged; never mutates ``task``. An entry that is dropped must still name
    a part the stimulus table has: a ``present`` entry no input provides is an
    error, never skipped.
    """
    if task is None or task.present is None:
        return task
    from dataclasses import replace
    from brainscore_core.io_catalog import column_channel
    from brainscore_core.responses import PART_CHANNELS, part_columns
    declared = frozenset(channels)
    columns = set(getattr(stimulus_set, 'columns', ()))
    kept = []
    for entry in task.present:
        kind, _, column = entry.partition(':')
        channel = column_channel(column) or PART_CHANNELS.get(kind)
        if channel_declared(channel, declared):
            kept.append(entry)
        elif stimulus_set is not None and not columns & ({column} if column else set(part_columns(kind))):
            raise ValueError(f'TaskContext.present names {entry!r}, which no stimulus column provides.')
    if len(kept) == len(task.present):
        return task
    return replace(task, present=tuple(kept))


# Keep the existing public module name stable.
declared_only.__module__ = 'brainscore_core.session_view'
