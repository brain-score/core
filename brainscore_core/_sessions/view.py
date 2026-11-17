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
    from .payloads import preserve_stimulus_set_type
    declared = frozenset(channels)

    omitted = []
    for column in list(getattr(stimulus_set, 'columns', [])):
        channel = column_channel(column)
        if channel is not None and not channel_declared(channel, declared):
            omitted.append(column)
    if not omitted:
        return stimulus_set
    return preserve_stimulus_set_type(stimulus_set, stimulus_set.drop(columns=omitted))


def _present_channel(entry):
    """The channel of one ``TaskContext.present`` entry: the channel of its column when
    the column is in the catalog's column table, else the channel of its kind."""
    from brainscore_core.io_catalog import column_channel
    from brainscore_core.responses import PART_CHANNELS
    kind, _, column = entry.partition(':')
    return column_channel(column) or PART_CHANNELS.get(kind)


def _check_present(entry, columns):
    """Refuse a ``present`` entry that names no column of the table."""
    from brainscore_core.responses import part_columns
    kind, _, column = entry.partition(':')
    if column and column not in columns:
        raise ValueError(f'TaskContext.present names column {column!r}, which the stimulus table '
                         f'does not have; its columns are {sorted(columns)}.')
    if not column and not columns & set(part_columns(kind)):
        raise ValueError(f'TaskContext.present names {entry!r}, which no stimulus column provides.')


def task_input_channels(stimulus_set, task=None) -> frozenset:
    """The input channels a session of ``stimulus_set`` under ``task`` presents.

    When the task gives ``present``, they are the channels of its entries (a
    ``'kind:column'`` entry presents that column as its kind); otherwise the
    channels of the table's catalog columns. ``instruction`` is one more when
    the task gives one (the session carries it in its task). An entry that
    names a column the table does not have is refused with ``ValueError``. The
    one derivation of a table's inputs: a benchmark that keeps its table and
    task (pre-flight) and a ``TaskProtocol`` both use it.
    """
    columns = set(getattr(stimulus_set, 'columns', ()))
    present = getattr(task, 'present', None)
    if present is None:
        from brainscore_core.io_catalog import column_channel
        channels = {c for c in map(column_channel, columns) if c is not None}
    else:
        for entry in present:
            _check_present(entry, columns)
        channels = {_present_channel(entry) for entry in present}
    if getattr(task, 'instruction', None):
        channels.add('instruction')
    return frozenset(channels)


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
    declared = frozenset(channels)
    columns = set(getattr(stimulus_set, 'columns', ()))
    kept = []
    for entry in task.present:
        if channel_declared(_present_channel(entry), declared):
            kept.append(entry)
        elif stimulus_set is not None:
            _check_present(entry, columns)
    if len(kept) == len(task.present):
        return task
    return replace(task, present=tuple(kept))
