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
    """``stimulus_set`` without the columns of input modalities not in ``channels``.

    Returns the same object when nothing has to be dropped, so identity (and
    the activation-cache key) survives. Never mutates the input table; a
    projection is a new table that keeps the stimulus set's identifier and
    paths. ``None`` passes through.
    """
    if stimulus_set is None:
        return None
    from brainscore_core._sessions.helpers import _channel_for_stimulus_column, preserve_stimulus_set_type
    declared = frozenset(channels)

    omitted = []
    for column in list(getattr(stimulus_set, 'columns', [])):
        channel = _channel_for_stimulus_column(column)
        if channel is not None and not channel_declared(channel, declared):
            omitted.append(column)
    if not omitted:
        return stimulus_set
    return preserve_stimulus_set_type(stimulus_set, stimulus_set.drop(columns=omitted))


# Keep the existing public module name stable.
declared_only.__module__ = 'brainscore_core.session_view'
