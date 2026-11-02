"""A session view that delivers only the streams a subject declares.

Contract Rules v3, rule 5.5: the benchmark sends every stream its human
participants received; a stream the subject cannot take in never enters it.
Every shipped subject (``BrainScoreModel`` and the domain adapters) wraps the
sessions it runs in this view with its ``in_channels``, so ignoring an input
is implicit: no stub, no placeholder, and no change to the experiment, which
keeps its own inputs and responses. A subject written against ``Subject``
directly reads only what it declares.
"""

from .streaming import Session, StreamEvent


def declared_only(stimulus_set, channels):
    """``stimulus_set`` without the columns of input modalities not in ``channels``.

    Returns the same object when nothing has to be dropped, so identity (and
    the activation-cache key) survives. Never mutates the input table; a
    projection is a new table that keeps the stimulus set's identifier and
    paths. ``None`` passes through.
    """
    if stimulus_set is None:
        return None
    from .streaming_helpers import _channel_for_stimulus_column, _preserve_stimulus_set_type
    declared = frozenset(channels)

    def is_declared(channel):
        return channel in declared or channel.split(':', 1)[0] in declared

    omitted = []
    for column in list(getattr(stimulus_set, 'columns', [])):
        channel = _channel_for_stimulus_column(column)
        if channel is not None and not is_declared(channel):
            omitted.append(column)
    if not omitted:
        return stimulus_set
    return _preserve_stimulus_set_type(stimulus_set, stimulus_set.drop(columns=omitted))


class DeclaredInputSession(Session):
    """Expose ``session`` to a subject that declares ``channels``.

    Events whose channel (or channel family) is not declared are skipped.
    Non-event inputs, such as typed environment steps, pass through: their
    stream is declared and checked at pre-flight. A stimulus table loses the
    columns of undeclared input modalities, and keeps its identity when there
    is nothing to drop. Measurement settings a facade attached to the session
    travel with it.
    """

    def __init__(self, session: Session, channels) -> None:
        self._session = session
        self._channels = frozenset(channels)
        for name in Session.context_fields:
            setattr(self, name, getattr(session, name, getattr(Session, name)))
        # Typed session subclasses carry their request as an attribute too.
        for name in ('record', 'state_change'):
            value = getattr(session, name, None)
            if value is not None:
                setattr(self, name, value)
        if self.stimulus_set is not None:
            self.stimulus_set = declared_only(self.stimulus_set, self._channels)

    def _declared(self, channel: str) -> bool:
        return channel in self._channels or channel.split(':', 1)[0] in self._channels

    @property
    def complete(self) -> bool:
        return self._session.complete

    @property
    def emitted(self):
        return getattr(self._session, 'emitted', [])

    def next_input(self):
        while (event := self._session.next_input()) is not None:
            if not isinstance(event, StreamEvent) or self._declared(event.channel):
                return event
        return None

    def respond(self, event: StreamEvent) -> None:
        self._session.respond(event)

    def collect(self, channel=None):
        return self._session.collect(channel)
