"""A session view that delivers only the streams a subject declares.

Contract Rules v3, rule 5.5: the benchmark sends every stream its human
participants received; a stream the subject cannot take in never enters it.
``BrainScoreModel.interact`` wraps each session in this view with its
``in_channels``, so ignoring an input is implicit: no stub, no placeholder,
and no change to the experiment, which keeps its own inputs and responses.
"""

from .streaming import Session, StreamEvent


class DeclaredInputSession(Session):
    """Expose ``session`` to a subject that declares ``channels``.

    Events whose channel (or channel family) is not declared are skipped.
    Non-event inputs, such as typed environment steps, pass through: their
    stream is declared and checked at pre-flight. A stimulus table loses the
    columns of undeclared input modalities, and keeps its identity when there
    is nothing to drop.
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
            self.stimulus_set = self._declared_columns(self.stimulus_set)

    def _declared(self, channel: str) -> bool:
        return channel in self._channels or channel.split(':', 1)[0] in self._channels

    def _declared_columns(self, stimulus_set):
        from .streaming_helpers import _channel_for_stimulus_column, _preserve_stimulus_set_type
        omitted = []
        for column in list(getattr(stimulus_set, 'columns', [])):
            channel = _channel_for_stimulus_column(column)
            if channel is not None and not self._declared(channel):
                omitted.append(column)
        if not omitted:
            return stimulus_set
        return _preserve_stimulus_set_type(stimulus_set, stimulus_set.drop(columns=omitted))

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
