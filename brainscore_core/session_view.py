"""A session view that delivers only the subject's declared input channels."""

from .streaming import Session, StreamEvent


class InputSession(Session):
    """Keep the experiment intact while ignoring streams the subject cannot use."""

    def __init__(self, session: Session, channels: set[str]) -> None:
        self._session = session
        self._channels = frozenset(channels)
        self.requested_output_channels = session.requested_output_channels
        self.task_context = session.task_context
        self.time_bins = session.time_bins
        self.streaming = session.streaming
        self.stimulus_set = session.stimulus_set
        if self.stimulus_set is not None:
            from .streaming_helpers import _channel_for_stimulus_column, _preserve_stimulus_set_type
            omitted = [column for column in self.stimulus_set.columns
                       if (channel := _channel_for_stimulus_column(column)) is not None
                       and channel not in self._channels]
            if omitted:
                self.stimulus_set = _preserve_stimulus_set_type(
                    self.stimulus_set, self.stimulus_set.drop(columns=omitted))

    @property
    def complete(self) -> bool:
        return self._session.complete

    def next_input(self):
        while (event := self._session.next_input()) is not None:
            # Permanent typed environment helpers still carry EnvironmentStep.
            # Their observation stream is declared and checked at pre-flight.
            if not isinstance(event, StreamEvent):
                from .events import EnvironmentStep
                from .io_catalog import INPUT, validate
                if not isinstance(event, EnvironmentStep):
                    raise TypeError('Session inputs must be StreamEvent objects')
                if 'observation' not in self._channels:
                    continue
                failures = validate('observation', event, direction=INPUT)
                if failures:
                    raise ValueError('Invalid session input: ' + '; '.join(failures))
                return event
            if (event.channel in self._channels
                    or event.channel.split(':', 1)[0] in self._channels):
                if isinstance(event, StreamEvent):
                    from .io_catalog import validate, INPUT
                    failures = validate(event.channel, event.payload, direction=INPUT)
                    if failures:
                        raise ValueError('Invalid session input: ' + '; '.join(failures))
                return event
        return None

    def respond(self, event: StreamEvent) -> None:
        self._session.respond(event)

    def collect(self, channel=None):
        return self._session.collect(channel)
