"""Streaming primitives for the Unified Model Interface session contract."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import re
from typing import Any, Deque, Iterable, Optional


_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ADDRESS_RE = re.compile(r"^[A-Za-z0-9_./\-\[\]:]+$")


@dataclass
class StreamEvent:
    channel: str
    payload: Any
    t_ms: Optional[float]  # None when an environment has no physical time scale.
    meta: dict = field(default_factory=dict)


def parse_channel(name: str) -> tuple[str, str | None]:
    """Parse a channel name into ``(family, address)``.

    Channel names follow the channel grammar ``family`` or ``family:address``.
    Registry-specific meaning and payload validation are handled elsewhere.
    """
    if not isinstance(name, str):
        raise TypeError("channel name must be a string")

    if ":" in name:
        family, address = name.split(":", 1)
    else:
        family, address = name, None

    if not _FAMILY_RE.fullmatch(family):
        raise ValueError(
            "channel family must start with a lowercase letter and contain "
            "only lowercase letters, digits, and underscores"
        )

    if address is None:
        return family, None

    if (
        not address
        or address.startswith(":")
        or address.endswith(":")
        or not _ADDRESS_RE.fullmatch(address)
    ):
        raise ValueError(
            "channel address must be non-empty and contain only non-space "
            "address characters"
        )

    return family, address


class Session(ABC):
    """Input delivery, requested outputs, task settings, and result collection.

    Sources implement next_input and respond. The optional settings below are the
    common configuration consumed by BrainScoreModel and experiment replay.

    ``complete`` says whether the inputs ran to the end of the episode. Fixed
    inputs are always complete; an environment session stays False until a
    terminal step. Runners must not accept an incomplete session as a measurement.
    """
    complete: bool = True
    requested_output_channels: Optional[tuple[str, ...]] = None
    stimulus_set: Any = None
    task_context: Any = None
    time_bins: Any = None
    streaming = False
    context_fields = ('requested_output_channels', 'stimulus_set', 'task_context',
                      'time_bins', 'streaming')

    def collect(self, channel: Optional[str] = None) -> Any:
        """Return emitted events; specialized sources may assemble their payloads."""
        events = getattr(self, 'emitted', ())
        return [event for event in events if channel is None or event.channel == channel]

    @abstractmethod
    def next_input(self) -> Optional[StreamEvent]:
        """Return the next input event, or ``None`` at end of input.

        This is a synchronous contract. A live source waits for its next event
        or raises on timeout; temporary unavailability must not return None."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Old source implementations fulfill the same response contract. Adapt
        # their method at class creation, before ABC checks abstract methods.
        if 'emit' in cls.__dict__ and 'respond' not in cls.__dict__:
            cls.respond = cls.__dict__['emit']
        elif 'respond' in cls.__dict__ and 'emit' not in cls.__dict__:
            # Bind the alias to this implementation, so super().emit() in an
            # existing subclass does not re-enter that subclass's override.
            cls.emit = cls.__dict__['respond']
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def respond(self, event: StreamEvent) -> None:
        """Accept a subject response; environments may advance in response."""

    def emit(self, event: StreamEvent) -> None:
        """Compatibility spelling of respond for existing subjects."""
        self.respond(event)


class BufferedSession(Session):
    """Deliver fixed inputs and collect responses without environment feedback."""

    def __init__(self, inputs: Iterable[StreamEvent] = (), *,
                 requested_output_channels: Optional[Iterable[str]] = None,
                 stimulus_set: Any = None, task_context: Any = None,
                 time_bins: Any = None, streaming: bool = False) -> None:
        from .streaming_helpers import _stimulus_events, _behavior_events, _behavior_stimuli_to_score
        inputs = tuple(inputs)
        if stimulus_set is not None:
            if inputs:
                raise ValueError('Provide input events or a stimulus table, not both.')
            inputs = _stimulus_events(stimulus_set)
        elif task_context is not None and not inputs:
            stimulus_set = _behavior_stimuli_to_score(task_context)
            inputs = _behavior_events(task_context, stimulus_set)
        if requested_output_channels is not None:
            if isinstance(requested_output_channels, str):
                raise TypeError('requested_output_channels must be a sequence of channel names.')
            self.requested_output_channels = tuple(requested_output_channels)
            for channel in self.requested_output_channels:
                parse_channel(channel)
        # Compatibility subclasses install their settings before calling super.
        if stimulus_set is not None:
            self.stimulus_set = stimulus_set
        if task_context is not None:
            self.task_context = task_context
        if time_bins is not None:
            self.time_bins = time_bins
        if streaming:
            self.streaming = True
        # Buffered inputs are already available, so reject the entire batch
        # before the subject can compute on its first item.
        from .io_catalog import validate, INPUT
        for event in inputs:
            if not isinstance(event, StreamEvent):
                raise TypeError('Session inputs must be StreamEvent objects')
            failures = validate(event.channel, event.payload, direction=INPUT)
            if failures:
                raise ValueError('Invalid session input: ' + '; '.join(failures))
        self._inputs: Deque[StreamEvent] = deque(inputs)
        self.emitted: list[StreamEvent] = []

    def collect(self, channel: Optional[str] = None) -> Any:
        """Collect neural/behavioral assemblies, or events for other channels."""
        from .streaming_helpers import _collect_neural_events, _collect_behavior_events
        if channel is None:
            channels = self.requested_output_channels
            if not channels:
                return list(self.emitted)
            if len(channels) == 1:
                return self.collect(channels[0])
            return {name: self.collect(name) for name in channels}
        events = super().collect(channel)
        if not events:
            raise ValueError(f'No emitted events for channel {channel!r}.')
        family, _ = parse_channel(channel)
        if family in ('neural', 'behavior'):
            if all('presentation' in getattr(event.payload, 'dims', ()) for event in events):
                if len(events) == 1:
                    return events[0].payload
                import xarray as xr
                return xr.concat([event.payload for event in events], dim='presentation')
            return (_collect_neural_events(channel, events) if family == 'neural'
                    else _collect_behavior_events(events))
        return events

    def next_input(self) -> Optional[StreamEvent]:
        if not self._inputs:
            return None
        return self._inputs.popleft()

    def respond(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)


# Compatibility import; BufferedSession is the name used in new examples.
InMemorySession = BufferedSession
