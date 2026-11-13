"""Streaming primitives for the Unified Model Interface session contract."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import re
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from brainscore_core._evaluation.measures import note_response


_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ADDRESS_RE = re.compile(r"^[A-Za-z0-9_./\-\[\]:]+$")


@dataclass
class StreamEvent:
    channel: str
    payload: Any
    t_ms: Optional[float]  # None when an environment has no physical time scale.
    meta: dict = field(default_factory=dict)


def channel_declared(channel: str, declared) -> bool:
    """True when ``channel`` is one of the ``declared`` channels, by exact string.

    The one match rule for pre-flight and for the session input filter: a bare
    family such as ``vision`` does not stand for ``vision:image``.
    """
    return channel in declared


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


# Control channel (input direction): one catalog entry whose payload is one of
# ``begin``, ``respond`` or ``end``. ``end`` carries ``meta['reason']``.
CONTROL_CHANNEL = 'control'
BEGIN, RESPOND, END = 'begin', 'respond', 'end'
END_REASONS = ('terminal', 'truncated', 'failure')


def begin_event(t_ms: Optional[float] = None) -> StreamEvent:
    """The control event that opens an episode."""
    return StreamEvent(CONTROL_CHANNEL, BEGIN, t_ms)


def respond_event(t_ms: Optional[float] = None) -> StreamEvent:
    """The control event that asks the subject to respond now."""
    return StreamEvent(CONTROL_CHANNEL, RESPOND, t_ms)


def end_event(reason: str, t_ms: Optional[float] = None) -> StreamEvent:
    """The control event that closes an episode, with why it closed."""
    if reason not in END_REASONS:
        raise ValueError(f"end reason must be one of {END_REASONS}; got {reason!r}")
    return StreamEvent(CONTROL_CHANNEL, END, t_ms, meta={'reason': reason})


def _with_trial_memory(events: Iterable[Any], memory: str = 'none') -> Iterator[Any]:
    """Mark the trials in ``events`` with ``begin`` / ``end('terminal')`` control events.

    A trial is the run of events of one stimulus-table row: events with
    ``meta['stimulus_index']`` that share ``meta['table_row']``
    (rows of one input column carry none) and name no input column twice.
    Events without ``meta['stimulus_index']`` are not trials and pass through. ``none`` wraps each trial in its own episode;
    ``history`` opens one episode at the first trial and closes it after the
    last event. Lazy: nothing is read ahead of the trial being wrapped.
    """
    from brainscore_core.contract import _TRIAL_MEMORIES
    if memory not in _TRIAL_MEMORIES:
        raise ValueError(f'trial_memory must be one of {_TRIAL_MEMORIES}; got {memory!r}')
    opened, key, columns = False, None, set()
    for event in events:
        meta = event.meta if isinstance(event, StreamEvent) else {}
        is_trial = 'stimulus_index' in meta
        new_key = meta.get('table_row')
        starts_row = is_trial and (new_key != key or meta.get('column') in columns)
        if memory == 'none' and opened and (starts_row or not is_trial):
            yield end_event('terminal')
            opened = False
        if is_trial and not opened:
            yield begin_event(event.t_ms)
            opened = True
        if is_trial:
            columns = {meta.get('column')} if starts_row else columns | {meta.get('column')}
            key = new_key
        yield event
    if opened:
        yield end_event('terminal')


# Closed-loop steps are rows: a plain mapping of channel -> payload. One stream
# per channel is ``row[channel] = payload``. Several named streams on one channel
# (two cameras) are ``row['vision:image'] = NamedStreams({'wrist': a, 'exterior': b})``;
# the events carry the name in ``meta[STREAM_NAME_KEY]`` and the receiving side
# regroups them. Episode signals are not row keys: ``begin`` and ``end(reason)``
# are control events, and ``reward`` is an input channel of its own.
STREAM_NAME_KEY = 'camera'


class NamedStreams(dict):
    """Marks a row payload as several named streams on one channel."""


def row_events(row: Mapping[str, Any], t_ms: Optional[float] = None,
               meta: Optional[Mapping[str, Mapping]] = None,
               **common_meta: Any) -> list[StreamEvent]:
    """The events of one row: one per channel, one per name for ``NamedStreams``.

    ``meta`` maps a channel to extra event meta (for example ``{'reward':
    {'discount': 0.99}}``); ``common_meta`` goes on every event.
    """
    events = []
    for channel, payload in row.items():
        parse_channel(channel)
        own = {**common_meta, **(meta or {}).get(channel, {})}
        if isinstance(payload, NamedStreams):
            events.extend(StreamEvent(channel, value, t_ms, {**own, STREAM_NAME_KEY: name})
                          for name, value in payload.items())
        else:
            events.append(StreamEvent(channel, payload, t_ms, own))
    return events


def events_row(events: Iterable[StreamEvent]) -> dict:
    """The row of one step: events on one channel with names become ``{name: payload}``.

    Control events are skipped. Two unnamed events on one channel are an error.
    """
    row: dict = {}
    for event in events:
        if event.channel == CONTROL_CHANNEL:
            continue
        name = event.meta.get(STREAM_NAME_KEY)
        if name is not None:
            row.setdefault(event.channel, {})[name] = event.payload
        elif event.channel in row:
            raise ValueError(f'Two events on channel {event.channel!r} in one step; '
                             f'name them with meta[{STREAM_NAME_KEY!r}].')
        else:
            row[event.channel] = event.payload
    return row


class Source:
    """Where a session's input events come from.

    ``next_input`` returns the next event, or ``None`` at end of input. It is
    synchronous: a live source waits for its next event or raises on timeout;
    temporary unavailability must not return ``None``. ``respond`` receives each
    subject response, so a closed-loop source can advance on it; fixed input
    ignores it. A source may return the event the session should keep in place of
    the response (a receipt). ``complete`` says whether the input ran to the end of
    its episode; ``history_limit`` bounds how many responses the session keeps.
    ``checked`` says the source already checked its events against the catalog;
    otherwise the session checks each event as it delivers it. ``listed`` says
    the events are a finite list known up front: a subject may answer them in one
    call. Any other source is answered one event at a time.
    """
    checked: bool = False
    listed: bool = False
    complete: bool = True
    history_limit: Optional[int] = None

    def next_input(self) -> Optional[StreamEvent]:
        raise NotImplementedError

    def respond(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Receive a subject response. A no-op for fixed input."""
        return None

    def close(self) -> None:
        """Release what the source holds. A no-op for fixed input."""

    @contextmanager
    def observe_actions(self, callback: Callable):
        """Call ``callback(event)`` for each action proposal and receipt the source
        records. A no-op for a source that records none."""
        yield


class FixedSource(Source):
    """Input that does not depend on responses: any iterable of events.

    A list or tuple is known up front, so it is checked against the channel
    catalog now (rule 3.4). A lazy iterator is read one event at a time and is
    never materialized; the session checks each event when it delivers it.
    """

    def __init__(self, events: Iterable[Any] = ()) -> None:
        if isinstance(events, (list, tuple)):
            _validate_inputs(events)
            self.checked = True
            self.listed = True
        self._events: Iterator[Any] = iter(events)

    def next_input(self) -> Optional[StreamEvent]:
        return next(self._events, None)


class Session(Source):
    """Input delivery, requested outputs, task settings, and result collection.

    ``source`` is a ``Source`` or any iterable of events (made a ``FixedSource``).
    The source decides the delivery: a listed source is one batch, any other is
    answered one event at a time. The other arguments are the common
    configuration consumed by ``BrainScoreModel`` and experiment replay; a
    stimulus table or task is kept on the session so the batch drivers can
    answer it in one ``process`` call.

    ``complete`` asks the source whether the inputs ran to the end of the episode.
    Fixed inputs are always complete; a closed-loop source stays False until a
    terminal step. Runners must not accept an incomplete session as a measurement.
    """
    context_fields = ('outputs', 'stimuli', 'task',
                      'time_bins', 'streaming', 'settings')

    def __init__(self, source: Any, outputs: Optional[Iterable[str]] = None,
                 task: Any = None, *, stimuli: Any = None, time_bins: Any = None,
                 settings: Any = None) -> None:
        self.source = source if isinstance(source, Source) else FixedSource(source)
        self.outputs: Optional[tuple[str, ...]] = None
        if outputs is not None:
            if isinstance(outputs, str):
                raise TypeError('outputs must be a sequence of channel names.')
            self.outputs = tuple(outputs)
            for channel in self.outputs:
                parse_channel(channel)
        self.task = task
        self.stimuli = stimuli
        self.time_bins = time_bins
        # Per-measurement settings a legacy facade attaches (trial count, variance
        # request, recording type, legacy text). None when the session has none.
        self.settings = settings
        # The state changes the subject received and has not answered yet; each
        # perturbation response is read against its own request.
        self._pending_changes: list = []
        self.dropped = 0
        self.emitted: list[StreamEvent] = []
        self._declared: Optional[frozenset] = None
        self._control = False
        self._trial = 0
        self._rows: list = []
        self._row_trials: dict = {}  # row label -> trial numbers, ascending
        self._read_all = False
        self._answered: set = set()
        self.one_answer_per_trial = False
        self._observers: list[tuple] = []
        self._rewriters: list[Any] = []

    @property
    def streaming(self) -> bool:
        """Answer one event at a time: the source is not a list known up front."""
        return not self.source.listed

    def declare_inputs(self, channels: Iterable[str]) -> 'Session':
        """Keep only the inputs a subject declares; the subject calls this with its ``in_channels``.

        Events on other channels are skipped (exact match, see ``channel_declared``),
        and the stimulus table loses the columns that feed undeclared channels. The
        task the subject sees loses the ``present`` entries of those inputs, so a
        trial shows only the parts the subject takes in; the caller's task is not
        changed. Control events pass only when ``control`` is declared; a session
        that was never given a declaration delivers no control events. A second
        call narrows further. Returns ``self``.
        """
        from brainscore_core._sessions.view import _declared_task, declared_only
        declared = frozenset(channels)
        self._declared = declared if self._declared is None else self._declared & declared
        self._control = CONTROL_CHANNEL in self._declared
        self.task = _declared_task(self.task, self.stimuli, self._declared)
        self.stimuli = declared_only(self.stimuli, self._declared)
        return self

    def _request_control(self, driver: str) -> None:
        """For a driver that needs the control signals: ask for them on a session
        that was never given a declaration; refuse a declaration without them."""
        if self._declared is None:
            self._control = True
        elif not self._control:
            raise ValueError(f"{driver} needs the 'control' input. Add 'control' to the "
                             "subject's in_channels and answer on the respond signal.")

    @contextmanager
    def observe(self, on_input: Optional[Callable] = None, on_output: Optional[Callable] = None,
                on_error: Optional[Callable] = None, on_action: Optional[Callable] = None):
        """Call ``on_input(event)`` for every event the source delivers, before the
        input filter, and ``on_output(event)`` for every response, before it is kept.
        ``on_error(event, error)`` runs when the source cannot accept a response.
        ``on_action(event)`` runs for each action proposal and receipt the source
        records, including a rejected or unknown-status one; a receipt is also the
        response ``on_output`` gets, as the same object."""
        hooks = (on_input, on_output, on_error)
        self._observers.append(hooks)
        try:
            with (self.source.observe_actions(on_action) if on_action else nullcontext()):
                yield self
        finally:
            self._observers.remove(hooks)

    @contextmanager
    def rewriting_responses(self, rewrite: Callable[[StreamEvent], StreamEvent]):
        """Pass every response through ``rewrite(event) -> event`` while the block runs."""
        self._rewriters.append(rewrite)
        try:
            yield self
        finally:
            self._rewriters.remove(rewrite)

    def _notify(self, index: int, *args: Any) -> None:
        for hooks in tuple(self._observers):
            if hooks[index] is not None:
                hooks[index](*args)

    def next_input(self) -> Optional[StreamEvent]:
        while (event := self.source.next_input()) is not None:
            if not self.source.checked and isinstance(event, StreamEvent):
                from . import io_catalog
                check_event(event, io_catalog.INPUT)
            self._notify(0, event)
            if not isinstance(event, StreamEvent):
                return event
            if event.channel == CONTROL_CHANNEL:
                if event.payload == BEGIN:
                    self._trial += 1
                    self._rows.append(None)
                    self._row_trials.setdefault(None, []).append(self._trial)
                if self._control:
                    return event
            elif self._declared is None or channel_declared(event.channel, self._declared):
                self._note_row(event)
                self._note_state_change(event)
                return event
        self._read_all = True
        return None

    def _note_state_change(self, event: StreamEvent) -> None:
        from .events import StateChange
        if isinstance(event.payload, StateChange):
            self._pending_changes.append(event.payload)

    def _request_for(self, event: StreamEvent) -> Any:
        """The state change a perturbation response answers, or None.

        A response that names a handle answers the reset of that handle, else the
        oldest apply request. A response with no handle answers the only pending
        request; with several pending it is ambiguous and is refused.
        """
        if event.channel != 'perturbation':
            return None
        from .events import PerturbationApplied
        pending = self._pending_changes
        handle = event.payload.handle_id if isinstance(event.payload, PerturbationApplied) else None
        handle = handle or event.meta.get('handle_id')
        if handle is not None:
            request = next((r for r in pending if r.kind == 'reset' and r.handle_id == handle), None)
            if request is None and event.meta.get('kind') != 'reset':
                request = next((r for r in pending if r.kind != 'reset'), None)
        elif len(pending) > 1:
            raise ValueError("A perturbation response without a handle cannot tell which of "
                             f"{len(pending)} pending state changes it answers; put the handle "
                             "in the PerturbationApplied payload or in meta['handle_id'].")
        else:
            request = pending[0] if pending else None
        if request is not None:
            pending.remove(request)
        return request

    def _note_row(self, event: StreamEvent) -> None:
        # The row identity of the open trial: its first stimulus event names it.
        meta = event.meta
        if self._rows and self._rows[-1] is None and 'stimulus_index' in meta:
            label = meta['stimulus_index']
            self._rows[-1] = label
            self._row_trials[None].pop()
            try:
                self._row_trials.setdefault(label, []).append(len(self._rows))
            except TypeError:  # unhashable label: found by a scan in _trials_named
                pass

    @property
    def complete(self) -> bool:
        return self.source.complete

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> 'Session':
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def respond(self, event: StreamEvent) -> None:
        """Accept one subject response.

        When outputs were requested, the response must be on one of them (exact
        string match) and its payload must pass the catalog check for its
        channel; otherwise it raises ``ValueError``. When no outputs were
        requested (``None``) nothing is checked and any response is kept.
        """
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        for rewrite in tuple(self._rewriters):
            event = rewrite(event)
        if self.outputs is not None:
            if event.channel not in self.outputs:
                raise ValueError(f"response on {event.channel!r} was not requested; "
                                 f"requested: {list(self.outputs)}")
            from . import io_catalog
            check_event(event, io_catalog.OUTPUT)
        self._check_answered_once(event)
        request = self._request_for(event)
        try:
            kept = self.source.respond(event)
        except BaseException as error:
            self._notify(2, event, error)
            raise
        kept = event if kept is None else kept
        self._notify(1, kept)
        note_response(kept, request)
        self.emitted.append(kept)
        limit = self.source.history_limit
        if limit is not None and len(self.emitted) > limit:
            del self.emitted[0]
            self.dropped += 1

    def _check_answered_once(self, event: StreamEvent) -> None:
        """Refuse a second answer for one trial on one channel.

        Only a stimulus session with ``trial_memory='none'`` marks its trials, so
        only it sets ``one_answer_per_trial``. An answer belongs to the trial
        named by the ``stimulus_index`` in its meta, which is how a subject that
        reads every row first answers each row later. A label that repeats across
        rows names the open trial if it has the label, else the first trial with
        it not yet answered. With no such identity it
        belongs to the trial open now (the latest ``begin``); once the subject
        has read every event it is a delayed answer and belongs to the first
        trial not yet answered on that channel, so delayed answers must come in
        row order. A session read by a batch driver (no events read) is not checked.
        """
        if not (self.one_answer_per_trial and self._trial):
            return
        trial = self._trial
        if event.meta and 'stimulus_index' in event.meta:
            # The label can repeat across rows: prefer the open trial, then
            # the first trial with this label not yet answered on the channel.
            known = self._trials_named(event.meta['stimulus_index'])
            free = [t for t in known if (t, event.channel) not in self._answered]
            if known:
                trial = (self._trial if self._trial in free else
                         free[0] if free else known[-1])
        elif self._read_all:
            trial = next((t for t in range(1, self._trial + 1)
                          if (t, event.channel) not in self._answered), trial)
        if (trial, event.channel) in self._answered:
            raise ValueError(f"Channel {event.channel!r} was answered twice for trial {trial}. "
                             "Answer each requested channel once per trial.")
        self._answered.add((trial, event.channel))

    def _trials_named(self, label: Any) -> list:
        """The trial numbers whose row has this label, in order."""
        try:
            if label != label:  # a null label that is not equal to itself matches no row
                return []
            return self._row_trials.get(label, [])
        except TypeError:  # unhashable label
            return [i + 1 for i, row in enumerate(self._rows) if row == label]

    def collect(self, channel: Optional[str] = None) -> Any:
        """Return the responses, shaped by the channel family.

        ``neural`` and ``behavior`` give an assembly (one, or all concatenated along
        ``presentation``); ``perturbation`` gives its single payload (or the
        ``handle_id``); any other channel gives its events. With no channel: the
        events when nothing was requested, one result for one requested channel,
        else a dict keyed by requested channel.
        """
        if channel is None:
            channels = self.outputs
            if not channels:
                return list(self.emitted)
            if len(channels) == 1:
                return self.collect(channels[0])
            return {name: self.collect(name) for name in channels}
        if self.dropped:
            raise RuntimeError(
                f"history_limit={self.source.history_limit} dropped {self.dropped} "
                "response(s); stream them with observe() instead of collect()")
        events = [event for event in self.emitted if event.channel == channel]
        if not events:
            raise ValueError(f'No emitted events for channel {channel!r}.')
        family, _ = parse_channel(channel)
        if family in ('neural', 'behavior'):
            from brainscore_core._sessions.payloads import _collect_behavior_events, _collect_neural_events
            if all('presentation' in getattr(event.payload, 'dims', ()) for event in events):
                if len(events) == 1:
                    return events[0].payload
                import xarray as xr
                return xr.concat([event.payload for event in events], dim='presentation')
            return (_collect_neural_events(channel, events) if family == 'neural'
                    else _collect_behavior_events(events))
        if family == 'perturbation':
            if len(events) != 1:
                raise ValueError(f"Expected one emitted event for channel {channel!r}; got {len(events)}.")
            event = events[0]
            return event.payload if event.payload is not None else event.meta.get('handle_id')
        return events


def check_event(event: StreamEvent, direction: str) -> None:
    """Raise ``ValueError`` unless ``event`` passes the channel catalog.

    The channel must be a catalog family, addressed as the family allows, used in
    ``direction`` (``io_catalog.INPUT`` or ``OUTPUT``), and its payload must pass
    the family's type, ndim, dtype and validator checks. An entry with none of
    those checks nothing. An input control event skips the payload checks; its
    payload must be one of ``begin``, ``respond`` or ``end``.
    """
    from . import io_catalog
    if direction == io_catalog.INPUT and event.channel == CONTROL_CHANNEL:
        problems = ([] if event.payload in (BEGIN, RESPOND, END) else
                    [f"control payload must be 'begin', 'respond' or 'end'; got {event.payload!r}"])
    else:
        problems = io_catalog.validate(event.channel, event.payload, direction=direction)
    if problems:
        raise ValueError(f"{direction} event on {event.channel!r} failed the channel catalog: "
                         + "; ".join(problems[:5]))


def _validate_inputs(inputs) -> None:
    """Check known inputs against the channel catalog before any is delivered.

    Contract Rules v3, rule 3.4: a session whose inputs are known up front
    fails at construction, not at the first event the subject reads.
    """
    from . import io_catalog
    for event in inputs:
        if isinstance(event, StreamEvent):
            check_event(event, io_catalog.INPUT)
