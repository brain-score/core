"""Typed convenience helpers over the UMI v2.0 streaming substrate."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .contract import Subject
from .io_catalog import modalities_to_input_channels
from .streaming import InMemorySession, Session, StreamEvent, parse_channel
from .events import (
    EnvironmentResponse,
    EnvironmentStep,
    PerturbationApplied,
    StateChange,
)
from .supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)


STIMULUS_COLUMN_TO_MODALITY = {
    "vision": "vision",
    "image": "vision",
    "image_file_name": "vision",
    "image_path": "vision",
    "filename": "vision",
    "text": "text",
    "sentence": "text",
    "audio": "audio",
    "audio_file": "audio",
    "audio_file_name": "audio",
    "audio_path": "audio",
    "video": "video",
    "video_path": "video",
}

_STATE_CHANGE_KIND_TO_CHANNEL_FAMILY = {
    "ablation": "lesion",
    "lesion": "lesion",
    "stimulation": "stimulation",
    "pharmacological": "pharmacological",
}
_STIMULUS_INPUT_CHANNEL_FAMILIES = frozenset(
    modalities_to_input_channels(STIMULUS_COLUMN_TO_MODALITY.values())
)


class StimulusSetSession(InMemorySession):
    """Buffered open-loop session built from a StimulusSet-like table."""

    def __init__(self, stimulus_set, record: str = "IT", time_bins=None):
        self.stimulus_set = stimulus_set
        self.record = record
        self.time_bins = time_bins
        self.requested_output_channels = (f"neural:{record}",)
        super().__init__(_stimulus_events(stimulus_set))

    @classmethod
    def from_stimulus_set(cls, stimulus_set, record: str = "IT",
                          time_bins=None):
        return cls(stimulus_set, record=record, time_bins=time_bins)

    def collect(self, channel: str) -> NeuroidAssembly:
        """Collect emitted neural events into a NeuroidAssembly."""
        matching = [event for event in self.emitted if event.channel == channel]
        if not matching:
            raise ValueError(f"No emitted events for channel {channel!r}.")

        if len(matching) == 1 and _is_neural_assembly_payload(
            matching[0].payload
        ):
            return matching[0].payload

        return _collect_neural_events(channel, matching)


class BehavioralSession(InMemorySession):
    """Buffered session for open-loop behavioral scoring."""

    def __init__(self, task_context):
        self.task_context = task_context
        self.scoring_stimuli = _behavior_stimuli_to_score(task_context)
        self.requested_output_channels = ("behavior",)
        super().__init__(_behavior_events(task_context, self.scoring_stimuli))

    @classmethod
    def from_task_context(cls, task_context):
        return cls(task_context)

    def collect(self, channel: str) -> BehavioralAssembly:
        if channel != "behavior":
            raise ValueError(
                f"BehavioralSession.collect only supports 'behavior'; "
                f"got {channel!r}."
            )
        matching = [event for event in self.emitted if event.channel == channel]
        if not matching:
            raise ValueError(f"No emitted events for channel {channel!r}.")

        if len(matching) == 1 and isinstance(
            matching[0].payload, BehavioralAssembly
        ):
            return matching[0].payload

        return _collect_behavior_events(matching)


class StateChangeSession(InMemorySession):
    """Buffered session for perturbation apply/reset interactions."""

    def __init__(self, state_change: StateChange):
        self.state_change = state_change
        self.requested_output_channels = ("perturbation",)
        super().__init__([_state_change_event(state_change)])

    @classmethod
    def from_state_change(cls, state_change: StateChange):
        return cls(state_change)

    def collect(self, channel: str = "perturbation"):
        matching = [event for event in self.emitted if event.channel == channel]
        if not matching:
            raise ValueError(f"No emitted events for channel {channel!r}.")
        if len(matching) != 1:
            raise ValueError(
                f"Expected one emitted event for channel {channel!r}; "
                f"got {len(matching)}."
            )
        payload = matching[0].payload
        if payload is None:
            return matching[0].meta.get("handle_id")
        return payload


class EnvironmentSession(Session):
    """Live closed-loop session over an environment reset/step API."""

    def __init__(self, environment):
        self.environment = environment
        self.requested_output_channels = ("motor",)
        self.input_events: list[StreamEvent] = []
        self.emitted: list[StreamEvent] = []
        self._started = False
        self._done = False
        self._awaiting_emit = False
        self._pending_step: Optional[EnvironmentStep] = None
        self._last_step: Optional[EnvironmentStep] = None

    def next_input(self) -> Optional[EnvironmentStep]:
        if self._done or self._awaiting_emit:
            return None
        if not self._started:
            self._started = True
            self._pending_step = _environment_reset(self.environment)
        step = self._pending_step
        self._pending_step = None
        if step is None:
            self._done = True
            return None
        self._last_step = step
        self._awaiting_emit = True
        self.input_events.append(_environment_step_event(step))
        return step

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        if event.channel != "motor":
            raise ValueError(
                f"EnvironmentSession.emit only supports 'motor'; "
                f"got {event.channel!r}."
            )
        self.emitted.append(event)
        self._awaiting_emit = False
        if _environment_step_terminates(self._last_step):
            self._done = True
            return
        self._pending_step = _environment_step(
            self.environment, _motor_action_payload(event.payload)
        )
        if self._pending_step is None:
            self._done = True

    def collect(self, channel: str = "motor") -> list:
        if channel != "motor":
            raise ValueError(
                f"EnvironmentSession.collect only supports 'motor'; "
                f"got {channel!r}."
            )
        return [event.payload for event in self.emitted if event.channel == channel]


def stimulus_session(stimulus_set, record: str = "IT",
                     time_bins=None) -> StimulusSetSession:
    return StimulusSetSession.from_stimulus_set(
        stimulus_set, record=record, time_bins=time_bins
    )


def behavior_session(task_context) -> BehavioralSession:
    return BehavioralSession.from_task_context(task_context)


def state_change_session(state_change: StateChange) -> StateChangeSession:
    return StateChangeSession.from_state_change(state_change)


def environment_session(environment) -> EnvironmentSession:
    return EnvironmentSession(environment)


def neural_response(subject, stimulus_set, record: str = "IT",
                  time_bins=None) -> NeuroidAssembly:
    """Run ``subject`` on ``stimulus_set`` and return its neural response as a
    ``NeuroidAssembly``. This is the model's output, not a Score -- comparing it
    to measured data (and ceiling-normalizing) is the benchmark's job, via a metric.
    """
    session = stimulus_session(stimulus_set, record=record,
                               time_bins=time_bins)
    if _has_native_interact(subject):
        subject.interact(session)
    else:
        _drive_via_process(subject, session, stimulus_set, record=record,
                           time_bins=time_bins)
    return session.collect(f"neural:{record}")


def behavioral_response(subject, task_context) -> BehavioralAssembly:
    """Run ``subject`` on the task and return its behavior as a
    ``BehavioralAssembly`` -- the model's output, not a Score.
    """
    session = behavior_session(task_context)
    if _has_native_interact(subject):
        subject.interact(session)
    else:
        _drive_behavior_via_process(subject, session, task_context)
    return session.collect("behavior")


def apply_state_change(subject, state_change: StateChange):
    session = state_change_session(state_change)
    if _has_native_interact(subject):
        subject.interact(session)
    else:
        _drive_state_change_via_process(subject, session, state_change)
    return session.collect("perturbation")


def run_environment(subject, environment) -> list:
    session = environment_session(environment)
    if _has_native_interact(subject):
        subject.interact(session)
    else:
        _drive_environment_via_process(subject, session)
    return session.collect("motor")


def _drive_via_process(subject, session: StimulusSetSession, stimulus_set,
                       record: str = "IT", time_bins=None) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
    _start_recording(subject, record, time_bins=time_bins)
    output = subject.process(stimulus_set)
    session.emit(StreamEvent(
        channel=f"neural:{record}",
        payload=output,
        t_ms=0.0,
        meta={"driver": "process", "record": record},
    ))


def _drive_neural_session_via_process(
    subject, session, driver: str = "interact"
) -> None:
    input_events = _drain_stream_events(session)
    stimuli = _stimuli_from_stream_session(session, input_events)
    output_t_ms = input_events[-1].t_ms if input_events else 0.0
    use_multi_modality = _should_use_multi_modality(subject, input_events)
    time_bins = getattr(session, "time_bins", None)

    for channel in _requested_output_channels(session):
        family, region = parse_channel(channel)
        if family != "neural" or region is None:
            raise NotImplementedError(
                "neural interact currently supports requested "
                f"neural:<region> output channels; got {channel!r}."
            )
        _start_recording(subject, region, time_bins=time_bins)
        if use_multi_modality:
            output = subject.process(stimuli, multi_modality=True)
        else:
            output = subject.process(stimuli)
        session.emit(StreamEvent(
            channel=channel,
            payload=output,
            t_ms=output_t_ms,
            meta={"driver": driver, "record": region},
        ))


def _start_recording(subject, region: str, time_bins=None) -> None:
    if time_bins is None:
        subject.start_recording(region)
    else:
        subject.start_recording(region, time_bins=time_bins)


def _drive_neural_session_streaming(
    subject, session, driver: str = "interact_streaming"
) -> None:
    """Streaming perception: pull ONE input at a time, process it, emit, repeat.

    This is the perception-path twin of ``_drive_environment_session_via_process``.
    The batch driver drains the whole session up front and makes a single
    ``process()`` call, which is faster but requires every input to exist before
    scoring starts. This driver never holds more than one stimulus, so a session
    may generate inputs on demand (decode a video frame, poll a camera).

    Scores are identical to the batch path -- the model sees the same stimuli, one
    at a time instead of all at once. What changes is *when* inputs must exist, not
    what comes out. It is slower: batch size collapses to 1 and per-stimulus calls
    do not share an activations-cache entry.

    Opt in with ``session.streaming = True``.
    """
    channels = _requested_output_channels(session)
    regions = []
    for channel in channels:
        family, region = parse_channel(channel)
        if family != "neural" or region is None:
            raise NotImplementedError(
                "streaming neural interact supports requested neural:<region> "
                f"output channels; got {channel!r}."
            )
        regions.append((channel, region))

    time_bins = getattr(session, "time_bins", None)
    for _, region in regions:
        _start_recording(subject, region, time_bins=time_bins)

    stream_index = 0
    event = session.next_input()
    while event is not None:
        if not isinstance(event, StreamEvent):
            raise TypeError(
                "streaming neural interact expects StreamEvent inputs; got "
                f"{type(event).__name__}."
            )
        stimuli = _one_row_stimuli(session, event, stream_index)
        for channel, region in regions:
            _start_recording(subject, region, time_bins=time_bins)
            output = subject.process(stimuli)
            session.emit(StreamEvent(
                channel=channel,
                payload=output,
                t_ms=event.t_ms,
                meta={
                    "driver": driver,
                    "record": region,
                    "stream_index": stream_index,
                    "streaming": True,
                },
            ))
        stream_index += 1
        event = session.next_input()


def _one_row_stimuli(session, event: StreamEvent, stream_index: int):
    """Dispatch: a window event already carries its own stimuli; a single-input event
    is turned into one row. Tier 1 emits the latter, Tier 2 the former."""
    payload = event.payload
    if hasattr(payload, "columns") and hasattr(payload, "iterrows"):
        return payload            # already a StimulusSet-like window
    return _single_row_stimuli(session, event, stream_index)


def _single_row_stimuli(session, event: StreamEvent, stream_index: int):
    """A one-row StimulusSet for a single streamed input event.

    Prefers slicing the session's own stimulus_set so presentation metadata that
    metrics depend on (object_name and friends) survives; falls back to
    reconstructing from the event when the session has no table -- which is the
    case that matters for a genuinely generated stream.
    """
    stimulus_set = getattr(session, "stimulus_set", None)
    if stimulus_set is not None:
        index = event.meta.get("stimulus_index")
        if index is not None:
            try:
                row = stimulus_set.loc[[index]]
            except KeyError:
                row = None
            if row is not None and len(row) == 1:
                return _preserve_stimulus_set_type(stimulus_set, row)
    return _reconstruct_stimulus_set_from_events([event])


def _preserve_stimulus_set_type(original, sliced):
    """Keep StimulusSet-specific attributes (stimulus_paths) across a row slice."""
    for attribute in ("stimulus_paths", "identifier", "name"):
        if hasattr(original, attribute):
            try:
                setattr(sliced, attribute, getattr(original, attribute))
            except Exception:
                pass
    return sliced


def _drain_stream_events(session) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    while True:
        event = session.next_input()
        if event is None:
            return events
        if not isinstance(event, StreamEvent):
            raise TypeError(
                "neural interact expects StreamEvent inputs; got "
                f"{type(event).__name__}."
            )
        events.append(event)


def _stimuli_from_stream_session(session, events: list[StreamEvent]):
    if hasattr(session, "stimulus_set"):
        return session.stimulus_set
    return _reconstruct_stimulus_set_from_events(events)


def _should_use_multi_modality(subject, events: list[StreamEvent]) -> bool:
    if not hasattr(subject, "_process_multi_modality"):
        return False
    families = set()
    for event in events:
        family, _ = parse_channel(event.channel)
        if family in _STIMULUS_INPUT_CHANNEL_FAMILIES:
            families.add(family)
    return len(families) > 1


def _reconstruct_stimulus_set_from_events(events: list[StreamEvent]):
    """Rebuild a minimal StimulusSet from event-carried input columns.

    This fallback preserves only input-channel payload columns plus
    ``stimulus_id``. It is not metadata-faithful: metric-critical
    presentation metadata such as ``object_name`` must come from
    session.stimulus_set, or from a future event-carried metadata path.
    """
    if not events:
        raise ValueError(
            "neural interact received no input events and no session.stimulus_set "
            "context to reconstruct."
        )

    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    rows_by_key: dict[Any, dict[str, Any]] = {}
    row_order: list[Any] = []
    for index, event in enumerate(events):
        key = event.meta.get(
            "stimulus_index", event.meta.get("stimulus_id", index)
        )
        if key not in rows_by_key:
            rows_by_key[key] = {}
            row_order.append(key)
        row = rows_by_key[key]
        # Only the stream input column is reconstructible in this fallback.
        column = event.meta.get("column", event.channel)
        row[column] = event.payload
        if "stimulus_id" in event.meta:
            row.setdefault("stimulus_id", event.meta["stimulus_id"])

    return StimulusSet(pd.DataFrame([rows_by_key[key] for key in row_order]))


def _requested_output_channels(session) -> list[str]:
    requested = getattr(session, "requested_output_channels", None)
    if requested is not None:
        return list(requested)

    record = getattr(session, "record", None)
    if record is not None:
        if isinstance(record, str):
            return [f"neural:{record}"]
        return [f"neural:{region}" for region in record]

    raise ValueError(
        "streaming interact requires the session to declare "
        "requested_output_channels or record."
    )


def _has_native_interact(subject) -> bool:
    return getattr(type(subject), "interact", None) is not Subject.interact


def _drive_environment_via_process(subject, session: EnvironmentSession) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
    _drive_environment_session_via_process(subject, session, driver="process")


def _drive_environment_session_via_process(
    subject, session: EnvironmentSession, driver: str = "interact"
) -> None:
    """Drive a live environment session through existing EnvironmentStep dispatch."""
    step = session.next_input()
    while step is not None:
        response = subject.process(step)
        session.emit(StreamEvent(
            channel="motor",
            payload=response,
            t_ms=float(step.step_num),
            meta={
                "driver": driver,
                "step_num": step.step_num,
            },
        ))
        step = session.next_input()


def _drive_state_change_via_process(subject, session: StateChangeSession,
                                    state_change: StateChange) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
    _drive_state_change_session_via_process(
        subject, session, state_change=state_change, driver="process"
    )


def _drive_state_change_session_via_process(
    subject, session: StateChangeSession, state_change: StateChange = None,
    driver: str = "interact"
) -> None:
    """Drive a perturbation session through existing StateChange dispatch."""
    if state_change is None:
        event = session.next_input()
        if event is not None:
            if not isinstance(event, StreamEvent):
                raise TypeError(
                    "perturbation interact expects StreamEvent inputs; got "
                    f"{type(event).__name__}."
                )
            if not isinstance(event.payload, StateChange):
                raise TypeError(
                    "perturbation interact expects StateChange payloads; got "
                    f"{type(event.payload).__name__}."
                )
            state_change = event.payload
        else:
            state_change = getattr(session, "state_change", None)
    if state_change is None:
        raise ValueError(
            "perturbation interact requires a StateChange input event."
        )
    result = subject.process(state_change)
    handle_id = _state_change_handle_id(state_change, result)
    session.emit(StreamEvent(
        channel="perturbation",
        payload=result,
        t_ms=0.0,
        meta={
            "driver": driver,
            "kind": state_change.kind,
            "handle_id": handle_id,
        },
    ))


def _drive_behavior_via_process(subject, session: BehavioralSession,
                                task_context) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
    _drive_behavior_session_via_process(
        subject, session, task_context=task_context, driver="process"
    )


def _drive_behavior_session_via_process(
    subject, session: BehavioralSession, task_context=None,
    driver: str = "interact"
) -> None:
    """Drive a behavioral session through existing start_task/process logic."""
    if task_context is None:
        task_context = getattr(session, "task_context", None)
    if task_context is None:
        raise ValueError(
            "behavior interact requires a session.task_context to drive."
        )
    subject.start_task(task_context)
    stimuli = _behavior_stimuli_to_score(task_context)
    if stimuli is None:
        raise ValueError(
            "behavioral_response requires stimuli to elicit a response. Provide "
            "task_context.metadata['stimulus_set'] or fitting_stimuli."
        )
    output = subject.process(stimuli)
    session.emit(StreamEvent(
        channel="behavior",
        payload=output,
        t_ms=0.0,
        meta={"driver": driver, "task_type": task_context.task_type},
    ))


def _drive_behavior_session_streaming(
    subject, session: BehavioralSession, task_context=None,
    driver: str = "interact_streaming"
) -> None:
    """Behavioral trials delivered one at a time.

    Mirrors a real experiment: the subject is prepared ONCE (``start_task`` fits any
    readout on the fitting stimuli), then trials arrive one by one and a response
    comes back per trial. Re-preparing per trial would both refit the readout on every
    trial -- wrong -- and be pointlessly slow, so ``start_task`` is deliberately
    outside the loop.

    Same responses as the batch driver; what changes is that the trial set need not
    exist in full before the first response is produced.
    """
    if task_context is None:
        task_context = getattr(session, "task_context", None)
    if task_context is None:
        raise ValueError(
            "behavior interact requires a session.task_context to drive."
        )
    subject.start_task(task_context)              # once, before any trial
    stimuli = _behavior_stimuli_to_score(task_context)
    if stimuli is None:
        raise ValueError(
            "behavioral_response requires stimuli to elicit a response. Provide "
            "task_context.metadata['stimulus_set'] or fitting_stimuli."
        )
    for trial_index, (row_index, _row) in enumerate(stimuli.iterrows()):
        trial = _preserve_stimulus_set_type(stimuli, stimuli.loc[[row_index]])
        output = subject.process(trial)
        session.emit(StreamEvent(
            channel="behavior",
            payload=output,
            t_ms=float(trial_index),
            meta={"driver": driver, "task_type": task_context.task_type,
                  "trial_index": trial_index, "streaming": True},
        ))


class StreamingStimulusSetSession(Session):
    """Open-loop session that GENERATES one input event at a time.

    ``StimulusSetSession`` materializes every event in a deque up front. This one
    holds a row iterator and builds each event only when it is asked for, so at no
    point does the whole stream exist in memory. That is the property a real feed
    (a decoded video, a camera) needs, and it is why this class exists separately
    rather than as a flag on the buffered one.

    ``streaming = True`` is what routes ``interact`` to the one-at-a-time driver.
    """

    streaming = True

    def __init__(self, stimulus_set, record: str = "IT", time_bins=None):
        self.stimulus_set = stimulus_set
        self.record = record
        self.time_bins = time_bins
        self.requested_output_channels = (f"neural:{record}",)
        self.emitted: list[StreamEvent] = []
        self._pending: list[StreamEvent] = []
        self._rows = stimulus_set.iterrows()
        self._columns = [
            (column, _channel_for_stimulus_column(column))
            for column in list(getattr(stimulus_set, "columns", []))
            if _channel_for_stimulus_column(column) is not None
        ]
        self.max_events_held = 0     # observability: proves nothing is pre-drained

    def next_input(self) -> Optional[StreamEvent]:
        while not self._pending:
            try:
                row_index, row = next(self._rows)
            except StopIteration:
                return None
            stimulus_id = _row_value(row, "stimulus_id", default=row_index)
            t_ms = float(_row_value(row, "t_ms", default=0.0))
            for column, channel in self._columns:
                self._pending.append(StreamEvent(
                    channel=channel, payload=row[column], t_ms=t_ms,
                    meta={"stimulus_id": stimulus_id,
                          "stimulus_index": row_index, "column": column},
                ))
        self.max_events_held = max(self.max_events_held, len(self._pending))
        return self._pending.pop(0)

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)

    def collect(self, channel: str):
        """Concatenate the per-stimulus outputs emitted on ``channel``."""
        payloads = [e.payload for e in self.emitted if e.channel == channel]
        if not payloads:
            raise ValueError(f"No emitted events for channel {channel!r}.")
        if len(payloads) == 1:
            return payloads[0]
        import xarray as xr
        return xr.concat(payloads, dim="presentation")


class WindowedStreamSession(Session):
    """Tier 2: consume an OPEN-ENDED frame feed in fixed windows.

    ``window_plan`` in ``temporal.py`` tiles a known duration. A real feed has no
    known duration, so windowing here is incremental: pull frames from an iterator,
    hold at most one window's worth, emit when the window fills, slide by the stride,
    keep going until the feed runs dry.

    This is the shape that makes streaming useful rather than merely possible --
    Tier 1 collapses the batch to one stimulus, which is correct but slow. A window
    restores batching (the model sees ``window_ms`` of material at once) while the
    memory held stays constant no matter how long the feed runs.

    ``frames`` is any iterable of per-frame payloads; it is never listed, so a lazy
    decoder stays lazy. ``window_to_stimuli(frames, start_ms, end_ms)`` converts one
    window into whatever the subject consumes -- that is the caller's business, not
    the session's, because a video model and a frame-aggregation model want different
    things from the same feed.
    """

    streaming = True

    def __init__(self, frames, *, fps: float, window_ms: float,
                 stride_ms: Optional[float] = None, record: str = "IT",
                 window_to_stimuli=None, time_bins=None):
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if window_ms <= 0:
            raise ValueError(f"window_ms must be > 0, got {window_ms}")
        stride_ms = window_ms if stride_ms is None else stride_ms
        if stride_ms <= 0:
            raise ValueError(f"stride_ms must be > 0, got {stride_ms}")
        self._frames = iter(frames)
        self.fps = float(fps)
        self.window_ms = float(window_ms)
        self.stride_ms = float(stride_ms)
        self.record = record
        self.time_bins = time_bins
        self.requested_output_channels = (f"neural:{record}",)
        self.emitted: list[StreamEvent] = []
        self._window_frames = max(1, int(round(self.window_ms * self.fps / 1000.0)))
        self._stride_frames = max(1, int(round(self.stride_ms * self.fps / 1000.0)))
        self._buffer: list[Any] = []
        self._frames_consumed = 0
        self._window_index = 0
        self._exhausted = False
        self._window_to_stimuli = window_to_stimuli or _default_window_to_stimuli
        self.max_frames_held = 0          # observability: bounded by the window
        self.windows_emitted = 0

    def next_input(self) -> Optional[StreamEvent]:
        while len(self._buffer) < self._window_frames and not self._exhausted:
            try:
                self._buffer.append(next(self._frames))
                self._frames_consumed += 1
            except StopIteration:
                self._exhausted = True
        self.max_frames_held = max(self.max_frames_held, len(self._buffer))
        if not self._buffer:
            return None
        # a trailing partial window is still real material -- emit it rather than drop it
        window = list(self._buffer[:self._window_frames])
        start_ms = self._window_index * self.stride_ms
        end_ms = start_ms + (len(window) / self.fps) * 1000.0
        stimuli = self._window_to_stimuli(window, start_ms, end_ms)
        event = StreamEvent(
            channel="vision", payload=stimuli, t_ms=start_ms,
            meta={"window_index": self._window_index, "window_start_ms": start_ms,
                  "window_end_ms": end_ms, "n_frames": len(window),
                  "partial": len(window) < self._window_frames},
        )
        self._buffer = self._buffer[self._stride_frames:]
        if self._exhausted and len(self._buffer) < self._window_frames:
            # no more full windows can form; drain what remains next call, then stop
            if not self._buffer:
                self._exhausted = True
        self._window_index += 1
        self.windows_emitted += 1
        return event

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)

    def collect(self, channel: str):
        payloads = [e.payload for e in self.emitted if e.channel == channel]
        if not payloads:
            raise ValueError(f"No emitted events for channel {channel!r}.")
        if len(payloads) == 1:
            return payloads[0]
        import xarray as xr
        return xr.concat(payloads, dim="presentation")


def _default_window_to_stimuli(frames, start_ms, end_ms):
    """One row per frame, timestamped within the window.

    Deliberately simple: it keeps the frames addressable so a frame-aggregation model
    works out of the box. A native-video model should pass its own converter that
    packs the window into a single clip row.
    """
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    n = len(frames)
    step = (end_ms - start_ms) / n if n else 0.0
    return StimulusSet(pd.DataFrame({
        "stimulus_id": [f"w{int(start_ms)}_f{i}" for i in range(n)],
        "image_file_name": list(frames),
        "t_ms": [start_ms + i * step for i in range(n)],
    }))


class RealTimeStreamSession(Session):
    """Tier 3: a windowed feed that runs against a wall clock.

    Tiers 1 and 2 are about *when inputs must exist*. This tier adds the constraint
    that makes streaming hard in practice: the feed advances whether or not the model
    is finished. A window covers ``window_ms`` of world time, so if the model takes
    longer than that to process one, it is falling behind and something must give.

    The policy is explicit rather than implicit, because all three answers are
    legitimate and they measure different things:
      * ``'drop'``   -- skip whatever arrived while busy and stay current. What a live
                        system does. Coverage is lost; latency is bounded.
      * ``'lag'``    -- process everything and fall behind. What an offline run does.
                        Coverage is complete; latency grows without bound.
      * ``'error'``  -- refuse to continue. For a benchmark that is only meaningful if
                        real-time was actually achieved.

    The clock is injected so this is testable without sleeping: ``clock()`` returns
    seconds, ``sleep(seconds)`` paces the feed. Defaults are the real ones.

    ``realtime_factor`` is the honest headline: processing seconds per second of
    material. Below 1.0 means it keeps up.
    """

    streaming = True
    _POLICIES = ('drop', 'lag', 'error')

    def __init__(self, inner: 'WindowedStreamSession', *, policy: str = 'lag',
                 clock=None, sleep=None):
        if policy not in self._POLICIES:
            raise ValueError(f"policy must be one of {self._POLICIES}, got {policy!r}")
        self.inner = inner
        self.policy = policy
        self.record = inner.record
        self.time_bins = inner.time_bins
        self.requested_output_channels = inner.requested_output_channels
        self.emitted: list[StreamEvent] = []
        import time as _time
        self._clock = clock or _time.monotonic
        self._sleep = sleep or _time.sleep
        self._t0 = None
        self._last_emit_at = None
        self.windows_dropped = 0
        self.windows_delivered = 0
        self.max_lag_ms = 0.0
        self.processing_ms = 0.0

    def next_input(self) -> Optional[StreamEvent]:
        now = self._clock()
        if self._t0 is None:
            self._t0 = now
        else:
            # time spent by the caller since the last window came out = processing time
            self.processing_ms += (now - self._last_emit_at) * 1000.0

        event = self.inner.next_input()
        while event is not None:
            elapsed_ms = (self._clock() - self._t0) * 1000.0
            lag_ms = elapsed_ms - event.meta['window_start_ms']
            if lag_ms > self.inner.window_ms and self.policy != 'lag':
                if self.policy == 'error':
                    raise RuntimeError(
                        f"real-time budget missed: {lag_ms:.0f} ms behind at window "
                        f"{event.meta['window_index']} (window is "
                        f"{self.inner.window_ms:.0f} ms). Model is too slow for this "
                        f"feed; use policy='drop' to skip, or 'lag' to run offline.")
                self.windows_dropped += 1
                event = self.inner.next_input()
                continue
            self.max_lag_ms = max(self.max_lag_ms, lag_ms)
            # pace: do not hand over a window before the world has produced it
            wait_s = (event.meta['window_start_ms'] - elapsed_ms) / 1000.0
            if wait_s > 0:
                self._sleep(wait_s)
            self.windows_delivered += 1
            self._last_emit_at = self._clock()
            return event

        self._last_emit_at = self._clock()
        return None

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)

    @property
    def realtime_factor(self) -> float:
        """Processing seconds per second of material. < 1.0 keeps up."""
        material_ms = (self.windows_delivered + self.windows_dropped) * self.inner.stride_ms
        return (self.processing_ms / material_ms) if material_ms else 0.0

    def report(self) -> dict:
        return {
            'policy': self.policy,
            'windows_delivered': self.windows_delivered,
            'windows_dropped': self.windows_dropped,
            'max_lag_ms': round(self.max_lag_ms, 1),
            'realtime_factor': round(self.realtime_factor, 3),
            'kept_up': self.realtime_factor < 1.0 and self.windows_dropped == 0,
        }


def _stimulus_events(stimulus_set) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    columns = list(getattr(stimulus_set, "columns", []))
    input_columns = [
        (column, _channel_for_stimulus_column(column))
        for column in columns
        if _channel_for_stimulus_column(column) is not None
    ]

    for row_index, row in stimulus_set.iterrows():
        stimulus_id = _row_value(row, "stimulus_id", default=row_index)
        t_ms = float(_row_value(row, "t_ms", default=0.0))
        for column, channel in input_columns:
            events.append(StreamEvent(
                channel=channel,
                payload=row[column],
                t_ms=t_ms,
                meta={
                    "stimulus_id": stimulus_id,
                    "stimulus_index": row_index,
                    "column": column,
                },
            ))
    return events


def _behavior_events(task_context, scoring_stimuli) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    if task_context.instruction:
        events.append(StreamEvent(
            channel="instruction",
            payload=task_context.instruction,
            t_ms=0.0,
            meta={"task_type": task_context.task_type, "role": "instruction"},
        ))

    if task_context.fitting_stimuli is not None:
        events.extend(
            _tag_events(
                _stimulus_events(task_context.fitting_stimuli),
                role="fitting",
            )
        )

    if scoring_stimuli is not None:
        events.extend(
            _tag_events(
                _stimulus_events(scoring_stimuli),
                role="scoring",
            )
        )

    return events


def _tag_events(events: list[StreamEvent], role: str) -> list[StreamEvent]:
    tagged = []
    for event in events:
        meta = dict(event.meta)
        meta["role"] = role
        tagged.append(StreamEvent(
            channel=event.channel,
            payload=event.payload,
            t_ms=event.t_ms,
            meta=meta,
        ))
    return tagged


def _behavior_stimuli_to_score(task_context):
    for key in ("stimulus_set", "stimuli", "scoring_stimuli", "test_stimuli"):
        if key in task_context.metadata:
            return task_context.metadata[key]
    return task_context.fitting_stimuli


def _environment_reset(environment) -> Optional[EnvironmentStep]:
    if not hasattr(environment, "reset"):
        raise TypeError("run_environment requires an environment.reset() method")
    return _coerce_environment_step(environment.reset())


def _environment_step(environment, action) -> Optional[EnvironmentStep]:
    if not hasattr(environment, "step"):
        raise TypeError("run_environment requires an environment.step(action) method")
    return _coerce_environment_step(environment.step(action))


def _coerce_environment_step(result) -> Optional[EnvironmentStep]:
    if result is None:
        return None
    if isinstance(result, EnvironmentStep):
        return result
    raise TypeError(
        "environment reset/step must return an EnvironmentStep or None; "
        f"got {type(result).__name__}."
    )


def _environment_step_event(step: EnvironmentStep) -> StreamEvent:
    return StreamEvent(
        channel="observation",
        payload=step,
        t_ms=float(step.step_num),
        meta={
            "step_num": step.step_num,
            "is_first": step.is_first,
            "is_last": step.is_last,
            "is_terminal": step.is_terminal,
        },
    )


def _environment_step_terminates(step: Optional[EnvironmentStep]) -> bool:
    return step is None or bool(step.is_last or step.is_terminal)


def _motor_action_payload(payload):
    if isinstance(payload, EnvironmentResponse):
        return payload.action
    return payload


def _state_change_event(state_change: StateChange) -> StreamEvent:
    channel = _state_change_channel(state_change)
    meta = {"kind": state_change.kind}
    if state_change.handle_id is not None:
        meta["reset"] = state_change.handle_id
    return StreamEvent(
        channel=channel,
        payload=state_change,
        t_ms=0.0,
        meta=meta,
    )


def _state_change_channel(state_change: StateChange) -> str:
    family = _state_change_family(state_change)
    address = _state_change_address(state_change)
    return f"{family}:{address}" if address else family


def _state_change_family(state_change: StateChange) -> str:
    if state_change.kind == "reset":
        family = (
            state_change.metadata.get("family")
            or state_change.metadata.get("channel_family")
        )
        if family:
            return str(family)
        # Handle-id-only resets do not retain the original perturbation channel.
        return "lesion"

    family = _STATE_CHANGE_KIND_TO_CHANNEL_FAMILY.get(state_change.kind)
    if family is not None:
        return family

    # Unknown apply kinds use lesion as the generic perturbation fallback.
    return "lesion"


def _state_change_address(state_change: StateChange) -> Optional[str]:
    target = state_change.target
    if target is not None:
        layer = getattr(target, "layer", None)
        if layer:
            indices = getattr(target, "indices", None)
            if indices:
                start = indices[0]
                stop = indices[-1] + 1
                return f"{layer}[{start}:{stop}]"
            return str(layer)

    for key in ("address", "channel_address"):
        if state_change.metadata.get(key):
            return str(state_change.metadata[key])

    if state_change.handle_id:
        return str(state_change.handle_id)

    return None


def _state_change_handle_id(state_change: StateChange, result) -> Optional[str]:
    if isinstance(result, PerturbationApplied):
        return result.handle_id
    return state_change.handle_id


def _channel_for_stimulus_column(column: str) -> Optional[str]:
    modality = STIMULUS_COLUMN_TO_MODALITY.get(column)
    if modality is None:
        return None
    return next(iter(modalities_to_input_channels({modality})))


def _row_value(row, column: str, default: Any = None) -> Any:
    if column not in row.index:
        return default
    return row[column]


def _collect_neural_events(channel: str,
                           events: list[StreamEvent]) -> NeuroidAssembly:
    family, region = parse_channel(channel)
    if family != "neural":
        raise ValueError(f"collect({channel!r}) only supports neural channels.")

    values = []
    stimulus_ids = []
    for index, event in enumerate(events):
        payload = np.asarray(event.payload)
        if payload.ndim == 0:
            payload = payload.reshape(1)
        if payload.ndim != 1:
            raise ValueError(
                f"{channel}: raw neural event payloads must be 1-D; "
                f"got shape {payload.shape}."
            )
        values.append(payload)
        stimulus_ids.append(event.meta.get("stimulus_id", index))

    data = np.stack(values, axis=0)
    n_neuroids = data.shape[1]
    coords = {
        "stimulus_id": ("presentation", stimulus_ids),
        "stimulus_index": ("presentation", list(range(len(stimulus_ids)))),
        "neuroid_id": (
            "neuroid",
            [f"{channel}.{index}" for index in range(n_neuroids)],
        ),
        "neuroid_num": ("neuroid", list(range(n_neuroids))),
    }
    if region is not None:
        coords["region"] = ("neuroid", [region] * n_neuroids)

    return NeuroidAssembly(data, coords=coords,
                           dims=["presentation", "neuroid"])


def _is_neural_assembly_payload(payload) -> bool:
    dims = getattr(payload, "dims", None)
    return dims is not None and "presentation" in dims and "neuroid" in dims


def _collect_behavior_events(events: list[StreamEvent]) -> BehavioralAssembly:
    label_set = _behavior_label_set(events)
    values = []
    stimulus_ids = []
    for index, event in enumerate(events):
        payload = event.payload
        stimulus_ids.append(event.meta.get("stimulus_id", index))
        if _is_label_payload(payload):
            values.append(_one_hot(str(payload), label_set))
        else:
            row = np.asarray(payload, dtype=float)
            if row.ndim != 1:
                raise ValueError(
                    "raw behavior probability payloads must be 1-D; "
                    f"got shape {row.shape}."
                )
            values.append(row)

    data = np.stack(values, axis=0)
    if len(label_set) != data.shape[1]:
        label_set = [str(index) for index in range(data.shape[1])]

    return BehavioralAssembly(
        data,
        coords={
            "stimulus_id": ("presentation", stimulus_ids),
            "stimulus_index": ("presentation", list(range(len(stimulus_ids)))),
            "choice": ("choice", label_set),
        },
        dims=["presentation", "choice"],
    )


def _behavior_label_set(events: list[StreamEvent]) -> list[str]:
    for event in events:
        label_set = event.meta.get("label_set")
        if label_set is not None:
            return [str(label) for label in label_set]

    labels = []
    for event in events:
        if _is_label_payload(event.payload):
            label = str(event.payload)
            if label not in labels:
                labels.append(label)
    if labels:
        return labels

    first = np.asarray(events[0].payload)
    return [str(index) for index in range(first.shape[0])]


def _is_label_payload(payload) -> bool:
    return isinstance(payload, str)


def _one_hot(label: str, label_set: list[str]) -> np.ndarray:
    if label not in label_set:
        raise ValueError(f"label {label!r} not in label_set {label_set!r}.")
    row = np.zeros(len(label_set), dtype=float)
    row[label_set.index(label)] = 1.0
    return row
