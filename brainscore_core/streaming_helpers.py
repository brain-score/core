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


def score_stimuli(subject, stimulus_set, record: str = "IT",
                  time_bins=None) -> NeuroidAssembly:
    session = stimulus_session(stimulus_set, record=record,
                               time_bins=time_bins)
    if _has_native_interact(subject):
        subject.interact(session)
    else:
        _drive_via_process(subject, session, stimulus_set, record=record,
                           time_bins=time_bins)
    return session.collect(f"neural:{record}")


def score_behavior(subject, task_context) -> BehavioralAssembly:
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
            "score_behavior requires stimuli to score. Provide "
            "task_context.metadata['stimulus_set'] or fitting_stimuli."
        )
    output = subject.process(stimuli)
    session.emit(StreamEvent(
        channel="behavior",
        payload=output,
        t_ms=0.0,
        meta={"driver": driver, "task_type": task_context.task_type},
    ))


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
