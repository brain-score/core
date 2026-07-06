"""Typed convenience helpers over the UMI v2.0 streaming substrate."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .io_catalog import modalities_to_input_channels
from .streaming import InMemorySession, StreamEvent, parse_channel
from .events import PerturbationApplied, StateChange
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


class StimulusSetSession(InMemorySession):
    """Buffered open-loop session built from a StimulusSet-like table."""

    def __init__(self, stimulus_set, record: str = "IT"):
        self.stimulus_set = stimulus_set
        self.record = record
        super().__init__(_stimulus_events(stimulus_set))

    @classmethod
    def from_stimulus_set(cls, stimulus_set, record: str = "IT"):
        return cls(stimulus_set, record=record)

    def collect(self, channel: str) -> NeuroidAssembly:
        """Collect emitted neural events into a NeuroidAssembly."""
        matching = [event for event in self.emitted if event.channel == channel]
        if not matching:
            raise ValueError(f"No emitted events for channel {channel!r}.")

        if len(matching) == 1 and isinstance(matching[0].payload, NeuroidAssembly):
            return matching[0].payload

        return _collect_neural_events(channel, matching)


class BehavioralSession(InMemorySession):
    """Buffered session for open-loop behavioral scoring."""

    def __init__(self, task_context):
        self.task_context = task_context
        self.scoring_stimuli = _behavior_stimuli_to_score(task_context)
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


def stimulus_session(stimulus_set, record: str = "IT") -> StimulusSetSession:
    return StimulusSetSession.from_stimulus_set(stimulus_set, record=record)


def behavior_session(task_context) -> BehavioralSession:
    return BehavioralSession.from_task_context(task_context)


def state_change_session(state_change: StateChange) -> StateChangeSession:
    return StateChangeSession.from_state_change(state_change)


def score_stimuli(subject, stimulus_set, record: str = "IT") -> NeuroidAssembly:
    session = stimulus_session(stimulus_set, record=record)
    _drive_via_process(subject, session, stimulus_set, record=record)
    return session.collect(f"neural:{record}")


def score_behavior(subject, task_context) -> BehavioralAssembly:
    session = behavior_session(task_context)
    _drive_behavior_via_process(subject, session, task_context)
    return session.collect("behavior")


def apply_state_change(subject, state_change: StateChange):
    session = state_change_session(state_change)
    _drive_state_change_via_process(subject, session, state_change)
    return session.collect("perturbation")


def _drive_via_process(subject, session: StimulusSetSession, stimulus_set,
                       record: str = "IT") -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
    subject.start_recording(record)
    output = subject.process(stimulus_set)
    session.emit(StreamEvent(
        channel=f"neural:{record}",
        payload=output,
        t_ms=0.0,
        meta={"driver": "process", "record": record},
    ))


def _drive_state_change_via_process(subject, session: StateChangeSession,
                                    state_change: StateChange) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
    result = subject.process(state_change)
    handle_id = _state_change_handle_id(state_change, result)
    session.emit(StreamEvent(
        channel="perturbation",
        payload=result,
        t_ms=0.0,
        meta={
            "driver": "process",
            "kind": state_change.kind,
            "handle_id": handle_id,
        },
    ))


def _drive_behavior_via_process(subject, session: BehavioralSession,
                                task_context) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native interact()."""
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
        meta={"driver": "process", "task_type": task_context.task_type},
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
