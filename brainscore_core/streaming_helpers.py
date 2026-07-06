"""Typed convenience helpers over the UMI v2.0 streaming substrate."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .io_catalog import modalities_to_input_channels
from .streaming import InMemorySession, StreamEvent, parse_channel
from .supported_data_standards.brainio.assemblies import NeuroidAssembly


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


def stimulus_session(stimulus_set, record: str = "IT") -> StimulusSetSession:
    return StimulusSetSession.from_stimulus_set(stimulus_set, record=record)


def score_stimuli(subject, stimulus_set, record: str = "IT") -> NeuroidAssembly:
    session = stimulus_session(stimulus_set, record=record)
    _drive_via_process(subject, session, stimulus_set, record=record)
    return session.collect(f"neural:{record}")


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
