"""Convert session events, stimulus tables and response assemblies."""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
from ..io_catalog import modalities_to_input_channels
from ..streaming import StreamEvent, parse_channel
from ..events import PerturbationApplied, StateChange
from ..supported_data_standards.brainio.assemblies import (
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


def _one_row_stimuli(session, event: StreamEvent, stream_index: int):
    """Dispatch: a window event already carries its own stimuli; a single-input event
    is turned into one row. Single-input sessions emit the latter; windowed sessions emit the former."""
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
                return preserve_stimulus_set_type(stimulus_set, row)
    return reconstruct_stimulus_set_from_events([event])


def preserve_stimulus_set_type(original, sliced):
    """Keep StimulusSet-specific attributes (stimulus_paths) across a row slice."""
    for attribute in ("stimulus_paths", "identifier", "name"):
        if hasattr(original, attribute):
            try:
                setattr(sliced, attribute, getattr(original, attribute))
            except Exception:
                pass
    return sliced


def drain_session_inputs(session: Any) -> list[StreamEvent]:
    """Read every input event from ``session`` until it returns ``None``.

    Events come back in the order the session sent them. Anything that is not
    a ``StreamEvent`` raises ``TypeError``. Use this when a subject answers the
    whole session at once rather than one input at a time.
    """
    events: list[StreamEvent] = []
    while True:
        event = session.next_input()
        if event is None:
            return events
        if not isinstance(event, StreamEvent):
            raise TypeError(
                "neural participate expects StreamEvent inputs; got "
                f"{type(event).__name__}."
            )
        events.append(event)


def session_stimuli(session: Any, events: list[StreamEvent]) -> Any:
    """The stimulus table a session presented.

    Returns the session's own ``stimulus_set`` object unchanged when it has
    one, so presentation metadata (``object_name`` and friends) survives. When
    it has none, rebuilds a minimal ``StimulusSet`` from ``events`` that keeps
    only the input columns and ``stimulus_id``.
    """
    if getattr(session, "stimulus_set", None) is not None:
        return session.stimulus_set
    return reconstruct_stimulus_set_from_events(events)


def _spans_several_inputs(events: list[StreamEvent]) -> bool:
    families = set()
    for event in events:
        family, _ = parse_channel(event.channel)
        if family in _STIMULUS_INPUT_CHANNEL_FAMILIES:
            families.add(family)
    return len(families) > 1


def reconstruct_stimulus_set_from_events(events: list[StreamEvent]):
    """Rebuild a minimal StimulusSet from event-carried input columns.

    This fallback preserves only input-channel payload columns plus
    ``stimulus_id``. It is not metadata-faithful: metric-critical
    presentation metadata such as ``object_name`` must come from
    session.stimulus_set, or from a future event-carried metadata path.
    """
    if not events:
        raise ValueError(
            "neural participate received no input events and no session.stimulus_set "
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


def _collect_requested_channels(session):
    """Keep single-output collection simple without dropping other regions."""
    channels = session.requested_output_channels
    if len(channels) == 1:
        return session.collect(channel=channels[0])
    return {channel: session.collect(channel=channel) for channel in channels}


def _neural_channels(record) -> tuple:
    """Build the requested neural channel names for a ``record`` target.

    ``record`` accepts a single region name or a sequence of them, mirroring
    ``start_recording``. Formatting a sequence straight into the channel string
    would produce one channel literally named ``neural:['V4', 'IT']``, which then
    fails to match any region -- so the sequence case gets one channel each.
    """
    if isinstance(record, str):
        return (f"neural:{record}",)
    try:
        regions = list(record)
    except TypeError:
        raise TypeError(
            f"record must be a region name or a sequence of them; got "
            f"{type(record).__name__}")
    if not regions:
        raise ValueError("record must name at least one region")
    non_str = [r for r in regions if not isinstance(r, str)]
    if non_str:
        raise TypeError(f"every record entry must be a region name; got {non_str!r}")
    return tuple(f"neural:{region}" for region in regions)


def requested_output_channels(session) -> list[str]:
    requested = getattr(session, "requested_output_channels", None)
    if requested is not None:
        return list(requested)

    record = getattr(session, "record", None)
    if record is not None:
        return list(_neural_channels(record))

    raise ValueError(
        "streaming participate requires the session to declare "
        "requested_output_channels or record."
    )


def behavior_stimuli_from_session(session, task_context):
    events = [event for event in drain_session_inputs(session)
              if event.channel != 'instruction' and event.meta.get('role') != 'fitting']
    if events or getattr(session, 'stimulus_set', None) is not None:
        return session_stimuli(session, events)
    return _behavior_stimuli_to_score(task_context)


_drain_stream_events = drain_session_inputs


_stimuli_from_stream_session = session_stimuli


def _stimulus_input_columns(stimulus_set) -> list[tuple[str, str]]:
    """``(column, channel)`` for each input column of a stimulus table."""
    return [
        (column, _channel_for_stimulus_column(column))
        for column in list(getattr(stimulus_set, "columns", []))
        if _channel_for_stimulus_column(column) is not None
    ]


def _table_row_events(input_columns, position: int, row_index,
                      row) -> list[StreamEvent]:
    """The input events of one table row: one event per input column.

    A row with several input columns (``video_path`` and ``audio_path``) is
    one presentation that reaches the subject on several streams. Its events
    then carry ``meta['table_row']``, the row's position in the table, so a
    driver that answers one event at a time can present the row once (rule
    4.5). Rows of a one-column table are one event each and stay unchanged.
    """
    stimulus_id = _row_value(row, "stimulus_id", default=row_index)
    t_ms = float(_row_value(row, "t_ms", default=0.0))
    events = []
    for column, channel in input_columns:
        meta = {"stimulus_id": stimulus_id, "stimulus_index": row_index,
                "column": column}
        if len(input_columns) > 1:
            meta["table_row"] = position
        events.append(StreamEvent(channel=channel, payload=row[column],
                                  t_ms=t_ms, meta=meta))
    return events


def _stimulus_events(stimulus_set) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    input_columns = _stimulus_input_columns(stimulus_set)
    for position, (row_index, row) in enumerate(stimulus_set.iterrows()):
        events.extend(_table_row_events(input_columns, position, row_index, row))
    return events


def _is_null(value) -> bool:
    import pandas as pd
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _same_value(left, right) -> bool:
    """Equality for a present identity field; a null equals only a null."""
    if _is_null(left) or _is_null(right):
        return _is_null(left) and _is_null(right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _table_row_position(session, event: StreamEvent) -> Optional[int]:
    """The table position of a row-part event, or ``None`` for any other event.

    An event counts as one input column of row ``p`` of the session's own
    stimulus table only when all of these hold: ``meta['table_row']`` is the
    integer ``p`` inside the table; ``meta['stimulus_index']`` is present and
    equals the table's index label at ``p`` (a null label equals a null
    label); ``meta['stimulus_id']`` equals
    that row's stimulus id; and ``meta['column']`` is an input column of the
    table whose channel is the event's channel. Everything is read from the
    event and the table, never from settings a user can replace. Any other
    event, including every event of a base record, is presented alone.
    """
    meta = event.meta if isinstance(event.meta, dict) else {}
    position = meta.get("table_row")
    if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
        return None
    table = getattr(session, "stimulus_set", None)
    if table is None or not 0 <= int(position) < len(table):
        return None
    if "stimulus_index" not in meta or "stimulus_id" not in meta:
        return None
    column = meta.get("column")
    if (not isinstance(column, str) or column not in table.columns
            or _channel_for_stimulus_column(column) != event.channel):
        return None
    position = int(position)
    label = table.index[position]
    expected_id = _row_value(table.iloc[position], "stimulus_id", default=label)
    if not (_same_value(meta["stimulus_index"], label)
            and _same_value(meta["stimulus_id"], expected_id)):
        return None
    return position


def _table_row_stimuli(session, position: int):
    """A one-row slice of the session's table, selected by position."""
    table = session.stimulus_set
    return preserve_stimulus_set_type(table, table.iloc[[position]])


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
