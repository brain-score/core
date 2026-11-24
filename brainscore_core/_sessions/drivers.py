"""Run typed subjects through the public session contract."""

from __future__ import annotations
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from ..contract import TaskContext
from ..streaming import BEGIN, CONTROL_CHANNEL, Session, StreamEvent, parse_channel
from ..events import StateChange
from ..compatibility import ExecutionNotFeasibleError
from .environment import EnvironmentSource, drive_environment_session
from .payloads import (
    behavior_stimuli_from_session,
    _one_row_stimuli,
    preserve_stimulus_set_type,
    session_outputs,
    _spans_several_inputs,
    _state_change_handle_id,
    _stimulus_input_columns,
    _table_row_position,
    _table_row_stimuli,
    drain_session_inputs,
    session_stimuli,
)

# "interact" and "interact_streaming" are stable record-format labels.
# Keep them unchanged when moving the session implementation.


def drive_neural_session(
    subject: Any, session: Any, driver: str = "interact",
    allow_multi_modality: bool = False,
    channel_runner: Optional[Callable] = None,
) -> None:
    """Answer a session's ``neural:<region>`` requests with one ``process`` call.

    For typed subjects such as legacy adapters, which already implement
    ``start_recording(target, time_bins=...)`` and ``process(stimuli)``. The contract does not require ``process``: an
    arbitrary ``Subject`` only implements ``participate`` and never needs this.

    Reads every input from ``session``, records all requested regions in one
    pass using the session's ``time_bins``, and responds once per requested
    channel. A request for trial-to-trial variance in the session settings is
    refused before any compute. ``allow_multi_modality`` is the subject's own
    statement that its ``process`` combines several input modalities; when
    several are present it is called with ``multi_modality=True``.

    ``channel_runner(stimuli, channel_regions, time_bins, multi_modality)``
    returns ``{channel: payload}``. A subject that records regions with
    separate extractors supplies one, so each channel keeps the payload it
    would have alone. Without it, all regions are recorded in one call.
    """
    refuse_variance(subject, session)
    channel_regions = _neural_channel_regions(session, "neural participate")
    input_events = drain_session_inputs(session)
    stimuli = session_stimuli(session, input_events)
    use_multi_modality = allow_multi_modality and _spans_several_inputs(input_events)
    time_bins = getattr(session, "time_bins", None)

    if channel_runner is not None:
        payloads = channel_runner(stimuli, channel_regions, time_bins, use_multi_modality)
        outputs = [(channel, region, payloads[channel])
                   for channel, region in channel_regions]
    else:
        recording_target = _combined_recording_target(channel_regions)
        _start_recording(subject, recording_target, time_bins=time_bins)
        if use_multi_modality:
            output = subject.process(stimuli, multi_modality=True)
        else:
            output = subject.process(stimuli)
        outputs = demultiplex_neural_output(output, channel_regions)
    for channel, region, channel_output in outputs:
        session.respond(StreamEvent(
            channel=channel,
            payload=channel_output,
            t_ms=None,   # one answer for the whole table has no time (rule 4.2)
            meta={"driver": driver, "record": region},
        ))


def _neural_channel_regions(session, driver: str) -> List[Tuple[str, str]]:
    """Return ``(channel, region)`` for each requested output, or refuse the request.

    Reads only the requested outputs, never the source: an impossible request is
    refused before any input is read or computed (rule 5.14).
    """
    channels = session_outputs(session)
    parsed = [(channel, *parse_channel(channel)) for channel in channels]
    other = [channel for channel, family, region in parsed
             if family != "neural" or region is None]
    if other:
        if len(other) < len(channels):
            remedy = "Request the neural and the other outputs in separate sessions."
        else:
            remedy = ("Run this session with a driver that answers those outputs "
                      "(drive_typed_session routes each combination), or request "
                      "neural:<region>.")
        raise ExecutionNotFeasibleError(
            f"{driver} answers only neural:<region> output channels; got {other!r}. {remedy}",
            channels=other)
    return [(channel, region) for channel, _, region in parsed]


def _combined_recording_target(
    channel_regions: List[Tuple[str, str]]
) -> Union[str, List[str]]:
    regions = [region for _, region in channel_regions]
    return regions[0] if len(regions) == 1 else regions


def _start_recording(
    subject, recording_target: Union[str, List[str]], time_bins=None
) -> None:
    if time_bins is None:
        subject.start_recording(recording_target)
    else:
        subject.start_recording(recording_target, time_bins=time_bins)


def demultiplex_neural_output(output, channel_regions):
    """Return one region-specific payload per requested neural channel.

    A single requested channel keeps the historical payload object unchanged.
    Multiple channels require explicit per-neuroid ``region`` provenance; an
    unattributed combined assembly is rejected instead of being silently assigned
    according to concatenation order.
    """
    if len(channel_regions) == 1:
        channel, region = channel_regions[0]
        return [(channel, region, output)]

    if not hasattr(output, 'dims') or 'neuroid' not in output.dims:
        raise ValueError(
            "Multiple neural output channels require an assembly with a "
            "'neuroid' dimension and per-neuroid provenance so outputs can be "
            "demultiplexed without relying on concatenation order."
        )
    try:
        layer_coord = output['layer']
    except KeyError as error:
        raise ValueError(
            "Multiple neural output channels require an assembly with a "
            "per-neuroid 'layer' coordinate; attribution by concatenation "
            "order is not allowed."
        ) from error
    if layer_coord.dims != ('neuroid',):
        raise ValueError(
            "The neural output 'layer' coordinate must be one-dimensional "
            "along 'neuroid' to preserve output provenance; got dims "
            f"{layer_coord.dims}."
        )
    try:
        region_coord = output['region']
    except KeyError as error:
        raise ValueError(
            "Multiple neural output channels require an assembly with a "
            "per-neuroid 'region' coordinate so outputs can be demultiplexed "
            "without relying on concatenation order."
        ) from error
    if region_coord.dims != ('neuroid',):
        raise ValueError(
            "The neural output 'region' coordinate must be one-dimensional "
            "along 'neuroid' to demultiplex requested channels; got dims "
            f"{region_coord.dims}."
        )

    region_values = np.asarray(region_coord.values)
    outputs = []
    for channel, region in channel_regions:
        positions = np.flatnonzero([
            region in str(value).split('|') for value in region_values
        ])
        if not len(positions):
            raise ValueError(
                f"Neural output contains no neuroids attributed to requested "
                f"region {region!r}; available region labels are "
                f"{sorted(set(str(value) for value in region_values))}."
            )
        outputs.append((channel, region, output.isel(neuroid=positions)))
    return outputs


def drive_neural_session_streaming(
    subject, session, driver: str = "interact_streaming",
    channel_runner: Optional[Callable] = None,
    on_begin: Optional[Callable[[], None]] = None,
) -> None:
    """Pull one session input event, process it, emit its outputs, and repeat.

    This is the perception-path twin of ``drive_environment_session``.
    The batch driver drains the whole session up front and makes a single
    ``process()`` call, which is faster but requires every input to exist before
    scoring starts. This driver does not buffer ahead: the session decides whether
    an event contains one stimulus, or a bounded window.

    ``stimulus_session(..., streaming=True)`` collapses batch size to one and may be slower;
    a source whose events each carry a bounded window keeps bounded batches.

    A session opts into this driver with ``streaming=True``.

    A table row with several input columns arrives as one event per column.
    Those events carry ``meta['table_row']``; consecutive events of one row
    are one presentation, so the row is presented once (rule 4.5) when its
    last input column arrives. Every other event, and every event whose
    marker does not match the session's own table, is presented alone, as
    before. No event is dropped.

    ``on_begin()`` runs at each ``begin`` control event, after the row before it
    was presented.
    """
    refuse_variance(subject, session)
    channel_regions = _neural_channel_regions(session, "streaming neural participate")

    if on_begin is not None:
        session._request_control('An episode reset')
    time_bins = getattr(session, "time_bins", None)
    if channel_runner is None:     # a runner sets its own recording for each batch
        _start_recording(subject, _combined_recording_target(channel_regions), time_bins=time_bins)

    stream_index = 0

    def present(stimuli, t_ms) -> None:
        nonlocal stream_index
        if channel_runner is not None:
            payloads = channel_runner(stimuli, channel_regions, time_bins, False)
            outputs = [(channel, region, payloads[channel])
                       for channel, region in channel_regions]
        else:
            outputs = demultiplex_neural_output(subject.process(stimuli), channel_regions)
        for channel, region, channel_output in outputs:
            session.respond(StreamEvent(
                channel=channel,
                payload=channel_output,
                t_ms=t_ms,
                meta={
                    "driver": driver,
                    "record": region,
                    "stream_index": stream_index,
                    "streaming": True,
                },
            ))
        stream_index += 1

    for stimuli, t_ms in _stream_rows(session, on_begin):
        present(stimuli, t_ms)


def _stream_rows(session, on_begin: Optional[Callable[[], None]] = None,
                 skip_channels: Sequence[str] = ()):
    """Read the session's inputs one event at a time; yield ``(stimuli, t_ms)`` per presentation.

    A row is yielded as soon as its last input column has arrived, before the next
    event is read, so a caller answers it first. A table row with several input
    columns arrives as one event per column (``meta['table_row']``); they are one
    presentation (rule 4.5). Every other event is presented alone. ``begin``
    control events call ``on_begin()`` after the row before them was answered;
    other control events carry no stimulus. Events on ``skip_channels`` are read
    and dropped.
    """
    # The input columns a row delivers through this session. A view that
    # drops an undeclared stream drops its table column too.
    table = getattr(session, "stimuli", None)
    row_columns = ({column for column, _ in _stimulus_input_columns(table)}
                   if table is not None else set())
    row_events: list[StreamEvent] = []   # arrived input columns of one row
    row_position = None
    stream_index = 0
    event = session.next_input()
    while True:
        # Flush the open row when the next event cannot belong to it.
        end_of_row = event is None or not isinstance(event, StreamEvent) or event.channel == CONTROL_CHANNEL
        position = None if end_of_row else _table_row_position(session, event)
        if row_events and (
                end_of_row or position != row_position
                or event.t_ms != row_events[0].t_ms
                or event.meta["column"] in {e.meta["column"] for e in row_events}):
            yield _table_row_stimuli(session, row_position), row_events[0].t_ms
            row_events = []
        if event is None:
            return
        if not isinstance(event, StreamEvent):
            raise TypeError(
                "streaming participate expects StreamEvent inputs; got "
                f"{type(event).__name__}."
            )
        if event.channel == CONTROL_CHANNEL:
            if on_begin is not None and event.payload == BEGIN:
                on_begin()
        elif event.channel in skip_channels:
            pass
        elif position is None:
            yield _one_row_stimuli(session, event, stream_index), event.t_ms
            stream_index += 1
        else:
            row_events.append(event)
            row_position = position
            if row_columns <= {e.meta["column"] for e in row_events}:
                yield _table_row_stimuli(session, row_position), row_events[0].t_ms
                row_events = []
        event = session.next_input()


def stream_trials(session, task, on_begin: Optional[Callable[[], None]] = None):
    """``(stimuli, t_ms)`` per trial: a single-row stimulus set and the time of its input.

    The next trial is read only after the caller has answered this one. Each ``begin``
    control event calls ``on_begin()`` after the trial before it was answered. ``t_ms`` is
    the time of the trial's input events, or None when they have none. A session
    that sends no input events presents its table, or the task's own stimuli,
    with no time.

    Limit: several ``begin`` events with no trial between them call ``on_begin()``
    once for each, and a caller that resets at the next answer resets once.
    Limit: a row of the session's table that no event presents is not given (a custom
    partial table); present every row that must be answered.
    """
    count = 0
    for stimuli, t_ms in _stream_rows(session, on_begin, skip_channels=("instruction",)):
        count += 1
        yield stimuli, t_ms
    if not count:
        stimuli = behavior_stimuli_from_session(session, task)
        if stimuli is None:
            raise ValueError(
                "A behavior session requires stimuli to elicit a response. Build it with "
                "stimulus_session(stimuli, ['behavior'], task), or set task.fitting_stimuli.")
        for index in range(len(stimuli)):
            yield preserve_stimulus_set_type(stimuli, stimuli.iloc[[index]]), None


def drive_typed_session(subject, session) -> None:
    """Run a session through a typed model's ``process`` / ``start_*`` paths.

    Routes on the session's requested output channels: ``behavior``,
    ``perturbation``, any set of ``neural:<region>`` channels, or any outputs of an
    environment source (a closed loop, ``process(row) -> outputs``).
    """
    requested_channels = session_outputs(session)
    streaming = getattr(session, "streaming", False)
    if requested_channels == ["behavior"]:
        if streaming:
            drive_behavior_session_streaming(subject, session)
        else:
            drive_behavior_session(subject, session)
        return
    if requested_channels == ["perturbation"]:
        drive_state_change_session(subject, session)
        return
    if isinstance(session.source, EnvironmentSource):
        drive_environment_session(subject, session)
        return
    if requested_channels and all(channel.startswith("neural:")
                                  for channel in requested_channels):
        # A session may ask to be consumed one input at a time (inputs generated
        # on demand rather than known up front). Same scores, different arrival.
        if streaming:
            drive_neural_session_streaming(subject, session)
        else:
            drive_neural_session(subject, session)
        return
    raise ExecutionNotFeasibleError(
        f"Subject {getattr(subject, 'identifier', type(subject).__name__)!r} "
        "runs sessions requesting neural:<region>, behavior, perturbation, or "
        f"motor output channels; got {requested_channels!r}. Request one of those "
        "combinations in each session.", channels=requested_channels)


def drive_state_change_session(
    subject, session: Session, state_change: StateChange = None,
    driver: str = "interact"
) -> None:
    """Drive a perturbation session through existing StateChange dispatch.

    The answer carries the time of the change's input event, or None when the
    change is given as an argument (rule 4.2).
    """
    t_ms = None
    if state_change is None:
        event = session.next_input()
        while event is not None and event.channel == CONTROL_CHANNEL:
            event = session.next_input()
        if event is not None:
            if not isinstance(event, StreamEvent):
                raise TypeError(
                    "perturbation participate expects StreamEvent inputs; got "
                    f"{type(event).__name__}."
                )
            if not isinstance(event.payload, StateChange):
                raise TypeError(
                    "perturbation participate expects StateChange payloads; got "
                    f"{type(event.payload).__name__}."
                )
            state_change = event.payload
            t_ms = event.t_ms
    if state_change is None:
        raise ValueError(
            "perturbation participate requires a StateChange input event. A subject "
            "receives one only on the family channel of the change: add 'lesion', "
            "'stimulation' or 'pharmacological' to its in_channels. A reset goes on "
            "the family its change was applied on."
        )
    result = subject.process(state_change)
    handle_id = _state_change_handle_id(state_change, result)
    session.respond(StreamEvent(
        channel="perturbation",
        payload=result,
        t_ms=t_ms,
        meta={
            "driver": driver,
            "kind": state_change.kind,
            "handle_id": handle_id,
        },
    ))


def measurement_settings(session) -> dict:
    """Per-measurement settings a legacy facade attached to ``session`` (trial count, variance request)."""
    return dict(getattr(session, "settings", None) or {})


def refuse_variance(subject, session) -> None:
    """The typed paths are deterministic: a request for trial-to-trial variance fails.

    A legacy ``look_at(require_variance=True)`` asks for repeated noisy
    presentations. Returning identical trials would be a silently corrected
    response (rule 6.2), so both the neural and the behavioral typed drivers
    refuse it before any compute.
    """
    settings = measurement_settings(session)
    if settings.get("require_variance"):
        raise ExecutionNotFeasibleError(
            f"Subject {getattr(subject, 'identifier', type(subject).__name__)!r} "
            "cannot produce trial-to-trial variance (look_at(require_variance=True, "
            f"number_of_trials={settings.get('number_of_trials', 1)})). Legacy models "
            "add it with microsaccades; this subject has no equivalent.",
            task_type=getattr(getattr(session, 'task', None), 'task_type', None))


def drive_behavior_session(
    subject: Any, session: Any, task: Optional[TaskContext] = None,
    driver: str = "interact", on_begin: Optional[Callable[[], None]] = None,
) -> None:
    """Answer a session's ``behavior`` request with ``start_task`` then one ``process`` call.

    For typed subjects such as legacy adapters, which already implement
    ``start_task(task)`` and ``process(stimuli)``.
    The contract does not require ``process``: an arbitrary ``Subject`` only
    implements ``participate`` and never needs this.

    Uses ``task`` when given, otherwise the session's. The scored
    trials are the session's ``stimuli``, or the task's own stimuli when
    the session sends none. An ``instruction`` event is not scored.
    A request for trial-to-trial variance is refused before any compute. The
    single response is sent on the ``behavior`` channel. ``on_begin()`` runs once
    before the trials: this path keeps no history between them.
    """
    if task is None:
        task = getattr(session, "task", None)
    if task is None:
        raise ValueError(
            "behavior participate requires a session.task to drive."
        )
    refuse_variance(subject, session)
    subject.start_task(task)
    if on_begin is not None:
        on_begin()
    stimuli = behavior_stimuli_from_session(session, task)
    if stimuli is None:
        raise ValueError(
            "A behavior session requires stimuli to elicit a response. Build it with "
            "stimulus_session(stimuli, ['behavior'], task), or set task.fitting_stimuli."
        )
    output = subject.process(stimuli)
    session.respond(StreamEvent(
        channel="behavior",
        payload=output,
        t_ms=None,          # one answer for every trial: no single time (rule 4.2)
        meta={"driver": driver, "task_type": task.task_type},
    ))




def drive_behavior_session_streaming(
    subject, session: Session, task=None,
    driver: str = "interact_streaming",
    on_begin: Optional[Callable[[], None]] = None,
) -> None:
    """Behavioral trials delivered one at a time.

    Mirrors a real experiment: the subject is prepared ONCE (``start_task`` fits any
    readout on the fitting stimuli), then trials arrive one by one and a response
    comes back per trial. Re-preparing per trial would both refit the readout on every
    trial -- wrong -- and be pointlessly slow, so ``start_task`` is deliberately
    outside the loop.

    Same responses as the batch driver; what changes is that the trial set need not
    exist in full before the first response is produced. ``on_begin()`` runs at each
    ``begin`` control event, after the trial before it was answered.
    """
    if task is None:
        task = getattr(session, "task", None)
    if task is None:
        raise ValueError(
            "behavior participate requires a session.task to drive."
        )
    refuse_variance(subject, session)
    if on_begin is not None:
        session._request_control('An episode reset')
    subject.start_task(task)              # once, before any trial
    for trial_index, (trial, t_ms) in enumerate(stream_trials(session, task, on_begin)):
        output = subject.process(trial)
        session.respond(StreamEvent(
            channel="behavior",
            payload=output,
            t_ms=t_ms,
            meta={"driver": driver, "task_type": task.task_type,
                  "trial_index": trial_index, "streaming": True},
        ))
