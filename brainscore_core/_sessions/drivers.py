"""Run typed subjects through the public session contract."""

from __future__ import annotations
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from ..contract import TaskContext
from ..streaming import StreamEvent, parse_channel
from ..events import StateChange
from .buffered import BehavioralSession, StateChangeSession, StimulusSetSession
from .environment import _drive_environment_session_via_process
from .payloads import (
    _behavior_stimuli_from_session,
    _one_row_stimuli,
    _preserve_stimulus_set_type,
    _requested_output_channels,
    _spans_several_inputs,
    _state_change_handle_id,
    drain_session_inputs,
    session_stimuli,
)

# "interact" and "interact_streaming" are stable record-format labels.
# Keep them unchanged when moving the session implementation.


def _drive_via_process(subject, session: StimulusSetSession, stimulus_set,
                       record: Union[str, Sequence[str]] = "IT", time_bins=None) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native participate()."""
    _start_recording(subject, record, time_bins=time_bins)
    output = subject.process(stimulus_set)
    session.emit(StreamEvent(
        channel=f"neural:{record}",
        payload=output,
        t_ms=0.0,
        meta={"driver": "process", "record": record},
    ))


def drive_neural_session(
    subject: Any, session: Any, driver: str = "interact",
    allow_multi_modality: bool = False,
    channel_runner: Optional[Callable] = None,
) -> None:
    """Answer a session's ``neural:<region>`` requests with one ``process`` call.

    For typed subjects such as legacy adapters and ``BrainScoreModel``, which
    already implement ``start_recording(target, time_bins=...)`` and
    ``process(stimuli)``. The contract does not require ``process``: an
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
    _refuse_variance(subject, session)
    input_events = drain_session_inputs(session)
    stimuli = session_stimuli(session, input_events)
    output_t_ms = input_events[-1].t_ms if input_events else 0.0
    use_multi_modality = allow_multi_modality and _spans_several_inputs(input_events)
    time_bins = getattr(session, "time_bins", None)

    channel_regions = []
    for channel in _requested_output_channels(session):
        family, region = parse_channel(channel)
        if family != "neural" or region is None:
            raise NotImplementedError(
                "neural participate currently supports requested "
                f"neural:<region> output channels; got {channel!r}."
            )
        channel_regions.append((channel, region))

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
        outputs = _demultiplex_neural_output(output, channel_regions)
    for channel, region, channel_output in outputs:
        session.emit(StreamEvent(
            channel=channel,
            payload=channel_output,
            t_ms=output_t_ms,
            meta={"driver": driver, "record": region},
        ))


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


def _demultiplex_neural_output(output, channel_regions):
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


def _drive_neural_session_streaming(
    subject, session, driver: str = "interact_streaming",
    channel_runner: Optional[Callable] = None,
) -> None:
    """Pull one session input event, process it, emit its outputs, and repeat.

    This is the perception-path twin of ``_drive_environment_session_via_process``.
    The batch driver drains the whole session up front and makes a single
    ``process()`` call, which is faster but requires every input to exist before
    scoring starts. This driver does not buffer ahead: the session decides whether
    an event contains one stimulus, a bounded window, or a real-time delivery.

    ``StreamingStimulusSetSession`` collapses batch size to one and may be slower;
    ``WindowedStreamSession`` restores bounded batches; ``RealTimeStreamSession``
    can additionally drop or reject late windows according to its policy.

    Built-in streaming session classes declare ``streaming = True``. A custom
    session can opt into this driver by declaring the same protocol flag.
    """
    channels = _requested_output_channels(session)
    channel_regions = []
    for channel in channels:
        family, region = parse_channel(channel)
        if family != "neural" or region is None:
            raise NotImplementedError(
                "streaming neural participate supports requested neural:<region> "
                f"output channels; got {channel!r}."
            )
        channel_regions.append((channel, region))

    time_bins = getattr(session, "time_bins", None)
    recording_target = _combined_recording_target(channel_regions)
    _start_recording(subject, recording_target, time_bins=time_bins)

    stream_index = 0
    event = session.next_input()
    while event is not None:
        if not isinstance(event, StreamEvent):
            raise TypeError(
                "streaming neural participate expects StreamEvent inputs; got "
                f"{type(event).__name__}."
            )
        stimuli = _one_row_stimuli(session, event, stream_index)
        if channel_runner is not None:
            payloads = channel_runner(stimuli, channel_regions, time_bins, False)
            outputs = [(channel, region, payloads[channel])
                       for channel, region in channel_regions]
        else:
            outputs = _demultiplex_neural_output(subject.process(stimuli), channel_regions)
        for channel, region, channel_output in outputs:
            session.emit(StreamEvent(
                channel=channel,
                payload=channel_output,
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


def drive_typed_session(subject, session) -> None:
    """Run a session through a typed model's ``process`` / ``start_*`` paths.

    Routes on the session's requested output channels: ``behavior``,
    ``perturbation``, ``motor``, or any set of ``neural:<region>`` channels.
    """
    requested_channels = _requested_output_channels(session)
    streaming = getattr(session, "streaming", False)
    if requested_channels == ["behavior"]:
        if streaming:
            _drive_behavior_session_streaming(subject, session)
        else:
            drive_behavior_session(subject, session)
        return
    if requested_channels == ["perturbation"]:
        _drive_state_change_session_via_process(subject, session)
        return
    if requested_channels == ["motor"]:
        _drive_environment_session_via_process(subject, session)
        return
    if requested_channels and all(channel.startswith("neural:")
                                  for channel in requested_channels):
        # A session may ask to be consumed one input at a time (inputs generated
        # on demand rather than known up front). Same scores, different arrival.
        if streaming:
            _drive_neural_session_streaming(subject, session)
        else:
            drive_neural_session(subject, session)
        return
    raise NotImplementedError(
        f"Subject {getattr(subject, 'identifier', type(subject).__name__)!r} "
        "runs sessions requesting neural:<region>, behavior, perturbation, or "
        f"motor output channels; got {requested_channels!r}."
    )


def _drive_state_change_via_process(subject, session: StateChangeSession,
                                    state_change: StateChange) -> None:
    """Interim Phase 2 bridge; Phase 3 swaps this for native participate()."""
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
                    "perturbation participate expects StreamEvent inputs; got "
                    f"{type(event).__name__}."
                )
            if not isinstance(event.payload, StateChange):
                raise TypeError(
                    "perturbation participate expects StateChange payloads; got "
                    f"{type(event.payload).__name__}."
                )
            state_change = event.payload
        else:
            state_change = getattr(session, "state_change", None)
    if state_change is None:
        raise ValueError(
            "perturbation participate requires a StateChange input event."
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
    """Interim Phase 2 bridge; Phase 3 swaps this for native participate()."""
    drive_behavior_session(
        subject, session, task_context=task_context, driver="process"
    )


def measurement_settings(session) -> dict:
    """Per-measurement settings a legacy facade attached to ``session`` (trial count, variance request)."""
    return dict(getattr(session, "settings", None) or {})


def _refuse_variance(subject, session) -> None:
    """The typed paths are deterministic: a request for trial-to-trial variance fails.

    A legacy ``look_at(require_variance=True)`` asks for repeated noisy
    presentations. Returning identical trials would be a silently corrected
    response (rule 6.2), so both the neural and the behavioral typed drivers
    refuse it before any compute.
    """
    settings = measurement_settings(session)
    if settings.get("require_variance"):
        raise NotImplementedError(
            f"Subject {getattr(subject, 'identifier', type(subject).__name__)!r} "
            "cannot produce trial-to-trial variance (look_at(require_variance=True, "
            f"number_of_trials={settings.get('number_of_trials', 1)})). Legacy models "
            "add it with microsaccades; this subject has no equivalent.")


def drive_behavior_session(
    subject: Any, session: Any, task_context: Optional[TaskContext] = None,
    driver: str = "interact",
) -> None:
    """Answer a session's ``behavior`` request with ``start_task`` then one ``process`` call.

    For typed subjects such as legacy adapters and ``BrainScoreModel``, which
    already implement ``start_task(task_context)`` and ``process(stimuli)``.
    The contract does not require ``process``: an arbitrary ``Subject`` only
    implements ``participate`` and never needs this.

    Uses ``task_context`` when given, otherwise the session's. Fitting inputs
    and instructions are not scored; the scored trials are the session's
    ``stimulus_set``, or the task's own stimuli when the session sends none.
    A request for trial-to-trial variance is refused before any compute. The
    single response is sent on the ``behavior`` channel.
    """
    if task_context is None:
        task_context = getattr(session, "task_context", None)
    if task_context is None:
        raise ValueError(
            "behavior participate requires a session.task_context to drive."
        )
    _refuse_variance(subject, session)
    subject.start_task(task_context)
    stimuli = _behavior_stimuli_from_session(session, task_context)
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


def drive_behavior_and_neural_session(model, session, driver: str = "interact") -> None:
    """Record ``neural:<region>`` while the subject does the session's task.

    ``behavior`` is sent too when requested.

    Each stimulus is shown once: the regions are recorded during the same
    forward pass that produced the answer (see
    ``BehavioralReadout.respond_and_record``). Only ``BrainScoreModel`` runs
    this; it is reached through its session capability.
    """
    task_context = getattr(session, "task_context", None)
    if task_context is None:
        raise ValueError("A behavior session needs session.task_context.")
    time_bins = getattr(session, "time_bins", None)
    if time_bins is not None and len(time_bins) > 1:
        raise NotImplementedError(
            "Recording during a task returns one time bin; request at most one.")
    requested = _requested_output_channels(session)
    regions = [parse_channel(c)[1] for c in requested if parse_channel(c)[0] == "neural"]
    answers = [c for c in requested if parse_channel(c)[0] == "behavior"]
    _refuse_variance(model, session)
    model.start_task(task_context)
    model._behavioral.history = []     # memory spans one session, not the next
    stimuli = _behavior_stimuli_from_session(session, task_context)
    if stimuli is None:
        raise ValueError("Recording during a task needs stimuli to present.")
    if task_context.response is None and set(answers) - {"behavior"}:
        raise ValueError(f"{answers} need TaskContext(response=...); task_type "
                         f"{task_context.task_type!r} answers on 'behavior'.")
    # A streaming session gets each trial's outputs as soon as that trial ends;
    # otherwise the whole set is answered and sent at once.
    streaming = getattr(session, "streaming", False)
    batches = ([(index, _preserve_stimulus_set_type(stimuli, stimuli.iloc[[index]]))
                for index in range(len(stimuli))] if streaming else [(0, stimuli)])
    for index, batch in batches:
        if task_context.response is not None:
            behavior, neural = model._behavioral.respond(batch, answers, regions)
        else:
            plain, neural = model._behavioral.respond_and_record(batch, regions)
            behavior = {"behavior": plain} if answers else {}
        for channel in answers:
            session.emit(StreamEvent(channel=channel, payload=behavior[channel], t_ms=float(index),
                                     meta={"driver": driver, "task_type": task_context.task_type}))
        for region in regions:
            session.emit(StreamEvent(channel=f"neural:{region}", payload=neural[region], t_ms=float(index),
                                     meta={"driver": driver, "record": region, "during": "behavior"}))


_drive_neural_session_via_process = drive_neural_session


_drive_behavior_session_via_process = drive_behavior_session


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
            "behavior participate requires a session.task_context to drive."
        )
    subject.start_task(task_context)              # once, before any trial
    stimuli = _behavior_stimuli_from_session(session, task_context)
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
