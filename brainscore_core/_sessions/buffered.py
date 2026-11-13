"""Builders for the one ``Session`` and the typed measurement helpers."""

from __future__ import annotations
from dataclasses import replace
from typing import Iterator, Sequence
from ..streaming import Session, StreamEvent, _with_trial_memory
from ..events import StateChange
from ..supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from .payloads import (
    _STATE_CHANGE_KIND_TO_CHANNEL_FAMILY,
    _state_change_event,
    _stimulus_events,
    _stimulus_input_columns,
    _table_row_events,
)


def _memory(task) -> str:
    return 'none' if task is None else task.trial_memory


def stimulus_session(stimuli, outputs=None, task=None, *,
                     time_bins=None, settings=None, streaming: bool = False) -> Session:
    """One session that presents every row of ``stimuli``.

    The table stays on the session, so the batch drivers answer it with one
    ``process`` call; ``task``, ``time_bins`` and ``settings`` ride along.
    The events mark the trials with ``begin`` / ``end`` as ``task.trial_memory``
    says (``none`` without a task).

    With ``streaming=True`` the events are made one row at a time from the
    table, and the session is marked ``streaming``: a subject answers one row
    and then gets the next. The requested outputs and the task are the same
    as for a batch session.
    """
    memory = _memory(task)
    if streaming:
        events = _with_trial_memory(_lazy_stimulus_events(stimuli), memory)
    else:
        events = list(_with_trial_memory(_stimulus_events(stimuli), memory))
    session = Session(events, outputs, task,
                      stimuli=stimuli, time_bins=time_bins, settings=settings,
                      streaming=streaming)
    session.one_answer_per_trial = memory == 'none'
    return session


def _lazy_stimulus_events(stimuli) -> Iterator[StreamEvent]:
    """The events of ``stimuli``, one row at a time.

    One event per input column; a several-column row marks its events with
    ``meta['table_row']`` so the row is presented once.
    """
    columns = _stimulus_input_columns(stimuli)
    for position, (row_index, row) in enumerate(stimuli.iterrows()):
        yield from _table_row_events(columns, position, row_index, row)


def _state_change_session(state_change: StateChange) -> Session:
    return Session([_state_change_event(state_change)], ('perturbation',))


def neural_response(subject, stimuli, outputs: Sequence[str],
                    time_bins=None) -> NeuroidAssembly:
    """Run ``subject`` on ``stimuli`` and return its neural response as a
    ``NeuroidAssembly``. This is the subject's output, not a Score -- comparing it
    to measured data (and ceiling-normalizing) is the benchmark's job, via a metric.

    ``outputs`` names one neural channel, as for ``stimulus_session``::

        assembly = neural_response(subject, stimuli, ['neural:IT'])
    """
    if isinstance(outputs, str):
        raise TypeError("outputs must be a sequence of channel names, for example "
                        f"['neural:IT']; got {outputs!r}.")
    outputs = list(outputs)
    if len(outputs) != 1:
        # This returns ONE assembly, so it cannot represent several regions without
        # silently picking one or inventing a container type. Say so instead.
        raise TypeError(
            "neural_response returns a single assembly and therefore takes one "
            f"channel; got {outputs!r}. To record several regions, use the session:\n"
            "    session = stimulus_session(stimuli, ['neural:V4', 'neural:IT'])\n"
            "    subject.participate(session)\n"
            "    v4, it = session.collect('neural:V4'), session.collect('neural:IT')")
    channel = outputs[0]
    if not channel.startswith('neural:'):
        raise ValueError(f"neural_response takes a neural channel such as 'neural:IT'; got {channel!r}.")
    session = stimulus_session(stimuli, [channel], time_bins=time_bins)
    subject.participate(session)
    return session.collect(channel)


def behavioral_response(subject, stimuli, task) -> BehavioralAssembly:
    """Run ``subject`` on ``stimuli`` under ``task`` and return its behavior as a
    ``BehavioralAssembly`` -- the subject's output, not a Score.

    ``stimuli`` are the stimuli to answer; ``task.fitting_stimuli`` is only the
    training set for a readout::

        task = TaskContext(task_type='probabilities', label_set=['real', 'pseudo'],
                           fitting_stimuli=train)
        behavior = behavioral_response(subject, test, task)

    The session presents only ``stimuli``. The instruction and the fitting
    stimuli are not sent as events; they stay on ``session.task``.
    """
    if stimuli is None:
        raise ValueError("behavioral_response needs the stimuli to answer.")
    session = stimulus_session(stimuli, ['behavior'], task)
    subject.participate(session)
    return session.collect("behavior")


def apply_state_change(subject, state_change: StateChange):
    if state_change.kind == "reset" and "family" not in state_change.metadata:
        # A reset that carries only a handle names no channel family. It goes on an
        # intervention channel the subject declares, so the subject receives it.
        declared = [family for family in dict.fromkeys(_STATE_CHANGE_KIND_TO_CHANNEL_FAMILY.values())
                    if family in subject.in_channels]
        if declared:
            state_change = replace(state_change, metadata={**state_change.metadata, "family": declared[0]})
    session = _state_change_session(state_change)
    subject.participate(session)
    return session.collect("perturbation")
