"""Builders for the one ``Session`` and the typed measurement helpers."""

from __future__ import annotations
from dataclasses import replace
from typing import Iterator, Sequence, Union
from ..streaming import Session, StreamEvent, _with_trial_memory
from ..events import StateChange
from ..supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from .payloads import (
    _behavior_stimuli_to_score,
    _STATE_CHANGE_KIND_TO_CHANNEL_FAMILY,
    _state_change_event,
    _stimulus_events,
    _stimulus_input_columns,
    _table_row_events,
)


def _memory(task) -> str:
    return 'none' if task is None else task.trial_memory


def stimulus_session(stimulus_set, outputs=None, task=None, *,
                     time_bins=None, settings=None, streaming: bool = False) -> Session:
    """One session that presents every row of ``stimulus_set``.

    The table stays on the session, so the batch drivers answer it with one
    ``process`` call; ``task``, ``time_bins`` and ``settings`` ride along.
    The events mark the trials with ``begin`` / ``end`` as ``task.trial_memory``
    says (``none`` without a task context).

    With ``streaming=True`` the events are made one row at a time from the
    table, and the session is marked ``streaming``: a subject answers one row
    and then gets the next. The requested outputs and the task are the same
    as for a batch session.
    """
    memory = _memory(task)
    if streaming:
        events = _with_trial_memory(_lazy_stimulus_events(stimulus_set), memory)
    else:
        events = list(_with_trial_memory(_stimulus_events(stimulus_set), memory))
    session = Session(events, outputs, task,
                      stimulus_set=stimulus_set, time_bins=time_bins, settings=settings,
                      streaming=streaming)
    session.one_answer_per_trial = memory == 'none'
    return session


def _lazy_stimulus_events(stimulus_set) -> Iterator[StreamEvent]:
    """The events of ``stimulus_set``, one row at a time.

    One event per input column; a several-column row marks its events with
    ``meta['table_row']`` so the row is presented once.
    """
    columns = _stimulus_input_columns(stimulus_set)
    for position, (row_index, row) in enumerate(stimulus_set.iterrows()):
        yield from _table_row_events(columns, position, row_index, row)


def _state_change_session(state_change: StateChange) -> Session:
    return Session([_state_change_event(state_change)], ('perturbation',))


def neural_response(subject, stimulus_set, record: Union[str, Sequence[str]] = "IT",
                  time_bins=None) -> NeuroidAssembly:
    """Run ``subject`` on ``stimulus_set`` and return its neural response as a
    ``NeuroidAssembly``. This is the model's output, not a Score -- comparing it
    to measured data (and ceiling-normalizing) is the benchmark's job, via a metric.
    """
    if not isinstance(record, str):
        # This returns ONE assembly, so it cannot represent several regions without
        # silently picking one or inventing a container type. Say so instead.
        raise TypeError(
            "neural_response returns a single assembly and therefore takes a single "
            f"region; got {record!r}. To record several regions, use the session API:\n"
            "    session = stimulus_session(stimuli, ['neural:V4', 'neural:IT'])\n"
            "    subject.participate(session)\n"
            "    v4, it = session.collect('neural:V4'), session.collect('neural:IT')\n"
            "Or record them in one pass and split on the region coord:\n"
            "    subject.start_recording(['V4', 'IT']); assembly = subject.process(stimuli)")
    session = stimulus_session(stimulus_set, [f"neural:{record}"], time_bins=time_bins)
    subject.participate(session)
    return session.collect(f"neural:{record}")


def behavioral_response(subject, task) -> BehavioralAssembly:
    """Run ``subject`` on the task and return its behavior as a
    ``BehavioralAssembly`` -- the model's output, not a Score.

    The stimuli to answer go in ``task.metadata['stimulus_set']``;
    ``fitting_stimuli`` is only the training set for a readout. Without that
    key the fitting stimuli are answered instead, so pass it for any held-out
    test set::

        task = TaskContext(task_type='probabilities', label_set=['real', 'pseudo'],
                           fitting_stimuli=train, metadata={'stimulus_set': test})
        behavior = behavioral_response(subject, task)

    The session presents only the stimuli to answer. The instruction and the
    fitting stimuli are not sent as events; they stay on ``session.task``.
    """
    stimuli = _behavior_stimuli_to_score(task)
    if stimuli is None:
        raise ValueError(
            "behavioral_response requires stimuli to elicit a response. Provide "
            "task.metadata['stimulus_set'] or fitting_stimuli.")
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
