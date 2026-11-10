"""Builders for the one ``Session`` and the typed measurement helpers."""

from __future__ import annotations
from typing import Sequence, Union
from ..streaming import Session
from ..events import StateChange
from ..supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from .payloads import (
    _behavior_events,
    _behavior_stimuli_to_score,
    _neural_channels,
    _state_change_event,
    _stimulus_events,
)


def table_session(stimulus_set, requested_output_channels=None, *, task_context=None,
                  time_bins=None, settings=None, record=None) -> Session:
    """One session that presents every row of ``stimulus_set``.

    The table stays on the session, so the batch drivers answer it with one
    ``process`` call; ``task_context``, ``time_bins`` and ``settings`` ride along.
    """
    return Session(_stimulus_events(stimulus_set), requested_output_channels, task_context,
                   stimulus_set=stimulus_set, time_bins=time_bins, settings=settings,
                   record=record)


def stimulus_session(stimulus_set, record: Union[str, Sequence[str]] = "IT",
                     time_bins=None) -> Session:
    return table_session(stimulus_set, _neural_channels(record), time_bins=time_bins,
                         record=record)


def behavior_session(task_context) -> Session:
    """A session whose events are the task's instruction, fitting and scoring stimuli."""
    scoring = _behavior_stimuli_to_score(task_context)
    return Session(_behavior_events(task_context, scoring), ('behavior',), task_context,
                   stimulus_set=scoring, scoring_stimuli=scoring)


def state_change_session(state_change: StateChange) -> Session:
    return Session([_state_change_event(state_change)], ('perturbation',),
                   state_change=state_change)


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
            "    session = stimulus_session(stimuli, record=['V4', 'IT'])\n"
            "    subject.participate(session)\n"
            "    v4, it = session.collect('neural:V4'), session.collect('neural:IT')\n"
            "Or record them in one pass and split on the region coord:\n"
            "    subject.start_recording(['V4', 'IT']); assembly = subject.process(stimuli)")
    session = stimulus_session(stimulus_set, record=record,
                               time_bins=time_bins)
    subject.participate(session)
    return session.collect(f"neural:{record}")


def behavioral_response(subject, task_context) -> BehavioralAssembly:
    """Run ``subject`` on the task and return its behavior as a
    ``BehavioralAssembly`` -- the model's output, not a Score.

    The stimuli to answer go in ``task_context.metadata['stimulus_set']``;
    ``fitting_stimuli`` is only the training set for a readout. Without that
    key the fitting stimuli are answered instead, so pass it for any held-out
    test set::

        task = TaskContext(task_type='probabilities', label_set=['real', 'pseudo'],
                           fitting_stimuli=train, metadata={'stimulus_set': test})
        behavior = behavioral_response(subject, task)
    """
    session = behavior_session(task_context)
    subject.participate(session)
    return session.collect("behavior")


def apply_state_change(subject, state_change: StateChange):
    session = state_change_session(state_change)
    subject.participate(session)
    return session.collect("perturbation")
