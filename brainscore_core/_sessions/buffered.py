"""Buffered convenience sessions and typed measurement helpers."""

from __future__ import annotations
from typing import Sequence, Union
from ..streaming import BufferedSession
from ..events import StateChange
from ..supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from .payloads import _neural_channels, _state_change_event


class StimulusSetSession(BufferedSession):
    """Compatibility convenience; configure BufferedSession for new experiments."""

    def __init__(self, stimulus_set, record: Union[str, Sequence[str]] = "IT", time_bins=None):
        self.record = record
        super().__init__(stimulus_set=stimulus_set,
                         requested_output_channels=_neural_channels(record),
                         time_bins=time_bins)

    @classmethod
    def from_stimulus_set(cls, stimulus_set, record: Union[str, Sequence[str]] = "IT",
                          time_bins=None):
        return cls(stimulus_set, record=record, time_bins=time_bins)


class BehavioralSession(BufferedSession):
    """Compatibility convenience; configure BufferedSession(task_context=...)."""

    def __init__(self, task_context):
        super().__init__(task_context=task_context, requested_output_channels=('behavior',))
        self.scoring_stimuli = self.stimulus_set

    @classmethod
    def from_task_context(cls, task_context):
        return cls(task_context)

    def collect(self, channel: str = "behavior") -> BehavioralAssembly:
        if channel != "behavior":
            raise ValueError(f"BehavioralSession.collect only supports 'behavior'; got {channel!r}.")
        return super().collect(channel)


class StateChangeSession(BufferedSession):
    """Compatibility convenience for a single intervention event."""

    def __init__(self, state_change: StateChange):
        self.state_change = state_change
        super().__init__([_state_change_event(state_change)],
                         requested_output_channels=('perturbation',))

    @classmethod
    def from_state_change(cls, state_change: StateChange):
        return cls(state_change)

    def collect(self, channel: str = "perturbation"):
        matching = super().collect(channel)
        if len(matching) != 1:
            raise ValueError(f"Expected one emitted event for channel {channel!r}; got {len(matching)}.")
        event = matching[0]
        return event.payload if event.payload is not None else event.meta.get("handle_id")


def stimulus_session(stimulus_set, record: Union[str, Sequence[str]] = "IT",
                     time_bins=None) -> StimulusSetSession:
    return StimulusSetSession.from_stimulus_set(
        stimulus_set, record=record, time_bins=time_bins
    )


def behavior_session(task_context) -> BehavioralSession:
    return BehavioralSession.from_task_context(task_context)


def state_change_session(state_change: StateChange) -> StateChangeSession:
    return StateChangeSession.from_state_change(state_change)


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
    from .._evaluation.measures import note_measure
    session = behavior_session(task_context)
    subject.participate(session)
    behavior = session.collect("behavior")
    note_measure(behavior)
    return behavior


def apply_state_change(subject, state_change: StateChange):
    session = state_change_session(state_change)
    subject.participate(session)
    return session.collect("perturbation")
