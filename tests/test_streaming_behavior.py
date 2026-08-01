"""Streaming behavioral trials: prepare once, then respond trial by trial.

The behavior driver was the last of the four that could only batch. This gives it the
same opt-in streaming path as perception.

The property that matters most here is NOT speed: it is that `start_task` runs exactly
once. A behavioral readout is fit during `start_task`; refitting it per trial would
silently change what is being measured, so the loop must not contain it.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    _drive_behavior_session_streaming,
    _drive_behavior_session_via_process,
)
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _stimuli(n=5):
    ss = StimulusSet(pd.DataFrame({
        'stimulus_id': [f'trial_{i}' for i in range(n)],
        'sentence': [f'word number {i}' for i in range(n)],
    }))
    ss.identifier = 'behavior-streaming-test'
    return ss


class _Session:
    """Minimal behavioral session; `streaming` decides which driver interact picks."""

    def __init__(self, task_context, streaming=False):
        self.task_context = task_context
        self.streaming = streaming
        self.requested_output_channels = ('behavior',)
        self.emitted = []

    def next_input(self):
        return None

    def emit(self, event):
        self.emitted.append(event)


class _Subject:
    def __init__(self):
        self.start_task_calls = 0
        self.process_sizes = []

    def start_task(self, task_context):
        self.start_task_calls += 1

    def process(self, stimuli, multi_modality=False):
        self.process_sizes.append(len(stimuli))
        ids = list(stimuli['stimulus_id'])
        return BehavioralAssembly(
            np.array([[float(len(str(i)))] for i in ids]),
            coords={'stimulus_id': ('presentation', ids),
                    'choice': ('choice', ['c0'])},
            dims=['presentation', 'choice'])


def _context(stimuli):
    return TaskContext(task_type='passive', metadata={'stimulus_set': stimuli})


@pytest.mark.unit
def test_streaming_behavior_runs_one_trial_at_a_time():
    stimuli = _stimuli(5)
    subject, session = _Subject(), _Session(_context(stimuli), streaming=True)
    _drive_behavior_session_streaming(subject, session)
    assert subject.process_sizes == [1, 1, 1, 1, 1]
    assert len(session.emitted) == 5


@pytest.mark.unit
def test_start_task_runs_exactly_once_not_per_trial():
    """Refitting the readout on every trial would change what is measured."""
    stimuli = _stimuli(8)
    subject, session = _Subject(), _Session(_context(stimuli), streaming=True)
    _drive_behavior_session_streaming(subject, session)
    assert subject.start_task_calls == 1, (
        f"start_task ran {subject.start_task_calls} times; the behavioral readout is "
        "fit there, so running it inside the trial loop refits per trial")


@pytest.mark.unit
def test_streaming_and_batch_behavior_agree():
    stimuli = _stimuli(6)

    batch_subject, batch_session = _Subject(), _Session(_context(stimuli))
    _drive_behavior_session_via_process(batch_subject, batch_session)
    batch_out = batch_session.emitted[0].payload

    stream_subject, stream_session = _Subject(), _Session(_context(stimuli), streaming=True)
    _drive_behavior_session_streaming(stream_subject, stream_session)
    import xarray as xr
    stream_out = xr.concat([e.payload for e in stream_session.emitted], dim='presentation')

    assert batch_subject.process_sizes == [6]        # batch really did it in one call
    np.testing.assert_allclose(np.asarray(batch_out), np.asarray(stream_out))


@pytest.mark.unit
def test_each_trial_is_tagged_with_its_index():
    stimuli = _stimuli(4)
    subject, session = _Subject(), _Session(_context(stimuli), streaming=True)
    _drive_behavior_session_streaming(subject, session)
    assert [e.meta['trial_index'] for e in session.emitted] == [0, 1, 2, 3]
    assert all(e.meta['streaming'] is True for e in session.emitted)
    assert all(e.channel == 'behavior' for e in session.emitted)


@pytest.mark.unit
def test_missing_task_context_fails_fast():
    class _NoContext(_Session):
        def __init__(self):
            self.streaming = True
            self.requested_output_channels = ('behavior',)
            self.emitted = []
            self.task_context = None

    with pytest.raises(ValueError, match="task_context"):
        _drive_behavior_session_streaming(_Subject(), _NoContext())
