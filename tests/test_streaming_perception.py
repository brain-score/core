"""Tier 1 streaming perception: consume a perception session one input at a time.

The motor path already streamed (`_drive_environment_session_via_process` pulls,
processes, emits, pulls again). The perception path drained the whole session up front
and made a single `process()` call, so "UMI streams" was only true for one of the four
drivers. These tests cover the perception twin.

The contract being tested: streaming changes WHEN inputs must exist, not WHAT comes out.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    StimulusSetSession,
    StreamingStimulusSetSession,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _stimulus_set(n=4):
    ss = StimulusSet(pd.DataFrame({
        'stimulus_id': [f'stim_{i}' for i in range(n)],
        'sentence': [f'a sentence number {i}' for i in range(n)],
    }))
    ss.identifier = 'streaming-test'
    return ss


class _CountingSubject:
    """Records how many times process() was called and with how many rows each time."""

    def __init__(self):
        self.call_sizes = []
        self.recorded_regions = []

    def start_recording(self, region, time_bins=None, recording_type=None):
        self.recorded_regions.append(region)

    def process(self, stimuli, multi_modality=False):
        n = len(stimuli)
        self.call_sizes.append(n)
        # deterministic per-row "activation" so batch and streaming are comparable
        ids = list(stimuli['stimulus_id'])
        data = np.array([[float(len(str(i))), 1.0] for i in ids])
        return NeuroidAssembly(
            data,
            coords={'stimulus_id': ('presentation', ids),
                    'neuroid_id': ('neuroid', ['n0', 'n1'])},
            dims=['presentation', 'neuroid'])


@pytest.mark.unit
def test_streaming_session_processes_one_stimulus_at_a_time():
    subject = _CountingSubject()
    session = StreamingStimulusSetSession(_stimulus_set(4), record='IT')
    from brainscore_core.streaming_helpers import _drive_neural_session_streaming
    _drive_neural_session_streaming(subject, session)
    # four stimuli -> four separate process() calls, each of exactly one row
    assert subject.call_sizes == [1, 1, 1, 1]


@pytest.mark.unit
def test_streaming_session_never_materializes_the_whole_stream():
    """The point of streaming: inputs need not all exist at once."""
    session = StreamingStimulusSetSession(_stimulus_set(50), record='IT')
    subject = _CountingSubject()
    from brainscore_core.streaming_helpers import _drive_neural_session_streaming
    _drive_neural_session_streaming(subject, session)
    # one row carries a single input column here, so at most one event is ever buffered
    assert session.max_events_held <= 1, (
        f"held {session.max_events_held} events at once; the stream was pre-drained "
        "and this is no longer a streaming session")


@pytest.mark.unit
def test_streaming_emits_one_output_event_per_input():
    subject = _CountingSubject()
    session = StreamingStimulusSetSession(_stimulus_set(3), record='IT')
    from brainscore_core.streaming_helpers import _drive_neural_session_streaming
    _drive_neural_session_streaming(subject, session)
    emitted = [e for e in session.emitted if e.channel == 'neural:IT']
    assert len(emitted) == 3
    assert [e.meta['stream_index'] for e in emitted] == [0, 1, 2]
    assert all(e.meta['streaming'] is True for e in emitted)


@pytest.mark.unit
def test_streaming_and_batch_agree():
    """Same stimuli, same subject, different arrival -> same values.

    This is the claim that makes Tier 1 worth having: the streaming path is an
    interface change, not a scientific one.
    """
    from brainscore_core.streaming_helpers import (
        _drive_neural_session_streaming, _drive_neural_session_via_process)

    stim = _stimulus_set(5)

    batch_subject = _CountingSubject()
    batch_session = StimulusSetSession(stim, record='IT')
    _drive_neural_session_via_process(batch_subject, batch_session)
    batch_out = [e.payload for e in batch_session.emitted if e.channel == 'neural:IT'][0]

    stream_subject = _CountingSubject()
    stream_session = StreamingStimulusSetSession(stim, record='IT')
    _drive_neural_session_streaming(stream_subject, stream_session)
    stream_out = stream_session.collect('neural:IT')

    assert batch_out.shape == stream_out.shape
    np.testing.assert_allclose(np.asarray(batch_out), np.asarray(stream_out))
    # and the batch path really did do it in one call, so the comparison is meaningful
    assert batch_subject.call_sizes == [5]


@pytest.mark.unit
def test_streaming_preserves_presentation_metadata():
    """Row slicing must keep the metadata metrics depend on, not just the payload."""
    stim = _stimulus_set(3)
    stim['object_name'] = ['a', 'b', 'c']
    seen = {}

    class _MetaSubject(_CountingSubject):
        def process(self, stimuli, multi_modality=False):
            seen.setdefault('columns', set()).update(stimuli.columns)
            return super().process(stimuli)

    session = StreamingStimulusSetSession(stim, record='IT')
    from brainscore_core.streaming_helpers import _drive_neural_session_streaming
    _drive_neural_session_streaming(_MetaSubject(), session)
    assert 'object_name' in seen['columns'], (
        "streamed rows lost presentation metadata; metrics keyed on it would break")
