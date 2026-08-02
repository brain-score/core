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


class _MultiRegionSubject:
    """Returns every active region in one assembly, as list recording requires."""

    def __init__(self):
        self.recording_calls = []
        self.active_regions = []
        self.process_calls = 0

    def start_recording(self, recording_target, time_bins=None,
                        recording_type=None):
        del time_bins, recording_type
        self.recording_calls.append(recording_target)
        self.active_regions = (
            [recording_target]
            if isinstance(recording_target, str)
            else list(recording_target)
        )

    def process(self, stimuli, multi_modality=False):
        del multi_modality
        self.process_calls += 1
        ids = list(stimuli['stimulus_id'])
        regions = list(self.active_regions)
        data = np.tile(np.arange(1, len(regions) + 1, dtype=float),
                       (len(ids), 1))
        return NeuroidAssembly(
            data,
            coords={
                'stimulus_id': ('presentation', ids),
                'neuroid_id': ('neuroid', [f'n{i}' for i in range(len(regions))]),
                'layer': ('neuroid', [f'{region}_layer' for region in regions]),
                'region': ('neuroid', regions),
            },
            dims=['presentation', 'neuroid'],
        )


def _request_two_regions(session):
    session.requested_output_channels = ('neural:early', 'neural:late')
    return session


@pytest.mark.unit
def test_batch_multi_region_records_once_and_demultiplexes_channels():
    from brainscore_core.streaming_helpers import _drive_neural_session_via_process

    subject = _MultiRegionSubject()
    session = _request_two_regions(StimulusSetSession(_stimulus_set(3),
                                                      record='early'))

    _drive_neural_session_via_process(subject, session)

    assert subject.recording_calls == [['early', 'late']]
    assert subject.process_calls == 1
    for channel, region in (('neural:early', 'early'),
                            ('neural:late', 'late')):
        payloads = [event.payload for event in session.emitted
                    if event.channel == channel]
        assert len(payloads) == 1
        assert payloads[0].sizes['neuroid'] == 1
        assert payloads[0]['region'].values.tolist() == [region]


@pytest.mark.unit
def test_streaming_multi_region_records_once_and_demultiplexes_each_window():
    from brainscore_core.streaming_helpers import _drive_neural_session_streaming

    subject = _MultiRegionSubject()
    session = _request_two_regions(StreamingStimulusSetSession(
        _stimulus_set(3), record='early'))

    _drive_neural_session_streaming(subject, session)

    assert subject.recording_calls == [['early', 'late']]
    assert subject.process_calls == 3
    for channel, region in (('neural:early', 'early'),
                            ('neural:late', 'late')):
        payloads = [event.payload for event in session.emitted
                    if event.channel == channel]
        assert len(payloads) == 3
        assert all(payload.sizes['neuroid'] == 1 for payload in payloads)
        assert all(payload['region'].values.tolist() == [region]
                   for payload in payloads)


@pytest.mark.unit
def test_streaming_multi_region_rejects_missing_layer_provenance():
    class _MissingLayerSubject(_MultiRegionSubject):
        def process(self, stimuli, multi_modality=False):
            output = super().process(stimuli, multi_modality=multi_modality)
            import xarray as xr
            return xr.DataArray(
                np.asarray(output),
                dims=('presentation', 'neuroid'),
                coords={'region': ('neuroid', list(self.active_regions))},
            )

    subject = _MissingLayerSubject()
    session = _request_two_regions(StimulusSetSession(_stimulus_set(1),
                                                      record='early'))

    from brainscore_core.streaming_helpers import _drive_neural_session_via_process
    with pytest.raises(ValueError, match="per-neuroid 'layer' coordinate"):
        _drive_neural_session_via_process(subject, session)


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
