"""One concrete Session composed from a source, a filter and a collector."""

import numpy as np
import pandas as pd
import pytest

from brainscore_core import io_catalog
from brainscore_core.events import PerturbationApplied
from brainscore_core.streaming import (
    CONTROL_CHANNEL, END_REASONS, FixedSource, Session, Source, StreamEvent,
    begin_event, end_event, respond_event,
)
from brainscore_core.streaming_helpers import (
    drive_neural_session, drive_neural_session_streaming, drive_state_change_session,
    stimulus_session, windowed_stream,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _neural(ids, offset=0.0):
    return NeuroidAssembly(
        np.arange(len(ids) * 2, dtype=float).reshape(len(ids), 2) + offset,
        coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['a', 'b'])},
        dims=['presentation', 'neuroid'])


def _table(n=3):
    table = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(n)],
                                      'sentence': [f'text {i}' for i in range(n)]}))
    table.identifier = 'composition'
    return table


# --- control channel -------------------------------------------------------------


def test_control_channel_is_registered_as_an_input_with_three_payloads():
    entry = io_catalog.get('control')
    assert entry.direction == io_catalog.INPUT
    assert io_catalog.validate(CONTROL_CHANNEL, 'begin', direction=io_catalog.INPUT) == []
    assert io_catalog.validate(CONTROL_CHANNEL, 'jump', direction=io_catalog.INPUT)


def test_control_constructors_and_end_reasons():
    assert (begin_event().payload, respond_event().payload) == ('begin', 'respond')
    for reason in END_REASONS:
        assert end_event(reason).meta == {'reason': reason}
    with pytest.raises(ValueError, match='end reason'):
        end_event('whim')


def test_a_bad_control_payload_fails_at_construction():
    with pytest.raises(ValueError, match='channel catalog'):
        Session([StreamEvent(CONTROL_CHANNEL, 'jump', 0.0)])


def test_batch_and_streaming_drivers_ignore_control_events():
    class Recorder:
        def __init__(self):
            self.sizes = []

        def start_recording(self, region, time_bins=None, recording_type=None):
            pass

        def process(self, stimuli, multi_modality=False):
            self.sizes.append(len(stimuli))
            return _neural(list(stimuli['stimulus_id']))

    for driver, streaming in ((drive_neural_session, False),
                              (drive_neural_session_streaming, True)):
        session = stimulus_session(_table(2), ['neural:IT'],
                                   streaming=streaming).declare_inputs({'text'})
        data = list(iter(session.next_input, None))
        session = stimulus_session(_table(2), ['neural:IT'], streaming=streaming)
        session.source = FixedSource([
            begin_event(0.0), data[0], respond_event(1.0), end_event('terminal', 2.0),
            begin_event(3.0), data[1], respond_event(4.0), end_event('terminal', 5.0)])
        recorder = Recorder()
        driver(recorder, session)
        assert sum(recorder.sizes) == 2
        assert session.collect('neural:IT').sizes['presentation'] == 2


# --- collect, by channel family ----------------------------------------------------


def test_neural_collect_returns_one_assembly_or_concatenates_along_presentation():
    session = Session((), ['neural:IT'])
    session.respond(StreamEvent('neural:IT', _neural(['a']), 0.0))
    assert session.collect('neural:IT').sizes['presentation'] == 1
    session.respond(StreamEvent('neural:IT', _neural(['b'], 10.0), 1.0))
    merged = session.collect()
    assert merged.sizes['presentation'] == 2


def test_neural_collect_packages_raw_vectors():
    session = Session((), ['neural:IT'])
    session.respond(StreamEvent('neural:IT', np.array([1.0, 2.0]), 0.0, {'stimulus_id': 'x'}))
    assembly = session.collect('neural:IT')
    assert assembly.dims == ('presentation', 'neuroid')
    assert list(assembly['stimulus_id'].values) == ['x']


def test_behavior_collect_one_hots_labels():
    session = Session((), ['behavior'])
    for stimulus, label in (('s0', 'cat'), ('s1', 'dog')):
        session.respond(StreamEvent('behavior', label, 0.0, {'stimulus_id': stimulus,
                                                             'label_set': ['cat', 'dog']}))
    assembly = session.collect('behavior')
    assert isinstance(assembly, BehavioralAssembly)
    assert assembly.values.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_perturbation_collect_returns_the_single_payload_or_the_handle():
    payload = PerturbationApplied(handle_id='h1', target=None, perturbation=None)
    session = Session((), ['perturbation'])
    session.respond(StreamEvent('perturbation', payload, 0.0, {'handle_id': 'h1'}))
    assert session.collect('perturbation') is payload
    only_handle = Session((), ['perturbation'])
    only_handle.respond(StreamEvent('perturbation', None, 0.0, {'handle_id': 'h2'}))
    assert only_handle.collect() == 'h2'
    twice = Session((), ['perturbation'])
    twice.respond(StreamEvent('perturbation', payload, 0.0))
    twice.respond(StreamEvent('perturbation', payload, 1.0))
    with pytest.raises(ValueError, match='Expected one emitted event'):
        twice.collect('perturbation')


def test_other_channels_return_events_and_no_channel_returns_a_dict_or_events():
    session = Session((), ['motor', 'neural:IT'])
    motor = StreamEvent('motor', [1], 0.0)
    session.respond(motor)
    session.respond(StreamEvent('neural:IT', _neural(['a']), 0.0))
    assert session.collect('motor') == [motor]
    assert set(session.collect()) == {'motor', 'neural:IT'}
    assert Session(()).collect() == []
    with pytest.raises(ValueError, match='No emitted events'):
        session.collect('behavior')


# --- sources ------------------------------------------------------------------------


def test_a_lazy_iterator_source_is_consumed_lazily():
    pulled = []

    def feed():
        for index in range(100):
            pulled.append(index)
            yield StreamEvent('text', f't{index}', float(index))

    session = Session(feed(), ['neural:IT'])
    assert pulled == []
    assert session.next_input().payload == 't0'
    assert pulled == [0]
    session.next_input()
    assert pulled == [0, 1]


def test_windowing_composes_with_the_one_session():
    stats = {}
    frames = (f'f{i}.png' for i in range(90))
    windows = windowed_stream(frames, fps=30, window_ms=1000, stats=stats)
    session = Session(windows, ['neural:IT'])

    class Recorder:
        def start_recording(self, region, time_bins=None, recording_type=None):
            pass

        def process(self, stimuli, multi_modality=False):
            return _neural(list(stimuli['stimulus_id']))

    drive_neural_session_streaming(Recorder(), session)
    assert stats['windows_emitted'] == 3
    assert session.collect().sizes['presentation'] == 90


def test_a_new_source_and_a_new_result_channel_need_no_new_class():
    """A generator is the source; a never-registered channel is collected as events."""
    def reading():
        yield StreamEvent('text', 'hello', 0.0)

    session = Session(reading(), ['pupil'])
    event = session.next_input()
    session.respond(StreamEvent('pupil', 3.5, event.t_ms))
    assert [e.payload for e in session.collect('pupil')] == [3.5]


def test_a_custom_source_receives_the_responses():
    class Echo(Source):
        def __init__(self):
            self.heard = []
            self.turn = 0

        def next_input(self):
            self.turn += 1
            return StreamEvent('text', f'turn {self.turn}', 0.0) if self.turn <= 2 else None

        def respond(self, event):
            self.heard.append(event.payload)

    source = Echo()
    session = Session(source, ['behavior'])
    while (event := session.next_input()) is not None:
        session.respond(StreamEvent('behavior', event.payload.upper(), 0.0))
    assert source.heard == ['TURN 1', 'TURN 2']


# --- observers and rewriting --------------------------------------------------------


def test_observers_see_filtered_inputs_and_rewritten_responses():
    seen = []
    session = Session([StreamEvent('text', 'a', 0.0), StreamEvent('audio', 'b', 0.0)],
                      ['behavior']).declare_inputs({'text'})
    rewrite = lambda event: StreamEvent(event.channel, event.payload, event.t_ms, {'tag': 1})
    with session.observe(on_input=lambda e: seen.append(('in', e.channel)),
                         on_output=lambda e: seen.append(('out', e.meta.get('tag')))):
        with session.rewriting_responses(rewrite):
            while session.next_input() is not None:
                pass
            session.respond(StreamEvent('behavior', 'x', 0.0))
    assert seen == [('in', 'text'), ('in', 'audio'), ('out', 1)]
    assert session.emitted[0].meta == {'tag': 1}
    session.respond(StreamEvent('behavior', 'y', 0.0))
    assert seen[-1] == ('out', 1)       # observers and rewriter are gone after the block


def test_state_change_session_driver_ignores_control_events():
    from brainscore_core.events import StateChange
    change = StateChange(kind='ablation')
    session = Session([begin_event(0.0), StreamEvent('lesion', change, 0.0, {'kind': 'ablation'})],
                      ['perturbation'])

    class Subject:
        def process(self, state_change):
            return PerturbationApplied(handle_id='h', target=None, perturbation=None)

    drive_state_change_session(Subject(), session)
    assert session.emitted and session.emitted[0].channel == 'perturbation'
