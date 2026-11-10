"""Fidelity tests for the legacy-benchmark facade (Contract Rules v3, rule 11.2).

Each legacy method a vision or language benchmark calls on a candidate is
checked against the legacy behavior it stands in for.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.legacy_facade import (
    LegacyBenchmarkFacade,
    text_stimulus_set,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _images(n=2):
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_path': [f'/tmp/s{i}.png' for i in range(n)],
    }))
    stimuli.identifier = f'images-{n}'
    return stimuli


def test_text_stimulus_set_keys_cache_by_content():
    first = text_stimulus_set(['a passage'])
    second = text_stimulus_set(['another passage'])
    assert first.identifier != second.identifier
    assert list(first['context_id']) == ['passage']
    stimuli = _images()
    assert text_stimulus_set(stimuli) is stimuli


# --- the facade runs any subject through participate ------------------------

from brainscore_core.model_interface import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly


class _Recorder(Subject):
    """A participate-only subject: no process, look_at or start_recording."""

    identifier = 'recording-subject'
    in_channels = {'text', 'vision'}
    out_channels = {'neural:IT', 'behavior'}
    required_channels = set()

    def __init__(self):
        self.sessions = []
        self.resets = 0

    def participate(self, session):
        self.sessions.append(session)
        while session.next_input() is not None:
            pass
        n = len(session.stimulus_set)
        for channel in session.requested_output_channels:
            neural = channel.startswith('neural:')
            cls = NeuroidAssembly if neural else BehavioralAssembly
            payload = cls(np.arange(n)[:, None],
                          dims=['presentation', 'neuroid' if neural else 'choice'])
            session.respond(StreamEvent(channel, payload, None))

    def reset(self):
        self.resets += 1


def test_facade_runs_an_participate_only_subject_on_vision_calls():
    subject = _Recorder()
    facade = LegacyBenchmarkFacade(subject, visual_degrees=12)
    facade.start_recording('IT', time_bins=[(70, 170), (170, 270)])
    out = facade.look_at(_images(), number_of_trials=7)
    np.testing.assert_array_equal(out.values, [[0], [1]])
    session, = subject.sessions
    assert session.requested_output_channels == ('neural:IT',)
    assert session.time_bins == [(70, 170), (170, 270)]
    assert session.settings['number_of_trials'] == 7
    assert facade.visual_degrees() == 12
    assert facade.identifier == 'recording-subject'


def test_facade_runs_language_neural_and_behavior_in_one_session():
    subject = _Recorder()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_neural_recording('IT', recording_type='spikes')
    facade.start_behavioral_task('reading_times')
    text = np.array(['one', 'two'])
    out = facade.digest_text(text)
    assert set(out) == {'neural', 'behavior'}
    session, = subject.sessions
    assert set(session.requested_output_channels) == {'neural:IT', 'behavior'}
    assert session.task_context.task_type == 'reading_times'
    assert session.settings['recording_type'] == 'spikes'
    assert session.settings['legacy_text'] is text


def test_facade_passes_a_text_stimulus_set_unchanged():
    subject = _Recorder()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_neural_recording('IT')
    text = StimulusSet({'sentence': ['a', 'b'], 'stimulus_id': ['x', 'y'],
                        'context_id': ['p', 'p']})
    facade.digest_text(text)
    session, = subject.sessions
    assert session.stimulus_set is text
    assert 'legacy_text' not in session.task_context.metadata


def test_behavioral_task_needs_a_declared_behavior_output():
    class Neural(_Recorder):
        out_channels = {'neural:IT'}

    with pytest.raises(NotImplementedError, match="declares no 'behavior' output"):
        LegacyBenchmarkFacade(Neural()).start_behavioral_task('reading_times')


def test_facade_without_visual_field_fails_instead_of_inventing_one():
    with pytest.raises(NotImplementedError, match='declares no visual field'):
        LegacyBenchmarkFacade(_Recorder()).visual_degrees()


def test_look_at_needs_a_recording_and_one_region():
    facade = LegacyBenchmarkFacade(_Recorder())
    with pytest.raises(ValueError, match='start_recording'):
        facade.look_at(_images())
    facade.start_recording(['IT', 'V4'])
    with pytest.raises(ValueError, match='one region'):
        facade.look_at(_images())
    with pytest.raises(TypeError, match='unsupported settings'):
        facade.look_at(_images(), fixation=True)


def test_facade_reset_clears_settings_and_resets_subject():
    subject = _Recorder()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_recording('IT')
    facade.reset()
    assert subject.resets == 1
    with pytest.raises(ValueError, match='start_recording'):
        facade.look_at(_images())


# --- applying the facade at the scoring boundary --------------------------

def test_domain_loaders_mark_legacy_benchmarks_and_scoring_wraps_them():
    from types import SimpleNamespace
    from brainscore_core.legacy_facade import mark_legacy_benchmark, subject_for_benchmark

    legacy = SimpleNamespace(identifier='legacy')
    assert mark_legacy_benchmark(legacy).subject_api == 'legacy'
    session_native = SimpleNamespace(identifier='native', required_input_channels={'vision'})
    assert mark_legacy_benchmark(session_native).subject_api == 'session'
    typed = SimpleNamespace(identifier='typed', subject_api='typed')
    assert mark_legacy_benchmark(typed).subject_api == 'typed'

    subject = _Recorder()
    facade = subject_for_benchmark(subject, legacy, visual_degrees=8)
    assert isinstance(facade, LegacyBenchmarkFacade)
    assert facade.subject is subject and facade.visual_degrees() == 8
    assert subject_for_benchmark(subject, session_native) is subject
    assert subject_for_benchmark(subject, typed) is subject


# --- setup once, measure many times (review findings 1 to 4) ---------------


def _labelled_images(n=4, with_text=False):
    frame = {'stimulus_id': [f'f{i}' for i in range(n)],
             'image_path': [f'/tmp/f{i}.png' for i in range(n)],
             'image_label': ['cat', 'dog'] * (n // 2)}
    if with_text:
        frame['sentence'] = [f'caption {i}' for i in range(n)]
    stimuli = StimulusSet(pd.DataFrame(frame))
    stimuli.identifier = 'fitting-images'
    return stimuli


def test_fitting_settings_stay_apart_from_measurement_settings():
    subject = _Recorder()
    facade = LegacyBenchmarkFacade(subject)
    fitting = _labelled_images()
    facade.start_task('probabilities', fitting_stimuli=fitting,
                      number_of_trials=2, require_variance=True)
    facade.look_at(_images(), number_of_trials=1)
    facade.look_at(_images(), number_of_trials=3)
    first, second = subject.sessions
    assert first.task_context is second.task_context  # one task object, sent each time
    assert first.task_context.metadata['task_settings'] == {
        'number_of_trials': 2, 'require_variance': True}
    assert first.task_context.fitting_stimuli is fitting
    assert first.settings == {'number_of_trials': 1, 'require_variance': False}
    assert second.settings == {'number_of_trials': 3, 'require_variance': False}


def test_text_identifiers_do_not_collide_on_the_delimiter():
    first = text_stimulus_set(['a|b', 'c'])
    second = text_stimulus_set(['a', 'b|c'])
    assert first.identifier != second.identifier
