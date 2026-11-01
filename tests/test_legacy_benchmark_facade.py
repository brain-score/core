"""Legacy benchmark methods retain their settings at the session boundary."""
import numpy as np

from brainscore_core.contract import Subject
from brainscore_core.legacy_benchmark import LegacyBenchmarkFacade
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class Recorder(Subject):
    identifier = 'recording-subject'
    in_channels = {'text', 'vision'}
    out_channels = {'neural:IT', 'behavior'}

    def __init__(self):
        self.sessions = []
        self.resets = 0

    def interact(self, session):
        self.sessions.append(session)
        channel, = session.requested_output_channels
        cls = NeuroidAssembly if channel.startswith('neural:') else BehavioralAssembly
        output = cls(np.arange(len(session.stimulus_set))[:, None],
                     dims=['presentation', 'neuroid' if channel.startswith('neural:') else 'choice'])
        session.respond(StreamEvent(channel=channel, payload=output, t_ms=None))

    def reset(self):
        self.resets += 1


def test_vision_methods_preserve_trial_count_bins_and_field_of_view():
    subject = Recorder()
    facade = LegacyBenchmarkFacade(subject, visual_degrees=12)
    facade.start_recording('IT', time_bins=[(70, 170)])
    result = facade.look_at(StimulusSet({'image_path': ['one', 'two']}), number_of_trials=7)
    np.testing.assert_array_equal(result.values, [[0], [1]])
    assert facade.identifier == subject.identifier
    assert facade.visual_degrees() == 12
    assert subject.sessions[0].time_bins == [(70, 170)]
    assert subject.sessions[0].task_context.metadata['number_of_trials'] == 7


def test_language_neural_method_preserves_existing_table_metadata():
    subject = Recorder()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_neural_recording('IT', recording_type='spikes')
    text = StimulusSet({'sentence': ['one', 'two'], 'stimulus_id': ['a', 'b'],
                        'context_id': ['first', 'second']})
    output = facade.digest_text(text)
    assert set(output) == {'neural'}
    assert subject.sessions[0].stimulus_set is text
    assert subject.sessions[0].task_context.metadata['recording_type'] == 'spikes'


def test_language_behavior_method_and_numpy_text():
    subject = Recorder()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_behavioral_task('reading_times')
    output = facade.digest_text(np.array(['one', 'two']))
    assert set(output) == {'behavior'}
    assert subject.sessions[0].task_context.task_type == 'reading_times'
    assert list(subject.sessions[0].stimulus_set['part_number']) == [0, 1]


def test_task_context_and_reset_have_one_owner():
    subject = Recorder()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_task('label', fitting_stimuli='imagenet')
    facade.look_at(StimulusSet({'image_path': ['one']}))
    assert subject.sessions[0].task_context.fitting_stimuli == 'imagenet'
    assert subject.resets == 0
    facade.reset()
    assert subject.resets == 1


def test_facade_selection_uses_benchmark_api_and_accepts_unhashable_subject():
    from types import SimpleNamespace
    from brainscore_core.legacy_benchmark import register_visual_field, subject_for_benchmark
    class UnhashableSubject(Recorder):
        __hash__ = None
    subject = UnhashableSubject()
    register_visual_field(subject, lambda: 12)
    native = SimpleNamespace(subject_api='session')
    legacy = SimpleNamespace(subject_api='legacy')
    assert subject_for_benchmark(subject, native) is subject
    assert subject_for_benchmark(subject, legacy).visual_degrees() == 12
