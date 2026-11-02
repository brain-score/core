"""Fidelity tests for the legacy-benchmark facade (Contract Rules v3, rule 11.2).

Each legacy method a vision or language benchmark calls on a candidate is
checked against the legacy behavior it stands in for.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.legacy_facade import (
    LegacyBenchmarkFacade,
    repeat_over_time_bins,
    text_stimulus_set,
)
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _Extractor:
    """Deterministic features: one neuroid per character of the stimulus id."""

    def __init__(self):
        self.calls = []
        self.seen = []

    def __call__(self, stimuli, layers=None, **kwargs):
        self.calls.append(list(stimuli['stimulus_id']))
        self.seen.append(stimuli)
        n = len(stimuli)
        return NeuroidAssembly(
            np.arange(n * 3, dtype=float).reshape(n, 3),
            coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                'neuroid_id': ('neuroid', np.arange(3)),
                'layer': ('neuroid', [layers[0]] * 3),
            },
            dims=['presentation', 'neuroid'],
        )


def _model(modality='vision'):
    extractor = _Extractor()
    model = BrainScoreModel('facade-test', region_layer_map={'IT': 'layer'},
                            extractor={modality: extractor}, visual_degrees=10)
    return model, extractor


def _images(n=2):
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_path': [f'/tmp/s{i}.png' for i in range(n)],
    }))
    stimuli.identifier = f'images-{n}'
    return stimuli


def test_look_at_matches_process():
    model, _ = _model()
    model.start_recording('IT', time_bins=[(70, 170)])
    via_legacy = model.look_at(_images())
    via_process = model.process(_images())
    np.testing.assert_array_equal(via_legacy.values, via_process.values)
    assert 'time_bin' not in via_legacy.dims  # one bin: unchanged, as in legacy


def test_several_time_bins_repeat_a_static_response():
    model, _ = _model()
    bins = [(70, 170), (170, 270)]
    model.start_recording('IT', time_bins=bins)
    responses = model.look_at(_images())
    assert responses.sizes['time_bin'] == 2
    assert list(responses['time_bin_start'].values) == [70, 170]
    assert list(responses['time_bin_end'].values) == [170, 270]
    first, second = responses.isel(time_bin=0), responses.isel(time_bin=1)
    np.testing.assert_array_equal(first.values, second.values)


def test_number_of_trials_alone_does_not_change_deterministic_responses():
    model, _ = _model()
    model.start_recording('IT', time_bins=[(70, 170)])
    once = model.look_at(_images(), number_of_trials=1)
    many = model.look_at(_images(), number_of_trials=5)
    np.testing.assert_array_equal(once.values, many.values)


def test_variance_request_fails_instead_of_returning_identical_trials():
    model, extractor = _model()
    model.start_recording('IT', time_bins=[(70, 170)])
    with pytest.raises(NotImplementedError, match='trial-to-trial variance'):
        model.look_at(_images(), number_of_trials=3, require_variance=True)
    assert extractor.calls == []


def test_visual_degrees_reported():
    model, _ = _model()
    assert model.visual_degrees() == 10


def test_start_behavioral_task_fails_clearly():
    model, _ = _model('text')
    with pytest.raises(NotImplementedError, match="'reading_times'"):
        model.start_behavioral_task('reading_times')


def test_digest_text_returns_neural_part_for_a_passage():
    model, extractor = _model('text')
    model.start_neural_recording('IT', 'fMRI')
    result = model.digest_text(['the quick brown', 'fox jumps'])
    assert set(result) == {'neural'}
    assert result['neural'].sizes['presentation'] == 2
    assert extractor.calls == [[0, 1]]


def test_text_stimulus_set_keys_cache_by_content():
    first = text_stimulus_set(['a passage'])
    second = text_stimulus_set(['another passage'])
    assert first.identifier != second.identifier
    assert list(first['context_id']) == ['passage']
    stimuli = _images()
    assert text_stimulus_set(stimuli) is stimuli


def test_facade_wraps_a_subject_directly():
    model, _ = _model()
    facade = LegacyBenchmarkFacade(model, visual_degrees=4)
    facade.start_recording('IT', time_bins=[(70, 170)])
    assert facade.identifier == 'facade-test'
    assert facade.visual_degrees() == 4
    assert facade.look_at(_images()).sizes['presentation'] == 2


def test_repeat_over_time_bins_keeps_existing_time_axis():
    model, _ = _model()
    model.start_recording('IT', time_bins=[(0, 10), (10, 20)])
    timed = model.process(_images())
    assert repeat_over_time_bins(timed, [(0, 10), (10, 20)]) is timed


@pytest.mark.parametrize('time_bins', [
    [(70, 170)],
    np.array([[70, 170]]),
    (70, 170),
    np.array([(70, 170)], dtype=object),
    pd.MultiIndex.from_tuples([(70, 170)]).values,
])
def test_single_time_bin_in_any_benchmark_form_leaves_response_unchanged(time_bins):
    model, _ = _model()
    model.start_recording('IT', time_bins=time_bins)
    assert 'time_bin' not in model.process(_images()).dims


@pytest.mark.parametrize('kwargs', [
    {'activations_model': {'vision': _Extractor()}},
    {'extractor': {'vision': _Extractor()}, 'generation_fn': lambda *a: None},
    {'extractor': {'vision': _Extractor()}, 'action_fn': lambda step: None},
    {'extractor': {'vision': _Extractor()}, 'state_change_fn': lambda c: (None, lambda: None)},
])
def test_older_constructor_names_warn_and_keep_working(kwargs):
    with pytest.warns(DeprecationWarning, match='is deprecated; use'):
        model = BrainScoreModel('old-names', region_layer_map={'IT': 'layer'}, **kwargs)
    assert 'neural:IT' in model.out_channels


def test_preferred_names_do_not_warn():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        BrainScoreModel('new-names', region_layer_map={'IT': 'layer'},
                        extractor={'vision': _Extractor()},
                        answer=lambda *a: None, policy=lambda step: None,
                        intervention=lambda c: (None, lambda: None))


# --- the facade runs any subject through interact ------------------------

from brainscore_core.model_interface import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly


class _Recorder(Subject):
    """An interact-only subject: no process, look_at or start_recording."""

    identifier = 'recording-subject'
    in_channels = {'text', 'vision'}
    out_channels = {'neural:IT', 'behavior'}
    required_channels = set()

    def __init__(self):
        self.sessions = []
        self.resets = 0

    def interact(self, session):
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


def test_facade_runs_an_interact_only_subject_on_vision_calls():
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


def test_typed_look_at_runs_through_interact():
    model, extractor = _model()
    sessions = []
    original = model.interact
    model.interact = lambda session: (sessions.append(session), original(session))[1]
    model.start_recording('IT', time_bins=[(70, 170)])
    via_legacy = model.look_at(_images())
    assert len(sessions) == 1
    assert sessions[0].requested_output_channels == ('neural:IT',)
    np.testing.assert_array_equal(via_legacy.values, model.process(_images()).values)


def test_typed_behavioral_look_at_matches_process():
    extractor = _Extractor()
    model = BrainScoreModel('facade-behavior', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor},
                            behavioral_readout_layer='layer')
    fitting = StimulusSet(pd.DataFrame({
        'stimulus_id': ['f0', 'f1', 'f2', 'f3'],
        'image_path': ['/tmp/f0.png', '/tmp/f1.png', '/tmp/f2.png', '/tmp/f3.png'],
        'image_label': ['cat', 'dog', 'cat', 'dog'],
    }))
    fitting.identifier = 'fitting'
    model.start_task('probabilities', fitting_stimuli=fitting)
    via_legacy = model.look_at(_images())
    via_process = model.process(_images())
    assert isinstance(via_legacy, BehavioralAssembly)
    np.testing.assert_array_equal(via_legacy.values, via_process.values)


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


def test_declared_visual_degrees_reads_the_legacy_vision_setting_or_none():
    from brainscore_core.legacy_facade import declared_visual_degrees
    model, _ = _model()
    assert declared_visual_degrees(model) == 10
    assert declared_visual_degrees(_Recorder()) is None


# --- setup once, measure many times (review findings 1 to 4) ---------------

from brainscore_core.model_interface import TaskContext

def _labelled_images(n=4, with_text=False):
    frame = {'stimulus_id': [f'f{i}' for i in range(n)],
             'image_path': [f'/tmp/f{i}.png' for i in range(n)],
             'image_label': ['cat', 'dog'] * (n // 2)}
    if with_text:
        frame['sentence'] = [f'caption {i}' for i in range(n)]
    stimuli = StimulusSet(pd.DataFrame(frame))
    stimuli.identifier = 'fitting-images'
    return stimuli


def _readout_model():
    extractor = _Extractor()
    model = BrainScoreModel('readout', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor},
                            behavioral_readout_layer='layer')
    return model, extractor


def _fits(extractor):
    """How many times the fitting stimuli were extracted (one per readout fit)."""
    return sum(1 for ids in extractor.calls if ids and str(ids[0]).startswith('f'))


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


def test_typed_behavioral_look_at_fits_the_readout_once():
    model, extractor = _readout_model()
    model.start_task('probabilities', fitting_stimuli=_labelled_images())
    assert _fits(extractor) == 1
    model.look_at(_images())
    model.look_at(_images())
    assert _fits(extractor) == 1


def test_scored_facade_fits_the_readout_once_across_measurements():
    model, extractor = _readout_model()
    facade = LegacyBenchmarkFacade(model, visual_degrees=8)
    facade.start_task('probabilities', fitting_stimuli=_labelled_images())
    facade.look_at(_images())
    facade.look_at(_images())
    assert _fits(extractor) == 1
    facade.start_task('probabilities', fitting_stimuli=_labelled_images())  # a new task
    facade.look_at(_images())
    assert _fits(extractor) == 2


def test_repeated_task_refits_after_a_state_change():
    from brainscore_core.events import PerturbationApplied, Perturbation, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change, behavioral_response
    extractor = _Extractor()
    model = BrainScoreModel(
        'lesioned', region_layer_map={'IT': 'layer'}, extractor={'vision': extractor},
        behavioral_readout_layer='layer',
        intervention=lambda change: (
            PerturbationApplied(handle_id='h', target=change.target,
                                perturbation=change.perturbation),
            lambda: None))
    task = TaskContext(task_type='probabilities', fitting_stimuli=_labelled_images(),
                       metadata={'stimulus_set': _images()})
    behavioral_response(model, task)
    behavioral_response(model, task)
    assert _fits(extractor) == 1
    apply_state_change(model, StateChange(kind='lesion', target=Selection(layer='layer'),
                                          perturbation=Perturbation(kind='zero')))
    behavioral_response(model, task)
    assert _fits(extractor) == 2  # the model changed under the readout


def test_fitting_stimuli_reach_the_readout_with_declared_inputs_only():
    model, extractor = _readout_model()
    fitting = _labelled_images(with_text=True)
    model.start_task(TaskContext(task_type='probabilities', fitting_stimuli=fitting))
    seen = extractor.seen[-1]
    assert 'sentence' not in seen.columns
    assert list(seen['image_label']) == list(fitting['image_label'])
    assert 'sentence' in fitting.columns  # the benchmark's table is untouched
    assert seen.identifier == fitting.identifier


def test_typed_behavioral_variance_request_fails_before_compute():
    model, extractor = _readout_model()
    model.start_task('probabilities', fitting_stimuli=_labelled_images())
    calls = len(extractor.calls)
    with pytest.raises(NotImplementedError, match='trial-to-trial variance'):
        model.look_at(_images(), number_of_trials=3, require_variance=True)
    assert len(extractor.calls) == calls


def test_text_identifiers_do_not_collide_on_the_delimiter():
    first = text_stimulus_set(['a|b', 'c'])
    second = text_stimulus_set(['a', 'b|c'])
    assert first.identifier != second.identifier


def test_changed_fitting_data_in_the_same_task_object_refits():
    """TaskContext is mutable: an explicit start_task after a change fits again."""
    model, extractor = _readout_model()
    task = TaskContext(task_type='probabilities', fitting_stimuli=_labelled_images())
    model.start_task(task)
    model.start_task(task)
    assert _fits(extractor) == 1  # unchanged task sent again: no refit
    task.fitting_stimuli = _labelled_images()  # a new table
    task.fitting_stimuli['image_label'] = ['dog', 'cat', 'dog', 'cat']
    model.start_task(task)
    assert _fits(extractor) == 2
    task.fitting_stimuli['image_label'] = ['cat', 'cat', 'dog', 'dog']  # relabelled in place
    model.start_task(task)
    assert _fits(extractor) == 3
    model.look_at(_images())  # a measurement on the unchanged task: no refit
    assert _fits(extractor) == 3
