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

from brainscore_core.contract import Subject
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
        n = len(session.stimuli)
        for channel in session.outputs:
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
    assert session.outputs == ('neural:IT',)
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
    assert set(session.outputs) == {'neural:IT', 'behavior'}
    assert session.task.task_type == 'reading_times'
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
    assert session.stimuli is text
    assert 'legacy_text' not in session.task.metadata


def test_behavioral_task_needs_a_declared_behavior_output():
    class Neural(_Recorder):
        out_channels = {'neural:IT'}

    from brainscore_core.compatibility import CompatibilityError, ExecutionNotFeasibleError
    subject = Neural()
    with pytest.raises(ExecutionNotFeasibleError, match="declares no 'behavior' output") as caught:
        LegacyBenchmarkFacade(subject).start_behavioral_task('reading_times')
    assert caught.value.channels == {'behavior'}
    assert caught.value.task_type == 'reading_times'
    assert not isinstance(caught.value, CompatibilityError)
    assert subject.sessions == []


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


def test_facade_reset_clears_settings_even_when_the_subject_reset_raises():
    subject = _Recorder()

    def failing_reset():
        raise RuntimeError('undo failed')
    subject.reset = failing_reset
    facade = LegacyBenchmarkFacade(subject)
    facade.start_recording('IT')
    with pytest.raises(RuntimeError, match='undo failed'):
        facade.reset()
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
    assert mark_legacy_benchmark(legacy).subject_api == 'legacy'   # a second call keeps the mark
    unmarked = SimpleNamespace(identifier='unmarked')

    subject = _Recorder()
    facade = subject_for_benchmark(subject, legacy, visual_degrees=8)
    assert isinstance(facade, LegacyBenchmarkFacade)
    assert facade.subject is subject and facade.visual_degrees() == 8
    assert subject_for_benchmark(subject, session_native) is subject
    assert subject_for_benchmark(subject, unmarked) is subject


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
    assert first.task is second.task  # one task object, sent each time
    assert first.task.metadata['task_settings'] == {
        'number_of_trials': 2, 'require_variance': True}
    assert first.task.fitting_stimuli is fitting
    assert first.settings == {'number_of_trials': 1, 'require_variance': False}
    assert second.settings == {'number_of_trials': 3, 'require_variance': False}


def test_text_identifiers_do_not_collide_on_the_delimiter():
    first = text_stimulus_set(['a|b', 'c'])
    second = text_stimulus_set(['a', 'b|c'])
    assert first.identifier != second.identifier


class _LabelAnswerer(_Recorder):
    """A subject whose bare behavior answer is a legacy label answer: strings, dims (choice, presentation)."""

    def participate(self, session):
        self.sessions.append(session)
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        session.respond(StreamEvent('behavior', BehavioralAssembly(
            [['cat'] * len(ids)], coords={'stimulus_id': ('presentation', ids),
                                          'truth': ('presentation', ['cat'] * len(ids))},
            dims=['choice', 'presentation']), None))


def test_a_legacy_answer_of_labels_reaches_the_benchmark_unchanged():
    """resolve_invalid_answers (core #208) called isnan on label strings and raised for
    every legacy label task and odd_one_out task; a non-float answer now passes unchanged."""
    from brainscore_core.legacy_facade import resolve_invalid_answers
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b']}))
    stimuli.identifier = 'facade-labels'
    facade = LegacyBenchmarkFacade(_LabelAnswerer(), visual_degrees=8)
    facade.start_task('odd_one_out')
    answer = facade.look_at(stimuli)
    assert answer.dims == ('choice', 'presentation') and list(answer.values[0]) == ['cat', 'cat']
    strings = BehavioralAssembly([['x', 'y']], coords={'stimulus_id': ('presentation', ['a', 'b'])},
                                 dims=['choice', 'presentation'])
    assert resolve_invalid_answers(strings) is strings


class _ChoiceAnswerer(_Recorder):
    """Answers behavior:choice with one-hot rows (the second label for every stimulus)."""
    out_channels = {'behavior', 'behavior:choice'}

    def participate(self, session):
        from brainscore_core.response_formats import response_record
        self.sessions.append(session)
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        labels = list(session.task.response.labels)
        rows = np.zeros((len(ids), len(labels)), dtype=np.float32)
        rows[:, 1] = 1.0
        session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
            rows, coords={'stimulus_id': ('presentation', ids),
                          'truth': ('presentation', list(session.stimuli['truth'])),
                          'choice': labels},
            dims=['presentation', 'choice'],
            attrs={'response': response_record('behavior:choice', 'test')}), None))


def _label_stimuli():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'], 'truth': ['dog', 'cat']}))
    stimuli.identifier = 'facade-label-task'
    return stimuli


def test_a_label_task_with_a_list_is_asked_as_a_choice_and_answered_as_labels():
    from brainscore_core.responses import Choice
    subject = _ChoiceAnswerer()
    facade = LegacyBenchmarkFacade(subject, visual_degrees=8)
    facade.start_task('label', ['cat', 'dog'])
    labels = facade.look_at(_label_stimuli())
    (session,) = subject.sessions
    assert session.task.response == Choice(['cat', 'dog'])
    assert session.task.instruction is None and session.task.fitting_stimuli is None
    assert list(session.outputs) == ['behavior:choice']
    assert labels.dims == ('choice', 'presentation')
    assert list(labels.values[0]) == ['dog', 'dog']
    assert list(labels['truth'].values) == ['dog', 'cat']
    assert 'response_state' not in labels.coords and 'raw_answer' not in labels.coords


@pytest.mark.parametrize('task, setting', [('label', 'imagenet'), ('label', ['cat']),
                                          ('probabilities', None), ('odd_one_out', None)])
def test_other_legacy_tasks_stay_on_bare_behavior(task, setting):
    facade = LegacyBenchmarkFacade(_ChoiceAnswerer(), visual_degrees=8)
    facade.start_task(task, setting)
    assert facade._task.response is None and facade._task.task_type == task


def test_a_label_task_refuses_a_subject_without_a_choice_output():
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    subject = _Recorder()
    facade = LegacyBenchmarkFacade(subject, visual_degrees=8)
    facade.start_task('label', ['cat', 'dog'])
    with pytest.raises(ExecutionNotFeasibleError, match="declares no 'behavior:choice' output"):
        facade.look_at(_label_stimuli())
    assert subject.sessions == []


class _RowsWithOneInvalid(_Recorder):
    """Answers 20 stimuli with one-hot rows; row 3 is all NaN (an invalid answer). It answers
    behavior:choice (the new label route) and bare behavior (the plain old path) alike."""
    out_channels = {'behavior', 'behavior:choice'}

    def participate(self, session):
        from brainscore_core.response_formats import response_record
        self.sessions.append(session)
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        rows = np.zeros((len(ids), 2), dtype=np.float32)
        rows[:, 0] = 1.0
        rows[3] = np.nan
        (channel,) = session.outputs
        attrs = ({'response': response_record('behavior:choice', 'test')}
                 if channel == 'behavior:choice' else {})
        session.respond(StreamEvent(channel, BehavioralAssembly(
            rows, coords={'stimulus_id': ('presentation', ids), 'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs=attrs), None))


def test_the_label_route_keeps_the_count_of_invalid_answers():
    """S9-01: the turn back from the choice keeps every attribute but the format's record."""
    from brainscore_core.contract import TaskContext
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(20)]}))
    stimuli.identifier = 'facade-invalid-count'
    new = LegacyBenchmarkFacade(_RowsWithOneInvalid(), visual_degrees=8)
    new.start_task('label', ['cat', 'dog'])
    plain = LegacyBenchmarkFacade(_RowsWithOneInvalid(), visual_degrees=8)
    plain.start_task(TaskContext(task_type='label', fitting_stimuli=['cat', 'dog']))   # bare behavior
    with pytest.warns(UserWarning, match='1 invalid answers'):
        labels = new.look_at(stimuli)
    with pytest.warns(UserWarning, match='1 invalid answers'):
        old = plain.look_at(stimuli)
    assert labels.attrs['n_invalid_answers'] == 1
    assert 'response' not in labels.attrs
    assert labels.attrs == {key: value for key, value in old.attrs.items() if key != 'response'}


def test_only_text_answers_pass_resolve_invalid_answers_unchanged():
    """S9-02: text labels (string or object dtype) pass; a numeric answer of any kind
    takes the invalid-answer path (a complex all-NaN answer raises the limit error)."""
    from brainscore_core.legacy_facade import resolve_invalid_answers
    ids = {'stimulus_id': ('presentation', ['a', 'b'])}
    for values in (np.array([['x', 'y']]), np.array([['x', 'y']], dtype=object)):
        answer = BehavioralAssembly(values, coords=ids, dims=['choice', 'presentation'])
        assert resolve_invalid_answers(answer) is answer
    complex_nan = BehavioralAssembly(np.full((2, 2), np.nan, dtype=complex),
                                     coords={**ids, 'choice': ['x', 'y']}, dims=['presentation', 'choice'])
    with pytest.raises(ValueError, match='above the 10% limit'):
        resolve_invalid_answers(complex_nan)


def test_the_facade_counts_replaced_answers_for_the_score():
    """The count is summed over look_at calls, starts at None, and reaches a score only
    when a numeric behavior answer was checked."""
    from brainscore_core.legacy_facade import record_replaced_answers
    from brainscore_core.metrics import Score
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(20)]}))
    stimuli.identifier = 'facade-count-sum'
    facade = LegacyBenchmarkFacade(_RowsWithOneInvalid(), visual_degrees=8)
    assert facade.n_invalid_answers is None
    untouched = Score(1.0)
    record_replaced_answers(untouched, facade)
    assert 'n_invalid' not in untouched.attrs          # no answer checked: no claim
    facade.start_task('label', ['cat', 'dog'])
    with pytest.warns(UserWarning):
        facade.look_at(stimuli)
        facade.look_at(stimuli)
    score = Score(1.0)
    record_replaced_answers(score, facade)
    assert score.attrs['n_invalid'] == 2 and 'random label' in score.attrs['answer_policy']
    plain = Score(1.0)
    record_replaced_answers(plain, _Recorder())         # a plain subject: nothing written
    assert 'n_invalid' not in plain.attrs


# --- the answer of an old benchmark carries the stimulus columns -----------------------

def _table(ids=('a', 'b', 'c', 'd')):
    table = StimulusSet(pd.DataFrame({
        'stimulus_id': list(ids), 'animal': ['cat', 'dog', 'cat', 'dog'][:len(ids)],
        'image_type': ['w', 'w', 'f', 'f'][:len(ids)]}))
    table.identifier = f'columns-{len(ids)}'
    return table


class _IdOnlyChoice(_Recorder):
    """The chance pattern: answers behavior:choice with the stimulus_id of each row, no other column."""
    out_channels = {'behavior:choice'}

    def participate(self, session):
        from brainscore_core.response_formats import response_record
        self.sessions.append(session)
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        rows = np.tile(np.float32([1.0, 0.0]), (len(ids), 1))
        session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
            rows, coords={'stimulus_id': ('presentation', ids), 'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'],
            attrs={'response': response_record('behavior:choice', 'test')}), None))


def _baker_like_metric(source):
    """Reads a stimulus column other than the id, as the Baker metric reads 'animal'."""
    scores = []
    for category in sorted(set(source['animal'].values)):
        this = source.sel(animal=category, image_type='w')
        scores.append(float((this.values == category).mean()))
    return scores


def _bmd_like_metric(source):
    """Sorts by stimulus_id, as the BMD metric does."""
    return list(source.sortby('stimulus_id')['stimulus_id'].values)


def test_a_subject_that_answers_by_stimulus_id_reaches_an_old_metric_with_the_columns():
    facade = LegacyBenchmarkFacade(_IdOnlyChoice(), visual_degrees=8)
    facade.start_task('label', ['cat', 'dog'])
    answer = facade.look_at(_table())
    assert list(answer.indexes['presentation'].names) == ['stimulus_id', 'animal', 'image_type']
    assert _baker_like_metric(answer) == [1.0, 0.0]
    assert _bmd_like_metric(answer) == ['a', 'b', 'c', 'd']


def test_the_columns_are_added_once_and_in_the_order_of_the_answer_rows():
    from brainscore_core.legacy_facade import _stimulus_columns
    table = _table()
    added = _stimulus_columns({'stimulus_id': ['c', 'a', 'a'], 'animal': ['x', 'y', 'z']}, table)
    assert list(added) == ['image_type']                 # a column the answer has is not replaced
    assert added['image_type'] == ['f', 'w', 'w']
    assert _stimulus_columns({'stimulus_id': ['zzz']}, table) == {}      # an id the table lacks: nothing
    assert _stimulus_columns({'animal': ['x']}, table) == {}             # no stimulus_id: nothing


def test_an_answer_that_has_every_column_is_returned_unchanged():
    from brainscore_core.legacy_facade import _with_stimulus_columns
    table = _table()
    answer = BehavioralAssembly(
        np.full((4, 2), 0.5), dims=['presentation', 'choice'],
        coords={'stimulus_id': ('presentation', list(table['stimulus_id'])),
                'animal': ('presentation', list(table['animal'])),
                'image_type': ('presentation', list(table['image_type'])), 'choice': ['cat', 'dog']})
    assert _with_stimulus_columns(answer, table) is answer
