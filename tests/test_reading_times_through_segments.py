"""The old reading-times task through behavior:log_probabilities, segment layout (ADR-0075).

A subject that declares the format answers one Segments trial for each digest_text call;
the old benchmark receives minus each value in bits. A subject that declares only bare
behavior keeps the old route.
"""
import math

import numpy as np
import pandas as pd
import pytest

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade, READING_TIME_POLICY
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Segments, trial_prompt
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

F6 = 'behavior:log_probabilities'


def _parameters(**changes):
    parameters = {'layout': 'segment', 'context': 'prefix', 'aggregation': 'sum', 'units': 'bits',
                  'tokenization': {'join': 'stand-in', 'special_tokens': 'none',
                                   'template': 'none', 'truncation': 'none'}}
    parameters.update(changes)
    return parameters


class _Segmenter(Subject):
    """Answers a Segments trial with -(position) bits per part; position 1 is NaN."""

    identifier = 'segmenter'
    in_channels = {'text'}
    out_channels = {F6, 'neural:language_system'}
    required_channels = set()

    def __init__(self, parameters=None, extra_positions=0, first_nan=True, values=None):
        self.sessions = []
        self.parameters = parameters or _parameters()
        self.extra_positions = extra_positions
        self.first_nan = first_nan
        self.values = values

    def participate(self, session):
        self.sessions.append(session)
        while session.next_input() is not None:
            pass
        for channel in session.outputs:
            if channel.startswith('neural:'):
                n = len(session.stimuli)
                session.respond(StreamEvent(channel, NeuroidAssembly(
                    np.zeros((n, 1)), dims=['presentation', 'neuroid']), None))
                continue
            row = session.stimuli.iloc[0]
            parts = trial_prompt(session.task, row).response.segments
            n = len(parts) + self.extra_positions
            values = (np.array(self.values, dtype=np.float64) if self.values is not None
                      else -np.arange(1, n + 1, dtype=np.float64) * 0.5)
            valid, error = True, ''
            if self.first_nan:
                values[0] = np.nan
                valid, error = False, 'position 1: no conditioning context'
            record = response_record(F6, 'likelihood', category='model-derived',
                                     parameters=self.parameters)
            payload = BehavioralAssembly(values[None, :], dims=['presentation', 'choice'], coords={
                'stimulus_id': ('presentation', [row['stimulus_id']]),
                'answer_valid': ('presentation', [valid]),
                'parse_error': ('presentation', [error]),
                'choice': [str(i) for i in range(1, n + 1)]}, attrs={'response': record})
            session.respond(StreamEvent(channel, payload, None))

    def reset(self):
        pass


class _OldModel(Subject):
    """Declares bare behavior only, as the language adapter of an old model does."""

    identifier = 'old'
    in_channels = {'text'}
    out_channels = {'behavior'}
    required_channels = set()

    def __init__(self):
        self.tasks = []

    def participate(self, session):
        self.tasks.append(session.task)
        while session.next_input() is not None:
            pass
        n = len(session.stimuli)
        session.respond(StreamEvent('behavior', BehavioralAssembly(
            np.full(n, 300.0), dims=['presentation']), None))

    def reset(self):
        pass


def _reading_times(subject, text):
    facade = LegacyBenchmarkFacade(subject)
    facade.start_behavioral_task('reading_times')
    return facade, facade.digest_text(text)['behavior']


@pytest.mark.parametrize('text', [['the', 'dog', 'ran.', 'the'], np.array(['the', 'dog', 'ran.', 'the'])])
def test_a_likelihood_subject_gives_the_surprisal_of_each_part(text):
    facade, answer = _reading_times(_Segmenter(), text)
    assert answer.dims == ('presentation',)
    assert list(answer['stimulus'].values) == ['the', 'dog', 'ran.', 'the']   # repeats kept
    assert list(answer['part_number'].values) == [0, 1, 2, 3]
    assert list(answer['context'].values)[-1] == 'the dog ran. the'
    assert np.isnan(answer.values[0])                      # the raw first value, as it is
    np.testing.assert_array_equal(answer.values[1:], [1.0, 1.5, 2.0])
    assert answer.values.dtype == np.float64               # a NaN part: float64
    assert facade.n_invalid_answers == 1
    assert READING_TIME_POLICY in facade.answer_policy


def test_one_trial_per_call_with_the_parts_in_a_column_and_the_join_as_text():
    subject = _Segmenter()
    _reading_times(subject, ['If', 'you', 'were', 'here', ','])
    session, = subject.sessions
    assert isinstance(session.task.response, Segments)
    assert session.task.trial_memory == 'none'
    assert session.outputs == (F6,)
    assert len(session.stimuli) == 1
    assert list(session.stimuli['sentence']) == ['If you were here,']


def test_a_string_and_a_table_are_one_passage_each():
    _, answer = _reading_times(_Segmenter(), 'one word')
    assert list(answer['stimulus'].values) == ['one word']
    table = StimulusSet(pd.DataFrame({'sentence': ['a', 'b'], 'stimulus_id': [0, 1]}))
    table.identifier = 'two-parts'
    _, answer = _reading_times(_Segmenter(), table)
    assert list(answer['stimulus'].values) == ['a', 'b']


def test_all_finite_values_are_float32_and_nats_are_converted_to_bits():
    _, answer = _reading_times(_Segmenter(first_nan=False), ['a', 'b'])
    assert answer.values.dtype == np.float32
    np.testing.assert_array_equal(answer.values, [0.5, 1.0])
    assert answer.attrs['units'] == 'bits'
    subject = _Segmenter(parameters=_parameters(units='nats'), first_nan=False)
    _, answer = _reading_times(subject, ['a', 'b'])
    np.testing.assert_allclose(answer.values, [0.5 / math.log(2), 1.0 / math.log(2)], rtol=1e-6)
    assert answer.attrs['units'] == 'bits'
    assert answer.attrs['response']['parameters']['units'] == 'nats'   # the source record


def test_values_that_float32_cannot_hold_stay_float64():
    _, answer = _reading_times(_Segmenter(first_nan=False, values=[-0.1, -0.2]), ['a', 'b'])
    assert answer.values.dtype == np.float64
    np.testing.assert_array_equal(answer.values, [0.1, 0.2])


def test_an_invalid_trial_with_finite_values_is_counted_with_its_reason():
    facade, answer = _reading_times(_Segmenter(first_nan=False, values=[0.25, 0.25]), ['a', 'b'])
    np.testing.assert_array_equal(answer.values, [-0.25, -0.25])      # the raw values
    assert facade.n_invalid_answers == 1
    assert answer.attrs['response_state'] == 'invalid'
    assert 'above 0' in answer.attrs['response_reason']


def test_a_wrong_number_of_positions_is_malformed():
    with pytest.raises(ValueError, match='asks for 2 parts'):
        _reading_times(_Segmenter(extra_positions=1), ['a', 'b'])


def test_a_mean_aggregation_cannot_give_reading_times():
    with pytest.raises(ValueError, match='aggregation'):
        _reading_times(_Segmenter(parameters=_parameters(aggregation='mean')), ['a', 'b'])


def test_a_segments_task_refuses_a_continuation_answer():
    parameters = _parameters(layout='continuation', context='shared')
    with pytest.raises(ValueError, match='segment layout'):
        _reading_times(_Segmenter(parameters=parameters), ['a', 'b'])


def test_an_empty_part_cannot_be_scored_and_is_refused_before_any_answer():
    subject = _Segmenter()
    with pytest.raises(ValueError, match='an empty part cannot be scored'):
        _reading_times(subject, ['a', '', 'b'])
    with pytest.raises(ValueError, match='an empty part cannot be scored'):
        Segments(('a', ' '))


def test_a_row_whose_text_is_not_the_join_of_its_parts_is_refused():
    task = TaskContext(response=Segments('segments'))
    with pytest.raises(ValueError, match='join of its parts'):
        trial_prompt(task, {'sentence': 'other text', 'segments': ('a', 'b'), 'stimulus_id': 0})
    prompt = trial_prompt(task, {'sentence': 'a b.', 'segments': ['a', 'b.'], 'stimulus_id': 0})
    assert prompt.response.segments == ('a', 'b.')
    given = TaskContext(response=Segments(('dogs',)))      # the parts given on the request
    with pytest.raises(ValueError, match='join of its parts'):
        trial_prompt(given, {'sentence': 'fish', 'stimulus_id': 0})


def test_a_subject_with_only_bare_behavior_keeps_the_old_route():
    subject = _OldModel()
    _, answer = _reading_times(subject, ['a', 'b'])
    task, = subject.tasks
    assert task.task_type == 'reading_times' and task.response is None
    np.testing.assert_array_equal(answer.values, [300.0, 300.0])


def test_a_subject_with_neither_output_is_refused_before_any_session():
    subject = _Segmenter()
    subject.out_channels = {'neural:language_system'}
    facade = LegacyBenchmarkFacade(subject)
    with pytest.raises(CompatibilityError, match="no 'behavior:log_probabilities' output"):
        facade.start_behavioral_task('reading_times')
    assert subject.sessions == []


def test_a_joint_call_is_refused_before_any_session():
    subject = _Segmenter()
    facade = LegacyBenchmarkFacade(subject)
    facade.start_neural_recording('language_system')
    facade.start_behavioral_task('reading_times')
    with pytest.raises(CompatibilityError, match='neural recording'):
        facade.digest_text(['a', 'b'])
    assert subject.sessions == []
