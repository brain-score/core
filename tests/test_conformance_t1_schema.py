"""Conformance test T1 (Response Format Standard, section 11.1): the schema of each format.

A stub subject answers through ``participate`` (and a direct ``Session.respond``
for single cases). A well-formed answer, as an assembly or as one cell, is
stored with the full schema and a state for every trial. An invalid value is
stored as an invalid trial. A malformed response raises and nothing is kept.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

IDS = ['s0', 's1']
TASKS = {
    'behavior:choice': TaskContext(response=Choice(['cat', 'dog'])),
    'behavior:probabilities': TaskContext(response=Choice(['cat', 'dog'], probabilities=True)),
    'behavior:rating': TaskContext(response=Scale(1, 7, 1)),
    'behavior:confidence': TaskContext(response=Choice(['cat', 'dog'], confidence=Scale(1, 5))),
    'behavior:rt': TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal')),
    'behavior:text': TaskContext(response=Text()),
}
DTYPES = {'behavior:choice': np.float32, 'behavior:probabilities': np.float32,
          'behavior:text': object}


def _stimuli():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': IDS, 'text': ['x', 'y']}))
    stimuli.identifier = 't1-stimuli'
    return stimuli


def _record(channel):
    parameters = {'units': 'ms'} if channel == 'behavior:rt' else {}
    return response_record(channel, 'external', parameters=parameters)


def _rows(channel, values, ids=IDS, axis=('cat', 'dog'), dtype=None, record='default', **coords):
    values = np.array(values, dtype=dtype or DTYPES.get(channel, np.float64))
    built = {'stimulus_id': ('presentation', list(ids)),
             **{name: ('presentation', list(column)) for name, column in coords.items()}}
    dims = ['presentation']
    if values.ndim == 2:
        built['choice'] = list(axis)
        dims.append('choice')
    attrs = {} if record is None else {'response': _record(channel) if record == 'default' else record}
    return BehavioralAssembly(values, coords=built, dims=dims, attrs=attrs)


class _Stub(Subject):
    """Answers every trial of a table session with one fixed response."""
    identifier = 'stub'
    in_channels = {'text'}
    required_channels = set()

    out_channels = set(TASKS)

    def __init__(self, channel, payload, meta=None):
        self.channel, self.payload, self.meta = channel, payload, meta or {}

    def participate(self, session):
        while session.next_input() is not None:
            pass
        session.respond(StreamEvent(self.channel, self.payload, None, dict(self.meta)))

    def reset(self):
        pass


def _participate(channel, payload, meta=None):
    session = stimulus_session(_stimuli(), [channel], TASKS[channel])
    _Stub(channel, payload, meta).participate(session)
    return session


WELL_FORMED = {
    'behavior:choice': [[1.0, 0.0], [0.0, 1.0]],
    'behavior:probabilities': [[0.25, 0.75], [0.5, 0.5]],
    'behavior:rating': [3.0, 7.0],
    'behavior:confidence': [1.0, 5.0],
    'behavior:rt': [0.2, 1.5],
    'behavior:text': ['a cat', 'a dog'],
}


@pytest.mark.parametrize('channel', sorted(WELL_FORMED))
def test_a_well_formed_assembly_is_stored_with_the_full_schema(channel):
    answer = _participate(channel, _rows(channel, WELL_FORMED[channel])).collect(channel)
    assert list(answer['stimulus_id'].values) == IDS
    assert list(answer['response_state'].values) == ['valid', 'valid']
    assert list(answer['response_reason'].values) == ['', '']
    assert len(answer['raw_answer'].values) == 2
    assert answer.attrs['response'] == _record(channel)


CELLS = {   # one cell per trial: (cell, the stored value)
    'behavior:choice': ('dog', [0.0, 1.0]),
    'behavior:probabilities': (np.array([0.25, 0.75]), [0.25, 0.75]),
    'behavior:rating': (4, 4.0),
    'behavior:confidence': (2.0, 2.0),
    'behavior:rt': (0.5, 0.5),
    'behavior:text': ('a cat', 'a cat'),
}


@pytest.mark.parametrize('channel', sorted(CELLS))
def test_one_cell_per_trial_is_stored_with_the_full_schema(channel):
    cell, stored = CELLS[channel]
    session = stimulus_session(_stimuli(), [channel], TASKS[channel])
    meta = {'units': 'ms'} if channel == 'behavior:rt' else {}
    for stimulus_id in IDS:
        session.respond(StreamEvent(channel, cell, None, {'stimulus_id': stimulus_id, **meta}))
    answer = session.collect(channel)
    assert list(answer['stimulus_id'].values) == IDS
    assert list(answer['response_state'].values) == ['valid', 'valid']
    assert answer.attrs['response']['method'] == 'external'
    assert answer.attrs['response']['origin'] == 'native'
    if channel == 'behavior:text':
        assert list(answer.values) == [stored, stored]
    else:
        np.testing.assert_array_equal(answer.values[0], stored)


MALFORMED = [
    ('behavior:choice', 'wrong dims', lambda: _rows('behavior:choice', [1.0, 0.0], dtype=np.float32)),
    ('behavior:choice', 'float64 values', lambda: _rows('behavior:choice', [[1, 0], [0, 1]],
                                                        dtype=np.float64)),
    ('behavior:choice', 'a soft row', lambda: _rows('behavior:choice', [[0.6, 0.4], [0, 1]])),
    ('behavior:choice', 'a row with NaN and a value', lambda: _rows('behavior:choice',
                                                                     [[np.nan, 1], [0, 1]])),
    ('behavior:choice', 'options that are not the task options',
     lambda: _rows('behavior:choice', [[1, 0], [0, 1]], axis=('dog', 'cat'))),
    ('behavior:choice', 'three options for two', lambda: _rows('behavior:choice', [[1, 0, 0], [0, 1, 0]],
                                                               axis=('cat', 'dog', 'cow'))),
    ('behavior:choice', 'a row of another stimulus', lambda: _rows('behavior:choice', [[1, 0], [0, 1]],
                                                                   ids=['s0', 'other'])),
    ('behavior:choice', 'no record', lambda: _rows('behavior:choice', [[1, 0], [0, 1]], record=None)),
    ('behavior:choice', 'a contradictory record', lambda: _rows(
        'behavior:choice', [[1, 0], [0, 1]], record={**_record('behavior:choice'), 'origin': 'derived'})),
    ('behavior:choice', 'a supplied valid state on an empty row', lambda: _rows(
        'behavior:choice', [[np.nan, np.nan], [0, 1]], response_state=['valid', 'valid'])),
    ('behavior:choice', 'a 1-D array cell', lambda: np.array([1.0, 0.0])),
    ('behavior:probabilities', 'wrong dims', lambda: _rows('behavior:probabilities', [0.5, 0.5],
                                                           dtype=np.float32)),
    ('behavior:probabilities', 'a label', lambda: 'cat'),
    ('behavior:probabilities', 'a 2-D array', lambda: np.zeros((2, 2))),
    ('behavior:probabilities', 'float64 values without the raw-score record',
     lambda: _rows('behavior:probabilities', [[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)),
    ('behavior:rating', 'two axes', lambda: _rows('behavior:rating', [[1.0, 2.0], [3.0, 4.0]])),
    ('behavior:rating', 'a label', lambda: 'seven'),
    ('behavior:rating', 'float32 values', lambda: _rows('behavior:rating', [1.0, 2.0],
                                                        dtype=np.float32)),
    ('behavior:confidence', 'a label', lambda: 'sure'),
    ('behavior:rt', 'no units', lambda: _rows('behavior:rt', [0.1, 0.2],
                                              record=response_record('behavior:rt', 'external'))),
    ('behavior:rt', 'unknown units', lambda: _rows('behavior:rt', [0.1, 0.2], record=response_record(
        'behavior:rt', 'external', parameters={'units': 's'}))),
    ('behavior:text', 'numbers', lambda: _rows('behavior:text', [1.0, 2.0], dtype=np.float64)),
    ('behavior:text', 'a value that is not a str', lambda: _rows('behavior:text', [3, 'a dog'])),
    ('behavior:text', 'an array', lambda: np.array([1.0])),
]


@pytest.mark.parametrize('channel, name, make', MALFORMED, ids=[f'{c}-{n}' for c, n, _ in MALFORMED])
def test_a_malformed_response_raises_through_participate_and_is_not_kept(channel, name, make):
    session = stimulus_session(_stimuli(), [channel], TASKS[channel])
    with pytest.raises(ValueError, match='failed the channel catalog'):
        _Stub(channel, make()).participate(session)
    assert session.emitted == []


def test_a_cell_for_another_trial_than_its_event_names_raises():
    session = stimulus_session(_stimuli(), ['behavior:choice'], TASKS['behavior:choice'])
    one_row = _rows('behavior:choice', [[1, 0]], ids=['s1'])
    with pytest.raises(ValueError, match='answers stimulus'):
        session.respond(StreamEvent('behavior:choice', one_row, None, {'stimulus_id': 's0'}))


def test_a_reaction_time_cell_without_units_raises():
    session = stimulus_session(_stimuli(), ['behavior:rt'], TASKS['behavior:rt'])
    with pytest.raises(ValueError, match='units'):
        session.respond(StreamEvent('behavior:rt', 0.5, None, {'stimulus_id': 's0'}))


INVALID = [
    ('behavior:choice', [[np.nan, np.nan], [0, 1]], 'no option chosen'),
    ('behavior:choice', [[1, 1], [0, 1]], 'not exactly one option'),
    ('behavior:choice', [[0, 0], [0, 1]], 'not exactly one option'),
    ('behavior:probabilities', [[-0.1, 1.1], [0.5, 0.5]], 'negative'),
    ('behavior:probabilities', [[np.nan, 0.5], [0.5, 0.5]], 'not finite'),
    ('behavior:probabilities', [[0.5, 0.5002], [0.5, 0.5]], 'sum to'),
    ('behavior:rating', [9.0, 3.0], 'outside'),
    ('behavior:rating', [1.5, 3.0], 'scale step'),
    ('behavior:rating', [np.nan, 3.0], 'no number'),
    ('behavior:confidence', [0.0, 3.0], 'outside'),
    ('behavior:rt', [-1.0, 0.5], 'negative'),
    ('behavior:rt', [np.inf, 0.5], 'not finite'),
]


@pytest.mark.parametrize('channel, values, reason', INVALID)
def test_an_invalid_response_is_stored_as_an_invalid_trial(channel, values, reason):
    answer = _participate(channel, _rows(channel, values)).collect(channel)
    assert list(answer['response_state'].values) == ['invalid', 'valid']
    assert reason in answer['response_reason'].values[0]
    np.testing.assert_array_equal(answer.values, np.array(values, dtype=answer.values.dtype))


def test_a_text_the_model_side_could_not_read_is_invalid_with_its_reason():
    payload = _rows('behavior:text', ['', 'a dog'], answer_valid=[False, True],
                    parse_error=['the reply is not text', ''])
    answer = _participate('behavior:text', payload).collect('behavior:text')
    assert list(answer['response_state'].values) == ['invalid', 'valid']
    assert answer['response_reason'].values[0] == 'the reply is not text'


def test_probability_rows_off_by_less_than_one_in_ten_thousand_are_valid():
    payload = _rows('behavior:probabilities', [[0.5, 0.50009], [0.49991, 0.5]])
    answer = _participate('behavior:probabilities', payload).collect('behavior:probabilities')
    assert list(answer['response_state'].values) == ['valid', 'valid']


def test_raw_log_scores_need_the_explicit_record():
    record = response_record('behavior:probabilities', 'likelihood', category='model-derived',
                             parameters={'scale': 'log'})
    payload = _rows('behavior:probabilities', [[-0.5, -2.0], [0.5, -1.0]], dtype=np.float64,
                    record=record)
    answer = _participate('behavior:probabilities', payload).collect('behavior:probabilities')
    assert list(answer['response_state'].values) == ['valid', 'invalid']
    # Without scale 'log', model-derived values are probabilities and float32.
    other = response_record('behavior:probabilities', 'likelihood', category='model-derived')
    session = stimulus_session(_stimuli(), ['behavior:probabilities'],
                               TASKS['behavior:probabilities'])
    with pytest.raises(ValueError, match='float64'):
        _Stub('behavior:probabilities', _rows('behavior:probabilities', [[-0.5, -2.0], [0.5, -1.0]],
                                              dtype=np.float64, record=other)).participate(session)


def test_a_supplied_state_that_agrees_is_kept():
    payload = _rows('behavior:choice', [[np.nan, np.nan], [0, 1]],
                    response_state=['invalid', 'valid'])
    answer = _participate('behavior:choice', payload).collect('behavior:choice')
    assert list(answer['response_state'].values) == ['invalid', 'valid']


def test_a_record_with_its_fields_in_another_order_is_accepted():
    record = dict(reversed(list(_record('behavior:choice').items())))
    answer = _participate('behavior:choice', _rows('behavior:choice', [[1, 0], [0, 1]],
                                                   record=record)).collect('behavior:choice')
    assert list(answer['response_state'].values) == ['valid', 'valid']


def test_per_row_options_use_positions():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': IDS, 'text': ['x', 'y'],
                                        'options': [('a', 'b'), ('c', 'd')]}))
    stimuli.identifier = 't1-options'
    task = TaskContext(response=Choice('options'))
    session = stimulus_session(stimuli, ['behavior:choice'], task)
    session.respond(StreamEvent('behavior:choice', 'd', None, {'stimulus_id': 's1',
                                                               'stimulus_index': 1}))
    with pytest.raises(ValueError, match='choice axis'):
        session.respond(StreamEvent('behavior:choice', _rows('behavior:choice', [[1, 0]], ids=['s0']),
                                    None))
    session.respond(StreamEvent('behavior:choice', _rows('behavior:choice', [[1, 0]], ids=['s0'],
                                                         axis=(0, 1)), None))
    answer = session.collect('behavior:choice')
    assert list(answer['choice'].values) == [0, 1]
    rows = {i: row for i, row in zip(answer['stimulus_id'].values, answer.values)}
    np.testing.assert_array_equal(rows['s0'], [1, 0])
    np.testing.assert_array_equal(rows['s1'], [0, 1])
