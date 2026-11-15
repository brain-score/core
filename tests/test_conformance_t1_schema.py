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
    'behavior:probabilities': TaskContext(response=Choice(['cat', 'dog'])),
    'behavior:rating': TaskContext(response=Scale(1, 7, 1)),
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
    with pytest.raises(ValueError, match='answers stimuli'):
        session.respond(StreamEvent('behavior:choice', one_row, None, {'stimulus_id': 's0'}))


def test_a_malformed_cell_names_its_channel_once():
    session = stimulus_session(_stimuli(), ['behavior:choice'])   # no task: no options
    with pytest.raises(ValueError) as raised:
        session.respond(StreamEvent('behavior:choice', 'a', None, {'stimulus_id': 's0'}))
    assert str(raised.value) == (
        "output event on 'behavior:choice' failed the channel catalog: the options of this "
        "trial are not known; send an assembly with a choice axis or give the task a Choice")
    session = stimulus_session(_stimuli(), ['behavior:rating'], TASKS['behavior:rating'])
    with pytest.raises(ValueError) as raised:
        session.respond(StreamEvent('behavior:rating', 'seven', None, {'stimulus_id': 's0'}))
    assert str(raised.value) == ("output event on 'behavior:rating' failed the channel catalog: "
                                 "a cell is one number, not str")


def test_the_catalog_names_the_meta_keys_of_a_cell_answer():
    from brainscore_core import io_catalog
    entry = io_catalog.get('behavior')
    assert {'stimulus_id', 'units'} <= set(entry.meta_keys)
    assert "meta['stimulus_id']" in entry.payload_contract and "meta['units']" in entry.payload_contract


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


@pytest.mark.parametrize('parameters', [{'scale': 'log'}, {}], ids=['old raw-score record', 'plain'])
def test_log_values_on_the_probabilities_format_are_refused(parameters):
    # The temporary raw-score path is gone: probabilities are float32 rows, and log
    # probabilities travel on behavior:log_probabilities.
    record = response_record('behavior:probabilities', 'likelihood', category='model-derived',
                             parameters=parameters)
    session = stimulus_session(_stimuli(), ['behavior:probabilities'],
                               TASKS['behavior:probabilities'])
    with pytest.raises(ValueError, match='float64, not float32'):
        _Stub('behavior:probabilities', _rows('behavior:probabilities', [[-0.5, -2.0], [0.5, -1.0]],
                                              dtype=np.float64, record=record)).participate(session)
    float32 = _rows('behavior:probabilities', [[-0.5, -2.0], [0.5, -1.0]], record=record)
    answer = _participate('behavior:probabilities', float32).collect('behavior:probabilities')
    assert list(answer['response_state'].values) == ['invalid', 'invalid']   # negative values


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


# --- behavior:log_probabilities (F6; standard 2.2 and 2.3) ---------------------

F6 = 'behavior:log_probabilities'
F6_TASK = TaskContext(response=Choice(('a', 'b')))
TOKENIZATION = {'join': 'space', 'special_tokens': 'none', 'template': 'none', 'truncation': 'none'}
F6_PARAMETERS = {'layout': 'continuation', 'context': 'shared', 'tokenization': TOKENIZATION,
                 'aggregation': 'sum', 'units': 'nats'}


def _f6_record(**changes):
    """A well-formed F6 record as a plain dict, with ``changes`` to its parameters."""
    parameters = {**F6_PARAMETERS, **changes}
    return {'format': F6, 'category': 'model-derived', 'origin': 'native', 'method': 'likelihood',
            'conversion': None, 'source_format': None, 'parameters': parameters,
            'parsed_by': None}


def _f6(values, ids=('s0',), axis=('a', 'b'), dtype=np.float64, record=None):
    return BehavioralAssembly(
        np.array(values, dtype=dtype), dims=['presentation', 'choice'],
        attrs={'response': record if record is not None else _f6_record()},
        coords={'stimulus_id': ('presentation', list(ids)), 'choice': list(axis)})


def _f6_table(ids=('s0',)):
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(ids), 'text': ['x'] * len(ids)}))
    stimuli.identifier = 'f6-' + '-'.join(ids)
    return stimuli


def _f6_answer(payload, table=True):
    session = (stimulus_session(_f6_table(), [F6], F6_TASK) if table
               else Session((), [F6]))
    session.respond(StreamEvent(F6, payload, None))
    return session.collect(F6)


def test_the_f6_record_is_built_by_response_record():
    assert response_record(F6, 'likelihood', category='model-derived',
                           parameters=F6_PARAMETERS) == _f6_record()


@pytest.mark.parametrize('table', [True, False], ids=['stimulus_session', 'Session'])
def test_a_well_formed_f6_assembly_is_stored_valid_in_float64(table):
    answer = _f6_answer(_f6([[-0.1, -2.3]]), table)
    assert answer.dims == ('presentation', 'choice')
    assert answer.values.dtype == np.float64
    np.testing.assert_array_equal(answer.values, [[-0.1, -2.3]])
    assert list(answer['choice'].values) == ['a', 'b']
    assert list(answer['stimulus_id'].values) == ['s0']
    assert list(answer['response_state'].values) == ['valid']
    assert list(answer['response_reason'].values) == ['']
    assert list(answer['raw_answer'].values) == ['[-0.1, -2.3]']
    assert answer.attrs['response'] == _f6_record()


@pytest.mark.parametrize('values, state, reason', [
    ([[0.5, -1.0]], 'invalid', 'a log probability is above 0'),
    ([[5e-7, -1.0]], 'valid', ''),
    ([[np.nan, -1.0]], 'invalid', 'a value is not finite'),
    ([[np.inf, -1.0]], 'invalid', 'a value is not finite'),
    ([[-np.inf, -1.0]], 'invalid', 'a value is not finite'),
])
@pytest.mark.parametrize('table', [True, False], ids=['stimulus_session', 'Session'])
def test_an_f6_value_that_is_not_a_log_probability_is_stored_invalid(values, state, reason, table):
    answer = _f6_answer(_f6(values), table)
    assert list(answer['response_state'].values) == [state]
    assert list(answer['response_reason'].values) == [reason]
    assert answer.values.dtype == np.float64


@pytest.mark.parametrize('units, value, state', [
    ('nats', 1e-6, 'valid'),          # the tolerance itself
    ('nats', 1.1e-6, 'invalid'),
    ('bits', 1.4e-6, 'valid'),        # 1.4e-6 bits = 9.7e-7 nats
    ('bits', 1.5e-6, 'invalid'),      # 1.5e-6 bits = 1.04e-6 nats
    ('bits', 1.1e-6, 'valid'),        # above 1e-6 as a number, below it in nats
])
def test_the_positive_tolerance_is_one_millionth_of_a_nat_in_both_units(units, value, state):
    answer = _f6_answer(_f6([[value, -1.0]], record=_f6_record(units=units)))
    assert list(answer['response_state'].values) == [state]


def _without(name):
    record = _f6_record()
    del record['parameters'][name]
    return record


def _tokenization(**changes):
    return _f6_record(tokenization={**TOKENIZATION, **changes})


def _tokenization_without(name):
    return _f6_record(tokenization={k: v for k, v in TOKENIZATION.items() if k != name})


def _record_with(**fields):
    return {**_f6_record(), **fields}


GOOD = [[-0.1, -2.3]]
SCHEMA = 'not the behavior:log_probabilities schema'
TOKENS = 'tokenization is a dict with keys'
NATIVE = 'native: it has no conversion'
F6_MALFORMED = [   # (case, the reason it names, payload)
    *[(f'no {name}', SCHEMA, lambda name=name: _f6(GOOD, record=_without(name)))
      for name in F6_PARAMETERS],
    *[(f'no tokenization {name}', TOKENS,
       lambda name=name: _f6(GOOD, record=_tokenization_without(name))) for name in TOKENIZATION],
    ('an extra parameter', SCHEMA, lambda: _f6(GOOD, record=_f6_record(scale='log'))),
    ('layout rows', "layout is 'rows'", lambda: _f6(GOOD, record=_f6_record(layout='rows'))),
    ('units log10', "units is 'log10'", lambda: _f6(GOOD, record=_f6_record(units='log10'))),
    ('aggregation max', "aggregation is 'max'",
     lambda: _f6(GOOD, record=_f6_record(aggregation='max'))),
    ('context other', "context is 'other'", lambda: _f6(GOOD, record=_f6_record(context='other'))),
    ('continuation with prefix context', "continuation layout has context 'shared'",
     lambda: _f6(GOOD, record=_f6_record(context='prefix'))),
    ('segment with shared context', "segment layout has context 'prefix'",
     lambda: _f6(GOOD, record=_f6_record(layout='segment'))),
    ('tokenization not a dict', TOKENS, lambda: _f6(GOOD, record=_f6_record(tokenization='bpe'))),
    ('special tokens other', "special_tokens is 'bos'",
     lambda: _f6(GOOD, record=_tokenization(special_tokens='bos'))),
    ('template other', "template is 'instruct'",
     lambda: _f6(GOOD, record=_tokenization(template='instruct'))),
    ('empty join', 'join is a nonempty str', lambda: _f6(GOOD, record=_tokenization(join=''))),
    ('truncation zero', 'a positive int', lambda: _f6(GOOD, record=_tokenization(truncation=0))),
    ('truncation true', 'a positive int', lambda: _f6(GOOD, record=_tokenization(truncation=True))),
    ('category behavioral', "model-derived, not 'behavioral'",
     lambda: _f6(GOOD, record=_record_with(category='behavioral'))),
    ('a softmax conversion', NATIVE, lambda: _f6(GOOD, record=_record_with(
        conversion='softmax', origin='derived', source_format=F6))),
    ('a link', NATIVE, lambda: _f6(GOOD, record=_record_with(
        method='link', origin='derived', source_format='behavior:probabilities'))),
    ('no record', 'carries its response record', lambda: BehavioralAssembly(
        np.array(GOOD), dims=['presentation', 'choice'],
        coords={'stimulus_id': ('presentation', ['s0']), 'choice': ['a', 'b']})),
    ('float32 values', 'float32, not float64', lambda: _f6(GOOD, dtype=np.float32)),
    ('a cell', 'an assembly with its response record', lambda: np.array([-1.0, -2.0])),
    ('a number', 'an assembly with its response record', lambda: -1.0),
    ('one dimension', r"dims are \('presentation',\)", lambda: BehavioralAssembly(
        np.array([-0.1]), dims=['presentation'], attrs={'response': _f6_record()},
        coords={'stimulus_id': ('presentation', ['s0'])})),
    ('choice coordinates that are not the options', r"choice axis is \['b', 'a'\]",
     lambda: _f6(GOOD, axis=('b', 'a'))),
    ('three continuations for two options', r"choice axis is \['a', 'b', 'c'\]",
     lambda: _f6([[-0.1, -2.3, -3.0]], axis=('a', 'b', 'c'))),
    ('a row for a stimulus not in the table', 'not in this session', lambda: _f6(GOOD, ids=('s9',))),
    # Review S1, F6-03: an unhashable layout is a malformed payload, not a TypeError.
    ('layout a list', "layout is \\[\\]", lambda: _f6(GOOD, record=_f6_record(layout=[]))),
    ('layout a dict', "layout is \\{\\}", lambda: _f6(GOOD, record=_f6_record(layout={}))),
]


@pytest.mark.parametrize('name, reason, make', F6_MALFORMED, ids=[n for n, _, _ in F6_MALFORMED])
def test_a_malformed_f6_response_raises_and_nothing_is_stored(name, reason, make):
    session = stimulus_session(_f6_table(), [F6], F6_TASK)
    with pytest.raises(ValueError, match='failed the channel catalog') as raised:
        session.respond(StreamEvent(F6, make(), None))
    assert raised.match(reason)
    assert session.emitted == []


SEGMENT = {'layout': 'segment', 'context': 'prefix'}


def test_a_segment_answer_has_the_positions_as_its_axis():
    # Review S1, F6-01: the axis of a segment answer is the positions 1 to R, whatever
    # the task's options; other coordinates are refused, also without a task.
    answer = _f6_answer(_f6([[-1.0, -2.0]], axis=(1, 2), record=_f6_record(**SEGMENT)))
    assert list(answer['response_state'].values) == ['valid']
    for table in (True, False):
        with pytest.raises(ValueError, match=r"positions 1 to 2, not \['wrong', 'axis'\]"):
            _f6_answer(_f6([[-1.0, -2.0]], axis=('wrong', 'axis'), record=_f6_record(**SEGMENT)),
                       table=table)


def test_response_record_refuses_a_malformed_f6_record():
    with pytest.raises(ValueError, match='model-derived'):
        response_record(F6, 'likelihood', parameters=F6_PARAMETERS)
    with pytest.raises(ValueError, match='native'):
        response_record(F6, 'likelihood', category='model-derived', conversion='softmax',
                        source_format=F6, parameters=F6_PARAMETERS)
    with pytest.raises(ValueError, match="context 'prefix'"):
        response_record(F6, 'likelihood', category='model-derived',
                        parameters={**F6_PARAMETERS, 'layout': 'segment'})
