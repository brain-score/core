"""Conformance test T1 (Response Format Standard, section 11.1): the schema of each format.

For each behavioral format a fake subject gives one well-formed answer and
several malformed ones through ``Session.respond``. A well-formed answer is
kept, and an invalid value in it is recorded as an invalid trial, not refused.
A malformed payload (wrong type, wrong number of dimensions, wrong coordinate
layout, a malformed response record) raises and nothing is kept.
"""
import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import PROBABILITY_TOLERANCE, response_record
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

IDS = ['s0', 's1']


def _rows(channel, values, **attrs):
    """An answer over two trials: (presentation, choice) or (presentation,)."""
    values = np.asarray(values)
    coords = {'stimulus_id': ('presentation', IDS[:len(values)])}
    dims = ['presentation']
    if values.ndim == 2:
        coords['choice'] = ['cat', 'dog'][:values.shape[1]] + [str(i) for i in range(2, values.shape[1])]
        dims.append('choice')
    return BehavioralAssembly(values, coords=coords, dims=dims, attrs=attrs)


def _record(channel):
    return response_record(channel, 'external')


TASKS = {
    'behavior:choice': TaskContext(response=Choice(['cat', 'dog'])),
    'behavior:probabilities': TaskContext(response=Choice(['cat', 'dog'], probabilities=True)),
    'behavior:rating': TaskContext(response=Scale(1, 7)),
    'behavior:confidence': TaskContext(response=Choice(['cat', 'dog'], confidence=Scale(1, 5))),
    'behavior:rt': TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal')),
    'behavior:text': TaskContext(response=Text()),
}

# A well-formed answer of each format whose trials are all valid.
WELL_FORMED = {
    'behavior:choice': [[1.0, 0.0], [0.0, 1.0]],
    'behavior:probabilities': [[0.25, 0.75], [0.5, 0.5]],
    'behavior:rating': [3.0, 7.0],
    'behavior:confidence': [1.0, 5.0],
    'behavior:rt': [0.2, 1.5],
    'behavior:text': np.array(['a cat', 'a dog'], dtype=object),
}

# Malformed payloads: (name, payload). Each must raise.
MALFORMED = {
    'behavior:choice': [
        ('wrong dims', lambda: _rows('c', [1.0, 0.0])),
        ('transposed layout', lambda: _rows('c', [[1.0, 0.0], [0.0, 1.0]]).transpose()),
        ('2-D array', lambda: np.zeros((2, 2))),
        ('text values', lambda: _rows('c', np.array([['a', 'b'], ['c', 'd']], dtype=object))),
    ],
    'behavior:probabilities': [
        ('wrong dims', lambda: _rows('c', [0.5, 0.5])),
        ('a label', lambda: 'cat'),
        ('3-D array', lambda: np.zeros((1, 2, 2))),
    ],
    'behavior:rating': [
        ('two axes', lambda: _rows('c', [[1.0, 2.0], [3.0, 4.0]])),
        ('a label', lambda: 'seven'),
        ('2-D array', lambda: np.zeros((2, 2))),
    ],
    'behavior:confidence': [
        ('two axes', lambda: _rows('c', [[1.0, 2.0], [3.0, 4.0]])),
        ('a label', lambda: 'sure'),
    ],
    'behavior:rt': [
        ('two axes', lambda: _rows('c', [[1.0, 2.0], [3.0, 4.0]])),
        ('a label', lambda: 'slow'),
    ],
    'behavior:text': [
        ('two axes', lambda: _rows('c', np.array([['a', 'b'], ['c', 'd']], dtype=object))),
        ('numbers', lambda: _rows('c', [1.0, 2.0])),
        ('an array', lambda: np.array([1.0])),
    ],
}

# Invalid responses (well-formed, not usable): (values, reason fragment). Each is kept.
INVALID = {
    'behavior:choice': [([[np.nan, np.nan], [0.0, 1.0]], 'no option chosen'),
                        ([[1.0, 1.0], [0.0, 1.0]], 'not exactly one option')],
    'behavior:probabilities': [([[-0.1, 1.1], [0.5, 0.5]], 'negative'),
                               ([[np.nan, 0.5], [0.5, 0.5]], 'not finite'),
                               ([[0.6, 0.6], [0.5, 0.5]], 'sum to')],
    'behavior:rating': [([9.0, 3.0], 'outside'), ([np.nan, 3.0], 'no number')],
    'behavior:confidence': [([0.0, 3.0], 'outside'), ([np.inf, 3.0], 'no number')],
    'behavior:rt': [([-1.0, 0.5], 'negative'), ([np.nan, 0.5], 'not finite')],
    'behavior:text': [(np.array([3, 'a dog'], dtype=object), 'not a string')],
}


def _respond(channel, payload):
    session = Session((), [channel], TASKS[channel])
    session.respond(StreamEvent(channel, payload, None))
    return session


@pytest.mark.parametrize('channel', sorted(WELL_FORMED))
def test_a_well_formed_answer_is_kept_with_valid_states(channel):
    session = _respond(channel, _rows(channel, WELL_FORMED[channel], response=_record(channel)))
    answer = session.collect(channel)
    assert list(answer['response_state'].values) == ['valid', 'valid']
    assert list(answer['response_reason'].values) == ['', '']
    assert answer.attrs['response'] == _record(channel)


@pytest.mark.parametrize('channel, name, make',
                         [(c, n, m) for c, cases in sorted(MALFORMED.items()) for n, m in cases],
                         ids=lambda value: value if isinstance(value, str) else '')
def test_a_malformed_payload_raises_and_is_not_kept(channel, name, make):
    session = Session((), [channel], TASKS[channel])
    with pytest.raises(ValueError, match='failed the channel catalog'):
        session.respond(StreamEvent(channel, make(), None))
    assert session.emitted == []


@pytest.mark.parametrize('channel', sorted(WELL_FORMED))
def test_a_malformed_record_raises(channel):
    record = dict(_record(channel), format='behavior:other')
    session = Session((), [channel], TASKS[channel])
    with pytest.raises(ValueError, match='names format'):
        session.respond(StreamEvent(channel, _rows(channel, WELL_FORMED[channel], response=record),
                                    None))
    with pytest.raises(ValueError, match='fields'):
        session.respond(StreamEvent(channel, _rows(channel, WELL_FORMED[channel],
                                                   response={'kind': 'reported'}), None))
    assert session.emitted == []


@pytest.mark.parametrize('channel, values, reason',
                         [(c, v, r) for c, cases in sorted(INVALID.items()) for v, r in cases])
def test_an_invalid_response_is_recorded_not_refused(channel, values, reason):
    session = _respond(channel, _rows(channel, values, response=_record(channel)))
    answer = session.collect(channel)
    assert list(answer['response_state'].values) == ['invalid', 'valid']
    assert reason in answer['response_reason'].values[0]
    # The values are kept as given.
    if channel == 'behavior:text':
        assert list(answer.values) == list(values)
    else:
        np.testing.assert_array_equal(answer.values, np.asarray(values, dtype=float))


def test_probabilities_within_the_tolerance_are_valid():
    off = PROBABILITY_TOLERANCE / 2
    session = _respond('behavior:probabilities', _rows('p', [[0.5 + off, 0.5], [0.5, 0.5 - off]]))
    assert list(session.collect('behavior:probabilities')['response_state'].values) == ['valid'] * 2


def test_log_scale_scores_are_valid_when_finite_and_not_above_zero():
    record = response_record('behavior:probabilities', 'likelihood', category='model-derived',
                             parameters={'scale': 'log'})
    session = _respond('behavior:probabilities', _rows('p', [[-0.5, -2.0], [0.5, -1.0]],
                                                       response=record))
    answer = session.collect('behavior:probabilities')
    assert list(answer['response_state'].values) == ['valid', 'invalid']


@pytest.mark.parametrize('channel, payload', [('behavior:choice', 'cat'),
                                              ('behavior:text', 'a cat'),
                                              ('behavior:probabilities', np.array([0.4, 0.6]))])
def test_one_trial_payloads_of_the_event_form_are_accepted(channel, payload):
    session = _respond(channel, payload)
    assert len(session.emitted) == 1
