"""Conformance test T3 (Response Format Standard, section 11.1): the conversions.

``argmax``, ``softmax`` and ``softmax_argmax`` are deterministic (twice in one
process and in a second process) and follow their documented numerics: ties
go to the first column, the column order is the caller's, softmax loses each
row's offset, a ``'bits'`` source is multiplied by ln 2 first. Validity is kept
through each conversion: an invalid source row gives an invalid result. A
derived answer carries a record that names its conversion; an answer without
a record raises.
"""
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.response_formats import argmax, response_record, softmax, softmax_argmax
from brainscore_core.responses import Choice
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

LOGP = np.array([[-1.0, -2.0, -3.0],
                 [-2.0, -2.0, -5.0],      # a tie between the first two options
                 [-7.0, -1.0, -1.0],      # a tie between the last two options
                 [-np.inf, -1.0, -3.0],   # -inf: not finite, so the row is invalid
                 [np.nan, -1.0, -3.0],
                 [-np.inf, -np.inf, -np.inf],
                 [np.inf, -1.0, 0.0]])


def test_argmax_ties_go_to_the_first_column_of_the_given_order():
    indices, valid = argmax(np.array([[0.5, 0.5], [0.2, 0.8], [0.4, 0.6]]))
    assert indices.tolist() == [0, 1, 1] and valid.tolist() == [True, True, True]
    indices, _ = argmax(np.array([[0.5, 0.5]])[:, ::-1])
    assert indices.tolist() == [0]


@pytest.mark.parametrize('row', [[np.nan, 0.5], [np.inf, 0.0], [0.5, -np.inf],
                                 [-0.2, 1.2], [0.6, 0.6], [0.3, 0.3]])
def test_argmax_of_a_row_that_is_not_a_probability_row_has_no_choice(row):
    indices, valid = argmax(np.array([row, [0.1, 0.9]]))
    assert indices.tolist() == [-1, 1] and valid.tolist() == [False, True]


def test_argmax_accepts_a_sum_within_one_in_ten_thousand():
    _, valid = argmax(np.array([[0.5, 0.50009], [0.5, 0.5002]]))
    assert valid.tolist() == [True, False]


def test_argmax_takes_a_trials_by_options_array():
    with pytest.raises(ValueError, match='trials, options'):
        argmax(np.array([0.5, 0.5]))


def test_softmax_is_temperature_one_over_the_listed_options():
    p, valid = softmax(LOGP[:3])
    np.testing.assert_allclose(p[0], np.exp([-1, -2, -3]) / np.exp([-1, -2, -3]).sum())
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert valid.all()


def test_softmax_loses_the_offset_of_each_row():
    np.testing.assert_allclose(softmax(LOGP[:3] + 100.0)[0], softmax(LOGP[:3])[0])


def test_a_non_finite_source_row_is_invalid_also_with_minus_infinity():
    p, valid = softmax(LOGP)
    assert valid.tolist() == [True, True, True, False, False, False, False]
    assert p[3, 0] == 0.0 and np.isclose(p[3].sum(), 1.0)   # the values are computed as before


def test_bits_are_multiplied_by_ln2_first():
    p, _ = softmax(np.array([[-1.0, -2.0]]), units='bits')
    np.testing.assert_allclose(p[0], [2 / 3, 1 / 3])
    p, _ = softmax(np.array([[-1.0, -2.0]]))
    np.testing.assert_allclose(p[0], [math.e / (math.e + 1), 1 / (math.e + 1)])
    with pytest.raises(ValueError, match='units'):
        softmax(np.array([[-1.0, -2.0]]), units='decibans')


def test_softmax_argmax_keeps_the_source_validity():
    indices, valid = softmax_argmax(LOGP)
    assert indices.tolist() == [0, 0, 1, -1, -1, -1, -1]
    assert valid.tolist() == [True, True, True, False, False, False, False]
    assert softmax_argmax(np.array([[-1.0, -2.0]]), units='bits')[0].tolist() == [0]


def test_the_conversions_are_deterministic_within_and_across_processes():
    first = (*softmax(LOGP), *softmax_argmax(LOGP))
    second = (*softmax(LOGP), *softmax_argmax(LOGP))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    code = ('import json, numpy as np; from brainscore_core.response_formats import softmax, '
            'softmax_argmax; x = np.array(json.loads(input()), dtype=float); '
            'p, v = softmax(x); i, w = softmax_argmax(x); '
            'print(json.dumps([[[repr(e) for e in r] for r in p.tolist()], v.tolist(), '
            'i.tolist(), w.tolist()]))')
    payload = json.dumps(LOGP.tolist())      # NaN and Infinity are kept by json
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], input=payload, text=True,
                         capture_output=True, check=True).stdout
    probabilities, valid, indices, chosen = json.loads(out)
    assert probabilities == [[repr(e) for e in r] for r in softmax(LOGP)[0].tolist()]
    assert valid == softmax(LOGP)[1].tolist()
    assert indices == softmax_argmax(LOGP)[0].tolist() and chosen == softmax_argmax(LOGP)[1].tolist()


@pytest.mark.parametrize('conversion, source, parameters', [
    ('argmax', 'behavior:probabilities', {'tie': 'first', 'option_order': 'trial'}),
    ('softmax', 'behavior:log_probabilities',
     {'temperature': 1, 'option_order': 'trial', 'units': 'nats'}),
    ('softmax_argmax', 'behavior:log_probabilities',
     {'temperature': 1, 'tie': 'first', 'option_order': 'trial', 'units': 'nats'}),
])
def test_a_derived_answer_carries_its_conversion_record(conversion, source, parameters):
    record = response_record('behavior:choice', 'likelihood', conversion=conversion,
                             source_format=source)
    assert record['origin'] == 'derived' and record['conversion'] == conversion
    assert record['source_format'] == source and record['parameters'] == parameters
    json.dumps(record)        # serializable


@pytest.mark.parametrize('arguments, match', [
    (dict(conversion='argmax'), 'source_format'),
    (dict(source_format='behavior:probabilities'), 'source_format'),
    (dict(conversion='K1', source_format='x'), 'conversion'),
    (dict(parsed_by='bad'), 'parsed_by'),
    (dict(parameters={'f': object()}), 'parameters'),
    (dict(parameters={'x': float('nan')}), 'serializable'),
])
def test_an_inconsistent_record_is_refused(arguments, match):
    with pytest.raises(ValueError, match=match):
        response_record('behavior:choice', 'readout', **arguments)


def test_a_link_is_derived_and_names_its_source():
    record = response_record('behavior:rt', 'link', source_format='behavior:probabilities',
                             parameters={'link': 'surprisal', 'units': 'monotone'})
    assert record['origin'] == 'derived' and record['conversion'] is None
    with pytest.raises(ValueError, match='source_format'):
        response_record('behavior:rt', 'link')


def test_a_native_answer_has_no_conversion():
    record = response_record('behavior:text', 'generation')
    assert record['origin'] == 'native' and record['conversion'] is None
    assert record['parameters'] == {}


class _Converter(Subject):
    """A custom subject that converts its own scores with core's softmax_argmax."""
    identifier = 'converter'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}
    required_channels = set()

    def __init__(self, with_record):
        self.with_record = with_record

    def participate(self, session):
        while session.next_input() is not None:
            pass
        indices, valid = softmax_argmax(np.array([[-1.0, -2.0], [-3.0, -0.5]]))
        values = np.zeros((2, 2), dtype=np.float32)
        values[np.arange(2), indices] = 1
        attrs = ({'response': response_record(
            'behavior:choice', 'external', conversion='softmax_argmax',
            source_format='behavior:log_probabilities')} if self.with_record else {})
        session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
            values, dims=['presentation', 'choice'], attrs=attrs,
            coords={'stimulus_id': ('presentation', ['s0', 's1']), 'choice': ['cat', 'dog']}),
            None))

    def reset(self):
        pass


def _session():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['s0', 's1'], 'text': ['x', 'y']}))
    stimuli.identifier = 't3-stimuli'
    return stimulus_session(stimuli, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))


def test_a_custom_subject_with_a_derived_record_is_stored():
    session = _session()
    _Converter(with_record=True).participate(session)
    answer = session.collect('behavior:choice')
    assert answer.attrs['response']['conversion'] == 'softmax_argmax'
    np.testing.assert_array_equal(answer.values, [[1, 0], [0, 1]])


def test_a_custom_subject_that_omits_its_derived_record_is_refused():
    session = _session()
    with pytest.raises(ValueError, match='response record'):
        _Converter(with_record=False).participate(session)
    assert session.emitted == []
