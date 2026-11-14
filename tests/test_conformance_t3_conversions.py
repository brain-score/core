"""Conformance test T3 (Response Format Standard, section 11.1): the conversions.

``argmax``, ``softmax`` and ``softmax_argmax`` are deterministic (twice in one
process and in a second process), follow their documented numerics (ties go
to the first column, the column order is the caller's, a non-finite row has no
choice, softmax loses each row's offset), and a derived answer's record names
the conversion, its parameters and its source format.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from brainscore_core.response_formats import (
    RECORD_FIELDS, argmax, record_problems, response_record, softmax, softmax_argmax)

LOGP = np.array([[-1.0, -2.0, -3.0],
                 [-2.0, -2.0, -5.0],      # a tie between the first two options
                 [-7.0, -1.0, -1.0],      # a tie between the last two options
                 [-np.inf, -1.0, -3.0],   # an impossible option: probability 0
                 [np.nan, -1.0, -3.0],    # a non-finite row
                 [-np.inf, -np.inf, -np.inf],
                 [np.inf, -1.0, 0.0]])


def test_argmax_ties_go_to_the_first_column_of_the_given_order():
    indices, valid = argmax(np.array([[0.5, 0.5], [0.2, 0.8], [0.4, 0.4]]))
    assert indices.tolist() == [0, 1, 0] and valid.tolist() == [True, True, True]
    # The order is the caller's: reversed columns give the other option of a tie.
    indices, _ = argmax(np.array([[0.5, 0.5]])[:, ::-1])
    assert indices.tolist() == [0]


@pytest.mark.parametrize('row', [[np.nan, 0.5], [np.inf, 0.0], [0.5, -np.inf]])
def test_argmax_of_a_non_finite_row_has_no_choice(row):
    indices, valid = argmax(np.array([row, [0.1, 0.9]]))
    assert indices.tolist() == [-1, 1] and valid.tolist() == [False, True]


def test_argmax_takes_a_trials_by_options_array():
    with pytest.raises(ValueError, match='trials, options'):
        argmax(np.array([0.5, 0.5]))


def test_softmax_is_temperature_one_over_the_listed_options():
    p = softmax(LOGP[:4])
    np.testing.assert_allclose(p[0], np.exp([-1, -2, -3]) / np.exp([-1, -2, -3]).sum())
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[3, 0] == 0.0      # log probability -inf: probability 0, the row stays valid


def test_softmax_loses_the_offset_of_each_row():
    np.testing.assert_allclose(softmax(LOGP[:3] + 100.0), softmax(LOGP[:3]))


def test_softmax_of_a_row_without_a_finite_maximum_is_nan():
    assert np.isnan(softmax(LOGP[4:])).all()


def test_softmax_argmax_is_argmax_of_softmax():
    indices, valid = softmax_argmax(LOGP)
    assert indices.tolist() == [0, 0, 1, 1, -1, -1, -1]
    assert valid.tolist() == [True, True, True, True, False, False, False]
    expected = argmax(softmax(LOGP))
    np.testing.assert_array_equal(indices, expected[0])


def test_the_conversions_are_deterministic_within_and_across_processes():
    first = (softmax(LOGP), *softmax_argmax(LOGP), *argmax(softmax(LOGP)))
    second = (softmax(LOGP), *softmax_argmax(LOGP), *argmax(softmax(LOGP)))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    code = ('import json, numpy as np; from brainscore_core.response_formats import softmax, '
            'softmax_argmax; x = np.array(json.loads(input()), dtype=float); '
            'i, v = softmax_argmax(x); '
            'print(json.dumps([softmax(x).tolist(), i.tolist(), v.tolist()]))')
    payload = json.dumps(np.where(np.isfinite(LOGP), LOGP, -1e300).tolist())
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], input=payload, text=True,
                         capture_output=True, check=True).stdout
    probabilities, indices, valid = json.loads(out)
    x = np.array(json.loads(payload))
    np.testing.assert_array_equal(np.array(probabilities), softmax(x))
    assert indices == softmax_argmax(x)[0].tolist() and valid == softmax_argmax(x)[1].tolist()


@pytest.mark.parametrize('conversion, source, parameters', [
    ('argmax', 'behavior:probabilities', {'tie': 'first', 'option_order': 'trial'}),
    ('softmax', 'behavior:log_probabilities', {'temperature': 1, 'option_order': 'trial'}),
    ('softmax_argmax', 'behavior:log_probabilities',
     {'temperature': 1, 'tie': 'first', 'option_order': 'trial'}),
])
def test_a_derived_answer_carries_its_conversion_record(conversion, source, parameters):
    record = response_record('behavior:choice', 'likelihood', conversion=conversion,
                             source_format=source)
    assert tuple(record) == RECORD_FIELDS
    assert record['origin'] == 'derived' and record['conversion'] == conversion
    assert record['source_format'] == source and record['parameters'] == parameters
    assert record_problems(record, 'behavior:choice') == []
    json.dumps(record)        # serializable


def test_a_derived_answer_without_a_source_format_is_refused():
    with pytest.raises(ValueError, match='source_format'):
        response_record('behavior:choice', 'readout', conversion='argmax')
    with pytest.raises(ValueError, match='source_format'):
        response_record('behavior:rt', 'link')


def test_an_unknown_conversion_is_refused():
    with pytest.raises(ValueError, match='conversion'):
        response_record('behavior:choice', 'readout', conversion='K1', source_format='x')


def test_a_native_answer_has_no_conversion():
    record = response_record('behavior:text', 'generation')
    assert record['origin'] == 'native' and record['conversion'] is None
    assert record['parameters'] == {}
