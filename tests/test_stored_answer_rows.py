"""A session stores an answer given as a cell as its checked row; ``collect`` gives the table.

An answer given as an assembly stays an assembly on its event. A bad answer is refused
when it is given: nothing is stored and no observer sees it.
"""
import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

CHOICE = 'behavior:choice'


def _choice_session():
    return Session([], [CHOICE], TaskContext(response=Choice(['cat', 'dog'])))


def test_a_cell_answer_event_carries_its_checked_row():
    session = _choice_session()
    session.respond(StreamEvent(CHOICE, 'dog', None, {'stimulus_id': 's1'}))
    row = session.emitted[0].payload
    assert row['stimulus_id'] == 's1' and row['choice'] == ['cat', 'dog']
    np.testing.assert_array_equal(row['values'], [0, 1])
    assert (row['raw_answer'], row['response_state'], row['response']['format']) == ('dog', 'valid', CHOICE)


def test_collect_gives_one_table_of_the_rows_in_arrival_order():
    session = _choice_session()
    session.respond(StreamEvent(CHOICE, 'dog', None, {'stimulus_id': 's1'}))
    session.respond(StreamEvent(CHOICE, 'eel', None, {'stimulus_id': 's0'}))     # not an option
    answer = session.collect(CHOICE)
    assert isinstance(answer, BehavioralAssembly) and answer.dims == ('presentation', 'choice')
    assert list(answer['stimulus_id'].values) == ['s1', 's0']
    assert list(answer['response_state'].values) == ['valid', 'invalid']
    np.testing.assert_array_equal(answer.values[0], [0, 1])
    assert np.isnan(answer.values[1]).all()
    assert answer.attrs['response']['format'] == CHOICE


def test_an_answer_given_as_an_assembly_stays_an_assembly_on_its_event():
    session = _choice_session()
    given = BehavioralAssembly(np.array([[1, 0]], dtype=np.float32), dims=['presentation', 'choice'],
                               coords={'stimulus_id': ('presentation', ['s0']), 'choice': ['cat', 'dog']},
                               attrs={'response': response_record(CHOICE, 'external')})
    session.respond(StreamEvent(CHOICE, given, None))
    assert isinstance(session.emitted[0].payload, BehavioralAssembly)


@pytest.mark.parametrize('stimulus_id, error', [(['s0'], ValueError), ({'a': 1}, TypeError),
                                                (np.datetime64('NaT'), TypeError)])
def test_a_stimulus_id_the_table_refuses_is_refused_when_given(stimulus_id, error):
    session, observed = Session([], ['behavior:text'], TaskContext(response=Text())), []
    with session.observe(on_output=observed.append):
        with pytest.raises(error):
            session.respond(StreamEvent('behavior:text', 'hello', None, {'stimulus_id': stimulus_id}))
    assert session.emitted == [] and observed == []
