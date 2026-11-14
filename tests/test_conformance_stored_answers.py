"""How core stores answers on an addressed behavior channel: rows, order, occurrences, missing trials.

Each test goes through ``Session.respond``, ``collect`` or ``participate``.
A stored answer keeps its rows in the order they arrived. A table row is
answered by occurrence; a table row with no answer is a missing row once the
session is complete. In a session without a table, each ``respond`` signal of
the source requests one answer, also for a subject that does not declare
``control``. Every stored answer, an environment's included, has the full
schema; the environment gets the action as the subject sent it.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming import (Session, Source, StreamEvent, begin_event, end_event,
                                       respond_event)
from brainscore_core.streaming_helpers import environment_session, stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

TEXT = TaskContext(response=Text())
CHOICE = TaskContext(response=Choice(['cat', 'dog']))


def _table(ids, **columns):
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(ids), 'text': ['x'] * len(ids),
                                        **columns}))
    stimuli.identifier = 'stored-' + '-'.join(map(str, ids))
    return stimuli


def _choices(ids, values, axis=('cat', 'dog'), record=None, **coords):
    return BehavioralAssembly(
        np.array(values, dtype=np.float32), dims=['presentation', 'choice'],
        attrs={'response': record or response_record('behavior:choice', 'external')},
        coords={'stimulus_id': ('presentation', list(ids)), 'choice': list(axis),
                **{name: ('presentation', list(column)) for name, column in coords.items()}})


def _texts(ids, values):
    return BehavioralAssembly(
        np.array(values, dtype=object), dims=['presentation'],
        attrs={'response': response_record('behavior:text', 'external')},
        coords={'stimulus_id': ('presentation', list(ids))})


def _drain(session):
    while session.next_input() is not None:
        pass


# --- the rows an event names (N01) ---------------------------------------------

def test_an_event_that_names_fewer_stimuli_than_its_rows_raises():
    session = stimulus_session(_table(['s0', 's1']), ['behavior:choice'], CHOICE)
    with pytest.raises(ValueError, match="answers stimuli"):
        session.respond(StreamEvent('behavior:choice', _choices(['s0', 's1'], [[1, 0], [0, 1]]),
                                    None, {'stimulus_id': ['s0']}))
    assert session.emitted == []


def test_a_later_row_with_another_number_of_options_raises():
    table = _table(['s0', 's1'], options=[('a', 'b'), ('a', 'b', 'c')])
    session = stimulus_session(table, ['behavior:choice'], TaskContext(response=Choice('options')))
    with pytest.raises(ValueError, match="options .* of stimulus 's1'"):
        session.respond(StreamEvent('behavior:choice',
                                    _choices(['s0', 's1'], [[1, 0], [0, 1]], axis=(0, 1)), None))


def test_a_choice_row_supplied_as_invalid_must_be_all_nan():
    session = stimulus_session(_table(['s0']), ['behavior:choice'], CHOICE)
    with pytest.raises(ValueError, match='supplied as invalid is all NaN'):
        session.respond(StreamEvent('behavior:choice', _choices(
            ['s0'], [[0, 0]], response_state=['invalid']), None))


# --- an environment stores the full schema (N02) -------------------------------

class _Conversation:
    """An environment that takes text replies for a number of turns."""

    def __init__(self, turns):
        self.turns, self.heard = turns, []

    def reset(self, *, seed=None, options=None):
        return {'text': 'hello'}

    def step(self, outputs):
        self.heard.append(outputs['behavior:text'])
        done = len(self.heard) >= self.turns
        return {'text': f'turn {len(self.heard)}'}, ('terminal' if done else None)


class _Replier(Subject):
    identifier = 'replier'
    in_channels = {'control', 'text'}
    out_channels = {'behavior:text'}
    required_channels = set()

    def participate(self, session):
        session.declare_inputs(self.in_channels)
        while (event := session.next_input()) is not None:
            if event.channel == 'control' and event.payload == 'respond':
                session.respond(StreamEvent('behavior:text', 'yes', None))

    def reset(self):
        pass


@pytest.mark.parametrize('turns', [1, 2])
def test_an_environment_reply_is_stored_with_state_and_record(turns):
    environment = _Conversation(turns)
    session = environment_session(environment, action_channel='behavior:text', task=TEXT)
    _Replier().participate(session)
    answer = session.collect('behavior:text')
    assert environment.heard == ['yes'] * turns                 # the environment got the string
    assert list(answer.values) == ['yes'] * turns
    assert list(answer['response_state'].values) == ['valid'] * turns
    assert answer.attrs['response']['method'] == 'external'


# --- order and occurrences (N03) -----------------------------------------------

def test_a_batch_keeps_the_order_it_arrived_in_and_misses_nothing():
    session = stimulus_session(_table(['a', 'b']), ['behavior:choice'], CHOICE)
    _drain(session)
    session.respond(StreamEvent('behavior:choice', _choices(['b', 'a'], [[0, 1], [1, 0]]), None))
    batch = session.collect('behavior:choice')
    assert list(batch['stimulus_id'].values) == ['b', 'a']
    assert list(batch['response_state'].values) == ['valid', 'valid']

    single = stimulus_session(_table(['a', 'b']), ['behavior:choice'], CHOICE)
    _drain(single)
    single.respond(StreamEvent('behavior:choice', 'dog', None, {'stimulus_id': 'b'}))
    single.respond(StreamEvent('behavior:choice', 'cat', None, {'stimulus_id': 'a'}))
    rows = single.collect('behavior:choice')
    assert list(rows['stimulus_id'].values) == ['b', 'a']
    np.testing.assert_array_equal(rows.values, batch.values)
    assert list(rows['response_state'].values) == ['valid', 'valid']


def _three_texts(answers):
    session = stimulus_session(_table(['a', 'b', 'a']), ['behavior:text'], TEXT)
    _drain(session)
    for stimulus_id, text in answers:
        session.respond(StreamEvent('behavior:text', text, None, {'stimulus_id': stimulus_id}))
    return session


def test_repeated_stimuli_are_answered_by_occurrence():
    answer = _three_texts([('a', 'first'), ('b', 'second'), ('a', 'third')]).collect('behavior:text')
    assert list(answer['stimulus_id'].values) == ['a', 'b', 'a']
    assert list(answer.values) == ['first', 'second', 'third']
    assert list(answer['response_state'].values) == ['valid'] * 3


def test_the_second_occurrence_with_no_answer_is_one_missing_row():
    answer = _three_texts([('a', 'first'), ('b', 'second')]).collect('behavior:text')
    assert list(answer['stimulus_id'].values) == ['a', 'b', 'a']
    assert list(answer['response_state'].values) == ['valid', 'valid', 'missing']
    assert list(answer.values) == ['first', 'second', None]


def test_more_answers_than_the_table_has_occurrences_raise():
    session = _three_texts([('a', 'first'), ('b', 'second'), ('a', 'third')])
    with pytest.raises(ValueError, match="answered more often than the table has it"):
        session.respond(StreamEvent('behavior:text', 'fourth', None, {'stimulus_id': 'a'}))


def test_a_partial_batch_and_single_answers_give_one_row_per_trial_occurrence():
    session = stimulus_session(_table(['a', 'b', 'a', 'c']), ['behavior:text'], TEXT)
    _drain(session)
    session.respond(StreamEvent('behavior:text', _texts(['a', 'c'], ['one', 'four']), None))
    session.respond(StreamEvent('behavior:text', 'two', None, {'stimulus_id': 'b'}))
    session.respond(StreamEvent('behavior:text', _texts(['a'], ['three']), None))
    answer = session.collect('behavior:text')
    assert list(answer['stimulus_id'].values) == ['a', 'c', 'b', 'a']
    assert list(answer.values) == ['one', 'four', 'two', 'three']
    assert list(answer['response_state'].values) == ['valid'] * 4


def test_collect_before_the_session_is_complete_names_no_future_row_missing():
    session = stimulus_session(_table(['a', 'b', 'c']), ['behavior:text'], TEXT)
    session.next_input()                         # only the first trial has begun
    session.respond(StreamEvent('behavior:text', 'one', None, {'stimulus_id': 'a'}))
    early = session.collect('behavior:text')
    assert list(early['stimulus_id'].values) == ['a']
    assert list(early['response_state'].values) == ['valid']
    _drain(session)                              # complete: the rest are missing now
    late = session.collect('behavior:text')
    assert list(late['response_state'].values) == ['valid', 'missing', 'missing']
    assert session.collect('behavior:text').sizes['presentation'] == 3   # stored once


# --- respond signals without control (N04) --------------------------------------

class _TextOnly(Subject):
    """Declares only text, so it never sees the respond signals; answers 'a' only."""
    identifier = 'text-only'
    in_channels = {'text'}
    out_channels = {'behavior:text'}
    required_channels = set()

    def participate(self, session):
        session.declare_inputs(self.in_channels)
        while (event := session.next_input()) is not None:
            if event.payload == 'a':
                session.respond(StreamEvent('behavior:text', 'answer a', None))

    def reset(self):
        pass


def test_respond_signals_request_answers_from_a_subject_without_control():
    events = [begin_event(), StreamEvent('text', 'a', 0.0), respond_event(),
              StreamEvent('text', 'b', 1.0), respond_event(), end_event('terminal')]
    session = Session(events, ['behavior:text'], TEXT)
    _TextOnly().participate(session)
    answer = session.collect('behavior:text')
    assert list(answer['response_state'].values) == ['valid', 'missing']
    assert list(answer.values) == ['answer a', None]


def test_without_a_table_and_without_respond_signals_nothing_is_missing():
    events = [StreamEvent('text', 'a', 0.0), StreamEvent('text', 'b', 1.0)]
    session = Session(events, ['behavior:text'], TEXT)
    _TextOnly().participate(session)
    answer = session.collect('behavior:text')
    assert list(answer.values) == ['answer a']
    assert list(answer['response_state'].values) == ['valid']


# --- derived evidence (N06) ----------------------------------------------------

def test_a_derived_answer_without_its_source_evidence_raises():
    record = response_record('behavior:choice', 'external', conversion='argmax',
                             source_format='behavior:probabilities')
    session = stimulus_session(_table(['s0']), ['behavior:choice'], CHOICE)
    with pytest.raises(ValueError, match='source evidence'):
        session.respond(StreamEvent('behavior:choice', _choices(['s0'], [[1, 0]], record=record),
                                    None))
    session.respond(StreamEvent('behavior:choice', _choices(
        ['s0'], [[1, 0]], record=record, raw_answer=['[0.7, 0.3]']), None))
    assert session.collect('behavior:choice')['raw_answer'].values[0] == '[0.7, 0.3]'


def test_a_source_format_that_does_not_fit_the_conversion_raises():
    with pytest.raises(ValueError, match='argmax conversion reads'):
        response_record('behavior:choice', 'external', conversion='argmax',
                        source_format='anything')
    record = {**response_record('behavior:choice', 'external', conversion='argmax',
                                source_format='behavior:probabilities'),
              'source_format': 'anything'}
    session = stimulus_session(_table(['s0']), ['behavior:choice'], CHOICE)
    with pytest.raises(ValueError, match='argmax conversion reads'):
        session.respond(StreamEvent('behavior:choice', _choices(
            ['s0'], [[1, 0]], record=record, raw_answer=['[0.7, 0.3]']), None))


# --- retention (N07) ------------------------------------------------------------

class _Bounded(Source):
    """A streamed source that keeps one response and asks for three."""
    history_limit = 1

    def __init__(self):
        self._events = iter([begin_event(), respond_event(), respond_event(), respond_event(),
                             end_event('terminal')])

    def next_input(self):
        return next(self._events, None)


def test_missing_rows_obey_the_history_limit():
    session = Session(_Bounded(), ['behavior:text'], TEXT)
    seen = []
    with session.observe(on_output=seen.append):
        _drain(session)
    assert len(session.emitted) == 1 and session.dropped == 2
    assert len(seen) == 3 and all(event.meta.get('missing') for event in seen)
