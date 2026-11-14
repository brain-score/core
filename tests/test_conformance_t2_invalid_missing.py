"""Conformance test T2 (Response Format Standard, section 11.1): invalid and missing trials are kept.

Through ``participate``, a valid, an invalid (an unreadable reply) and a missing
trial (no reply) are each stored with their state, raw response and reason, and
the score's record says how the answer was given. A requested trial with no
answer event does not disappear: in a session built from a table it is a
missing row, also when the subject answers in one batch after reading every
input; in a session without a table, a ``respond`` signal with no answer
before the next ``begin`` or ``end`` is a missing trial. Nothing is filled. A
failure of the subject is an error of the run, never a trial.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.metrics import Score
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Choice
from brainscore_core.score_metadata import record_measures, stamp_score_metadata
from brainscore_core.streaming import (Session, StreamEvent, begin_event, end_event,
                                       respond_event)
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

RECORD = response_record('behavior:choice', 'external', parsed_by='model')
REPLIES = {'s0': 'cat', 's1': 'a bird', 's2': None}     # valid, invalid, missing
TASK = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))


def _stimuli(ids=tuple(REPLIES)):
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(ids), 'text': ['x'] * len(ids)}))
    stimuli.identifier = f't2-{len(ids)}'
    return stimuli


def _row(stimulus_id, reply):
    labels = ['cat', 'dog']
    chosen = reply if reply in labels else None
    values = [[float(chosen == label) for label in labels]] if chosen else [[np.nan] * 2]
    return BehavioralAssembly(
        np.array(values, dtype=np.float32), dims=['presentation', 'choice'],
        attrs={'response': RECORD},
        coords={'stimulus_id': ('presentation', [stimulus_id]), 'choice': labels,
                'raw_answer': ('presentation', [reply]),
                'answer_valid': ('presentation', [chosen is not None]),
                'parse_error': ('presentation',
                                ['' if chosen or reply is None else 'no label in the reply'])})


class _Replier(Subject):
    """Answers each trial it reads with a fixed reply, parsed on the model side."""
    identifier = 'replier'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}
    required_channels = set()

    def __init__(self, fail=False, skip=()):
        self.fail, self.skip = fail, set(skip)

    def participate(self, session):
        while (event := session.next_input()) is not None:
            if self.fail:
                raise RuntimeError('the model server is down')
            stimulus_id = event.meta['stimulus_id']
            if stimulus_id in self.skip:
                continue
            session.respond(StreamEvent('behavior:choice', _row(stimulus_id, REPLIES[stimulus_id]),
                                        event.t_ms, dict(event.meta)))

    def reset(self):
        pass


def _run(subject, stimuli=None):
    with record_measures() as measures:
        session = stimulus_session(stimuli if stimuli is not None else _stimuli(),
                                   ['behavior:choice'], TASK)
        subject.participate(session)
        answer = session.collect('behavior:choice')
    score = stamp_score_metadata(Score(0.5), subject, {'behavior:choice'}, harness_id='t2',
                                 measures=measures)
    return answer, score


def test_each_trial_keeps_its_state_raw_response_and_reason():
    answer, _ = _run(_Replier())
    assert list(answer['stimulus_id'].values) == ['s0', 's1', 's2']
    assert list(answer['response_state'].values) == ['valid', 'invalid', 'missing']
    assert list(answer['response_reason'].values) == ['', 'no label in the reply', 'no answer']
    raw = list(answer['raw_answer'].values)
    assert raw[:2] == ['cat', 'a bird'] and raw[2] != raw[2]   # None is kept as a null


def test_nothing_is_filled():
    answer, _ = _run(_Replier())
    np.testing.assert_array_equal(answer.values[0], [1.0, 0.0])
    assert np.isnan(answer.values[1:]).all()


def test_the_score_carries_the_record_of_the_answers():
    _, score = _run(_Replier())
    assert score.attrs['response'] == (RECORD,)


def test_a_trial_with_no_answer_event_is_a_missing_row():
    answer, _ = _run(_Replier(skip={'s1'}))
    assert list(answer['stimulus_id'].values) == ['s0', 's1', 's2']
    assert list(answer['response_state'].values) == ['valid', 'missing', 'missing']
    assert answer['response_reason'].values[1] == 'no answer event'
    assert np.isnan(answer.values[1]).all()


def test_a_batch_answer_after_every_input_completes_the_table():
    class Batch(_Replier):
        def participate(self, session):
            while session.next_input() is not None:
                pass
            batch = BehavioralAssembly(
                np.array([[1, 0], [0, 1]], dtype=np.float32), dims=['presentation', 'choice'],
                attrs={'response': RECORD},
                coords={'stimulus_id': ('presentation', ['s2', 's0']), 'choice': ['cat', 'dog']})
            session.respond(StreamEvent('behavior:choice', batch, None))

    answer, _ = _run(Batch(), _stimuli(('s0', 's1', 's2')))
    assert list(answer['stimulus_id'].values) == ['s0', 's1', 's2']   # the table's order
    assert list(answer['response_state'].values) == ['valid', 'missing', 'valid']
    np.testing.assert_array_equal(answer.values[[0, 2]], [[0, 1], [1, 0]])


def test_no_answer_at_all_gives_a_missing_row_for_every_trial():
    answer, _ = _run(_Replier(skip=set(REPLIES)))
    assert list(answer['response_state'].values) == ['missing'] * 3


def test_an_answer_row_without_a_stimulus_cannot_be_matched_and_raises():
    session = stimulus_session(_stimuli(), ['behavior:choice'], TASK)
    session.respond(StreamEvent('behavior:choice', 'cat', None))      # a cell with no stimulus_id
    with pytest.raises(ValueError, match='no stimulus_id'):
        session.collect('behavior:choice')


class _Speaker(Subject):
    """Answers the respond signals it is told to answer, in a session without a table."""
    identifier = 'speaker'
    in_channels = {'control', 'text'}
    out_channels = {'behavior:text'}
    required_channels = set()

    def __init__(self, answer_turns):
        self.answer_turns = answer_turns

    def participate(self, session):
        session.declare_inputs(self.in_channels)
        turn = 0
        while (event := session.next_input()) is not None:
            if event.channel == 'control' and event.payload == 'respond':
                if turn in self.answer_turns:
                    session.respond(StreamEvent('behavior:text', f'turn {turn}', None))
                turn += 1

    def reset(self):
        pass


def test_a_respond_signal_without_an_answer_is_a_missing_trial():
    events = [begin_event(), StreamEvent('text', 'a', 0.0), respond_event(),
              StreamEvent('text', 'b', 1.0), respond_event(), end_event('terminal')]
    session = Session(events, ['behavior:text'], TaskContext(response=None, task_type='passive'))
    _Speaker(answer_turns={0}).participate(session)
    answer = session.collect('behavior:text')
    assert list(answer['response_state'].values) == ['valid', 'missing']
    assert list(answer.values) == ['turn 0', None]


def test_a_failure_of_the_subject_is_an_error_not_a_missing_trial():
    with pytest.raises(RuntimeError, match='server is down'):
        _run(_Replier(fail=True))
