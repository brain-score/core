"""Conformance test T2 (Response Format Standard, section 11.1): invalid and missing trials are kept.

Through ``participate``, a valid, an invalid (an unreadable reply) and a missing
trial (no reply) are each stored with their state, raw response and reason, and
the score's record says how the answer was given. Nothing is filled. A
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
from brainscore_core.streaming import StreamEvent
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


def test_a_failure_of_the_subject_is_an_error_not_a_missing_trial():
    with pytest.raises(RuntimeError, match='server is down'):
        _run(_Replier(fail=True))
