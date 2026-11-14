"""Conformance test T2 (Response Format Standard, section 11.1): invalid and missing trials are kept.

A fake subject answers through ``participate`` with one valid, one invalid
(an unparseable reply) and one missing trial (no reply). The session keeps
every trial with its state, raw response and reason; ``collect`` returns them
unchanged; the score's record says how the answer was given. Nothing is
filled: the values of the invalid and missing trials stay NaN. A failure of
the subject itself is an error of the run, never a trial.
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


def _stimuli():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(REPLIES), 'text': ['x', 'y', 'z']}))
    stimuli.identifier = 't2-stimuli'
    return stimuli


class _Replier(Subject):
    """Answers each trial with a fixed reply; parses it like a model side does."""
    identifier = 'replier'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}
    required_channels = set()

    def __init__(self, fail=False):
        self.fail = fail

    def participate(self, session):
        labels = ['cat', 'dog']
        while (event := session.next_input()) is not None:
            if self.fail:
                raise RuntimeError('the model server is down')
            stimulus_id = event.meta['stimulus_id']
            reply = REPLIES[stimulus_id]
            chosen = reply if reply in labels else None
            row = [[float(chosen == label) for label in labels]] if chosen else [[np.nan] * 2]
            session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
                row, dims=['presentation', 'choice'], attrs={'response': RECORD},
                coords={'stimulus_id': ('presentation', [stimulus_id]), 'choice': labels,
                        'raw_answer': ('presentation', [reply]),
                        'answer_valid': ('presentation', [chosen is not None]),
                        'parse_error': ('presentation',
                                        ['' if chosen or reply is None else 'no label in the reply'])}),
                event.t_ms, dict(event.meta)))

    def reset(self):
        pass


TASK = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))


def _run(subject):
    with record_measures() as measures:
        session = stimulus_session(_stimuli(), ['behavior:choice'], TASK)
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
