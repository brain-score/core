"""Regression tests for the second review: batch semantics of a fixed event list,
and the streaming behavior driver."""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import (
    drive_behavior_session_streaming, drive_neural_session, drive_typed_session, stimulus_session)
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _Subject:
    def __init__(self):
        self.calls = []

    def start_recording(self, region, time_bins=None, recording_type=None):
        self.calls.append(('start_recording', region, time_bins))

    def start_task(self, task_context):
        self.calls.append(('start_task',))

    def process(self, stimuli, multi_modality=False):
        ids = list(stimuli['stimulus_id'])
        self.calls.append(('process', ids))
        if self.calls[0][0] == 'start_task':
            return BehavioralAssembly(np.ones((len(ids), 1)), dims=['presentation', 'choice'],
                                      coords={'stimulus_id': ('presentation', ids), 'choice': ('choice', ['c'])})
        return NeuroidAssembly(np.ones((len(ids), 2)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n0', 'n1']),
            'layer': ('neuroid', ['l0', 'l1']), 'region': ('neuroid', ['IT', 'V4'])})


def _two_column_row():
    meta = {'stimulus_index': 0, 'stimulus_id': 's0'}
    return [StreamEvent('text', 'a word', 0.0, meta={**meta, 'column': 'sentence'}),
            StreamEvent('vision:image', 'p.png', 0.0, meta={**meta, 'column': 'image_path'})]


@pytest.mark.parametrize('outputs,time_bins', [(['neural:IT'], None),
                                               (['neural:IT', 'neural:V4'], [(0, 100), (100, 200)])])
def test_fixed_event_list_without_a_table_is_one_batch(outputs, time_bins):
    session = Session(_two_column_row(), outputs, time_bins=time_bins)
    assert session.streaming is False
    subject = _Subject()
    drive_typed_session(subject, session)
    assert [call[1] for call in subject.calls if call[0] == 'process'] == [['s0']]
    assert len(session.emitted) == len(outputs)


def test_a_generator_of_events_is_still_answered_one_at_a_time():
    assert Session((event for event in _two_column_row()), ['neural:IT']).streaming is True


def _table(index, ids):
    table = StimulusSet(pd.DataFrame({'stimulus_id': ids, 'sentence': ['x', 'y']}, index=index))
    table.identifier = 'repeated-index'
    return table


def test_streaming_behavior_selects_rows_by_position():
    task = TaskContext(task_type='passive')
    session = stimulus_session(_table([0, 0], ['a', 'b']), ['behavior'], task, streaming=True)
    subject = _Subject()
    drive_behavior_session_streaming(subject, session)
    assert [call[1] for call in subject.calls if call[0] == 'process'] == [['a'], ['b']]


@pytest.mark.parametrize('streaming', [False, True])
def test_variance_request_is_refused_before_task_setup(streaming):
    task = TaskContext(task_type='passive')
    session = stimulus_session(_table([0, 1], ['a', 'b']), ['behavior'], task, streaming=streaming,
                               settings={'require_variance': True, 'number_of_trials': 3})
    subject = _Subject()
    with pytest.raises(NotImplementedError, match='variance'):
        drive_typed_session(subject, session)
    assert subject.calls == []


def test_streaming_behavior_answers_a_row_before_it_reads_the_next():
    table = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'y', 'z']}))
    table.identifier = 'order'
    session = stimulus_session(table, ['behavior'], TaskContext(task_type='passive'), streaming=True)
    log = []
    read, respond = session.next_input, session.respond

    def logged_read():
        event = read()
        if event is not None and event.channel != 'control':
            log.append(('read', event.payload))
        return event

    session.next_input = logged_read
    session.respond = lambda event: (log.append(('respond',)), respond(event))[1]
    drive_behavior_session_streaming(_Subject(), session)
    assert log == [('read', 'x'), ('respond',), ('read', 'y'), ('respond',), ('read', 'z'), ('respond',)]
