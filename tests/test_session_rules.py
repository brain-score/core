"""The session enforces its own rules (Contract Rules sections 3, 4 and 5.5, 5.14).

Each test states one rule and checks it through the public session surface
(``Session``, ``stimulus_session``, ``check_event`` and the typed drivers) with
stand-in subjects that are contract subjects. A new driver or source that breaks
one of these rules fails here.
"""

import numpy as np
import pandas as pd
import pytest

from brainscore_core import io_catalog
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent, check_event
from brainscore_core.streaming_helpers import (
    behavioral_response,
    drive_neural_session,
    drive_neural_session_streaming,
    drive_typed_session,
    stimulus_session,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _table(n=2, t_ms=None):
    frame = {'stimulus_id': [f's{i}' for i in range(n)],
             'sentence': [f'word {i}' for i in range(n)]}
    if t_ms is not None:
        frame['t_ms'] = t_ms
    stimuli = StimulusSet(pd.DataFrame(frame))
    stimuli.identifier = 'session-rules-test'
    return stimuli


def _neural(stimuli):
    ids = list(stimuli['stimulus_id'])
    return NeuroidAssembly(
        np.ones((len(ids), 1)),
        coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n0']),
                'region': ('neuroid', ['IT'])},
        dims=['presentation', 'neuroid'])


class _Reader(Subject):
    """Takes in text only; keeps the task and outputs it saw."""
    identifier = 'reader'
    in_channels = {'text'}
    out_channels = {'behavior'}

    def participate(self, session):
        session.declare_inputs(self.in_channels)
        self.task, self.outputs = session.task, session.outputs
        list(iter(session.next_input, None))

    def reset(self):
        pass


class _Recorder(Subject):
    """A neural subject run through the typed drivers; counts its forward passes."""
    identifier = 'recorder'
    in_channels = {'text'}
    out_channels = {'neural:IT'}

    def __init__(self, driver=drive_typed_session, **driver_kwargs):
        self.driver, self.driver_kwargs, self.passes = driver, driver_kwargs, 0

    def participate(self, session):
        self.driver(self, session, **self.driver_kwargs)

    def reset(self):
        pass

    def start_recording(self, target, time_bins=None):
        pass

    def process(self, stimuli, **kwargs):
        self.passes += 1
        return _neural(stimuli)


# Rule 5.5: the task a subject sees shows only the inputs it takes in.

def test_an_undeclared_instruction_does_not_reach_the_subject():
    task = TaskContext(task_type='probabilities', label_set=['a', 'b'], instruction='Pick one')
    subject = _Reader()
    subject.participate(stimulus_session(_table(), ['behavior'], task))
    assert not subject.task.instruction
    assert task.instruction == 'Pick one'          # the caller's task is not changed


def test_a_declared_instruction_reaches_the_subject():
    task = TaskContext(task_type='probabilities', label_set=['a', 'b'], instruction='Pick one')
    subject = _Reader()
    subject.in_channels = {'text', 'instruction'}
    subject.participate(stimulus_session(_table(), ['behavior'], task))
    assert subject.task is task


# Rule 4.8: a session carries only StreamEvent.

@pytest.mark.parametrize('item', [42, {'text': 'a'}, None])
def test_a_listed_source_refuses_an_object_that_is_not_an_event(item):
    with pytest.raises(ValueError, match='StreamEvent'):
        Session([StreamEvent('text', 'a', None), item])


@pytest.mark.parametrize('item', [42, None])
def test_a_lazy_source_refuses_an_object_that_is_not_an_event(item):
    session = Session(iter([item, StreamEvent('text', 'a', None)]))
    with pytest.raises(ValueError, match='StreamEvent'):
        session.next_input()


# Rule 4.6: control signals reach only a session whose subject declares control,
# and an end signal carries its reason.

def test_a_driver_that_needs_control_refuses_a_session_without_a_declaration():
    subject = _Recorder(drive_neural_session_streaming, on_begin=lambda: None)
    session = stimulus_session(_table(), ['neural:IT'], streaming=True)
    with pytest.raises(ValueError, match="'control'"):
        subject.participate(session)
    assert subject.passes == 0


@pytest.mark.parametrize('meta', [{}, {'reason': 'bored'}])
def test_an_end_signal_needs_a_known_reason(meta):
    with pytest.raises(ValueError, match='reason'):
        check_event(StreamEvent('control', 'end', None, meta), io_catalog.INPUT)
    with pytest.raises(ValueError, match='reason'):
        Session([StreamEvent('control', 'end', None, meta)])


def test_an_end_signal_with_a_known_reason_passes():
    check_event(StreamEvent('control', 'end', None, {'reason': 'truncated'}), io_catalog.INPUT)


# Rule 5.14: an infeasible combination is refused with ExecutionNotFeasibleError
# before model computation.

@pytest.mark.parametrize('streaming', [False, True])
def test_a_variance_request_is_refused_before_compute(streaming):
    subject = _Recorder()
    session = stimulus_session(_table(), ['neural:IT'], settings={'require_variance': True},
                               streaming=streaming)
    with pytest.raises(ExecutionNotFeasibleError):
        subject.participate(session)
    assert subject.passes == 0


@pytest.mark.parametrize('driver', [drive_neural_session, drive_neural_session_streaming,
                                    drive_typed_session])
def test_an_unsupported_output_combination_is_infeasible(driver):
    subject = _Recorder(driver)
    session = stimulus_session(_table(), ['neural:IT', 'behavior'])
    with pytest.raises(ExecutionNotFeasibleError):
        subject.participate(session)
    assert subject.passes == 0


# Rule 4.2: the one answer for a whole table has no time.

def test_a_whole_table_neural_answer_has_no_time():
    subject = _Recorder(drive_neural_session)
    session = stimulus_session(_table(t_ms=[10.0, 20.0]), ['neural:IT'])
    subject.participate(session)
    assert [event.t_ms for event in session.emitted] == [None]


def test_a_streamed_neural_answer_keeps_the_time_of_its_row():
    subject = _Recorder(drive_neural_session_streaming)
    session = stimulus_session(_table(t_ms=[10.0, 20.0]), ['neural:IT'], streaming=True)
    subject.participate(session)
    assert [event.t_ms for event in session.emitted] == [10.0, 20.0]


# Rule 3.4 and the response format standard: a task with a response is answered
# on an addressed format, so the session checks each trial.

def test_the_behavior_helper_requests_the_format_of_the_response():
    subject = _Reader()
    subject.out_channels = {'behavior:choice'}
    task = TaskContext(response=Choice(['a', 'b']))
    behavior = behavioral_response(subject, _table(), task)     # the reader answers nothing
    assert subject.outputs == ('behavior:choice',)
    assert list(behavior['response_state'].values) == ['missing', 'missing']


def test_the_behavior_helper_keeps_bare_behavior_for_a_task_without_a_response():
    subject = _Reader()
    with pytest.raises(ValueError):
        behavioral_response(subject, _table(), TaskContext(task_type='probabilities',
                                                           label_set=['a', 'b']))
    assert subject.outputs == ('behavior',)


# The response format standard: a choice parses a reply only against its labels.

def test_a_choice_with_a_column_of_labels_does_not_parse_before_the_row_gives_them():
    with pytest.raises(ValueError, match='column'):
        Choice('options').parse('o')
