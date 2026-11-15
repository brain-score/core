"""A session refuses an addressed behavior format that the task's response does not name.

Contract Rule 3.4: the request is checked when the session is built. The benchmark's
request is malformed, so the refusal is a ``ValueError``. A task without a response,
bare ``behavior`` and the other families are not checked here.
"""
import pandas as pd
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import environment_session, stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from tests.test_conformance_stored_answers import _Conversation

CHOICE = TaskContext(response=Choice(['cat', 'dog']))
SCALE = TaskContext(response=Scale(1, 7, 1))
TEXT = TaskContext(response=Text())


def _stimuli():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['s0'], 'text': ['x']}))
    stimuli.identifier = 'requested-formats'
    return stimuli


BUILDERS = {
    'Session': lambda outputs, task: Session((), outputs, task),
    'stimulus_session': lambda outputs, task: stimulus_session(_stimuli(), outputs, task),
    'environment_session': lambda outputs, task: environment_session(
        _Conversation(1), outputs, task, action_channel='behavior:text'),
}

REFUSED = [
    (CHOICE, ['behavior:choice', 'behavior:rating'],
     "['behavior:rating'] requested, but the task's response Choice answers only on "
     "['behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities']; "
     "ask for those or change the response."),
    (SCALE, ['behavior:choice'],
     "['behavior:choice'] requested, but the task's response Scale answers only on "
     "['behavior:rating']; ask for those or change the response."),
    (TEXT, ['behavior:text', 'behavior:rt'],
     "['behavior:rt'] requested, but the task's response Text answers only on "
     "['behavior:text']; ask for those or change the response."),
]


@pytest.mark.parametrize('builder', sorted(BUILDERS))
@pytest.mark.parametrize('task, outputs, message', REFUSED, ids=['choice', 'scale', 'text'])
def test_a_format_the_response_does_not_name_is_refused_when_the_session_is_built(
        builder, task, outputs, message):
    with pytest.raises(ValueError) as raised:
        BUILDERS[builder](outputs, task)
    assert str(raised.value) == message


ALLOWED = [
    (CHOICE, ['behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities']),
    (TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal')), ['behavior:rt']),
    (TaskContext(response=Choice(['cat', 'dog'], confidence=Scale(1, 5))), ['behavior:confidence']),
    (SCALE, ['behavior:rating', 'behavior', 'neural:IT']),
    (TEXT, ['behavior:text']),
]


@pytest.mark.parametrize('builder', ['Session', 'stimulus_session'])
@pytest.mark.parametrize('task, outputs', ALLOWED, ids=['choice', 'rt', 'confidence', 'scale', 'text'])
def test_formats_the_response_names_bare_behavior_and_other_families_are_allowed(builder, task, outputs):
    assert BUILDERS[builder](outputs, task).outputs == tuple(outputs)


@pytest.mark.parametrize('builder', ['Session', 'stimulus_session'])
@pytest.mark.parametrize('task', [None, TaskContext(task_type='passive'),
                                  TaskContext(task_type='probabilities', label_set=['cat', 'dog'])],
                         ids=['no task', 'passive', 'legacy'])
def test_a_task_without_a_response_is_not_checked(builder, task):
    outputs = ['behavior:rating', 'behavior:choice', 'behavior:rt']
    assert BUILDERS[builder](outputs, task).outputs == tuple(outputs)


def test_an_environment_session_with_a_named_format_is_allowed():
    assert environment_session(_Conversation(1), None, TEXT,
                               action_channel='behavior:text').outputs == ('behavior:text',)
