"""The generation settings of a written answer belong to the task (Contract Rule 3.9)."""
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text, trial_prompt


def test_text_without_settings_is_unchanged():
    assert Text() == Text(max_tokens=None, temperature=None, top_p=None, seed=None)
    assert Text().channels == ('behavior:text',)
    assert Text().directive() == ''


def test_text_keeps_its_settings():
    text = Text(max_tokens=512, temperature=1.0, top_p=0.9, seed=7)
    assert (text.max_tokens, text.temperature, text.top_p, text.seed) == (512, 1.0, 0.9, 7)
    assert TaskContext(response=text).task_type == 'text'


@pytest.mark.parametrize('settings', [
    {'max_tokens': 0}, {'max_tokens': 1.5}, {'max_tokens': True},
    {'temperature': -1}, {'temperature': float('nan')}, {'temperature': float('inf')},
    {'temperature': True}, {'top_p': 0.9}, {'temperature': 0.0, 'top_p': 0.9},
    {'temperature': 1.0, 'top_p': 0}, {'temperature': 1.0, 'top_p': 1.5}, {'seed': 1.0},
])
def test_text_settings_are_checked_when_the_task_is_made(settings):
    with pytest.raises(ValueError):
        Text(**settings)


@pytest.mark.parametrize('response', [Text(max_tokens=16, temperature=1.0), Choice(['a', 'b'])])
def test_trial_prompt_carries_the_task_response(response):
    """A reply part reads the settings from the prompt it already receives."""
    prompt = trial_prompt(TaskContext(instruction='Say it.', response=response),
                          {'stimulus_id': 'a', 'sentence': 'x'})
    assert prompt.response is response
    assert prompt.parts[:2] == (('text', 'Say it.'), ('text', 'x'))


# A choice whose options differ for each row, scored raw (Contract Rule 3.9).

def test_a_plain_choice_is_unchanged():
    choice = Choice(['a', 'b'])
    assert (choice.labels_column, choice.raw_scores, choice.raw_text) == (None, False, False)
    assert choice.channels == ('behavior:choice',)
    assert choice.directive() == 'Answer with exactly one of: a, b.'
    assert TaskContext(response=choice).label_set == ['a', 'b']


def test_options_from_a_column_become_the_rows_labels():
    choice = Choice(labels_column='options', raw_scores=True, raw_text=True)
    assert choice.channels == ('behavior:choice', 'behavior:probabilities')
    task = TaskContext(response=choice)
    assert task.label_set is None
    prompt = trial_prompt(task, {'stimulus_id': 'a', 'text': 'the cat', 'options': ('sat', 'ran')})
    # Raw text: the text exactly as given, no directive.
    assert prompt.parts == (('text', 'the cat'),)
    assert prompt.response.labels == ('sat', 'ran')
    assert prompt.response.labels_column is None and prompt.response.raw_scores


def test_a_column_choice_without_raw_text_phrases_the_rows_options():
    task = TaskContext(response=Choice(labels_column='options'))
    prompt = trial_prompt(task, {'stimulus_id': 'a', 'sentence': 'x', 'options': ['p', 'q']})
    assert prompt.parts[-1] == ('text', 'Answer with exactly one of: p, q.')


@pytest.mark.parametrize('fields, message', [
    ({'labels': ['a', 'b'], 'labels_column': 'options'}, 'not both'),
    ({'labels_column': ''}, 'name of a stimulus column'),
    ({}, 'at least two labels'),
    ({'labels': ['a', 'b'], 'raw_scores': 1}, 'True or False'),
    ({'labels': ['a', 'b'], 'raw_text': 'yes'}, 'True or False'),
    ({'labels': ['a', 'b'], 'raw_scores': True, 'reaction_time': 'surprisal'}, 'normalized'),
    ({'labels': ['a', 'b'], 'raw_text': True, 'confidence': Scale(1, 5, 1)}, 'confidence'),
])
def test_choice_fields_are_checked_when_the_task_is_made(fields, message):
    with pytest.raises(ValueError, match=message):
        Choice(**fields)


def test_raw_text_refuses_an_instruction():
    with pytest.raises(ValueError, match='exactly as given'):
        TaskContext(instruction='Pick one.', response=Choice(['a', 'b'], raw_text=True))


@pytest.mark.parametrize('options, message', [
    (None, 'sequence of option strings'), ('ab', 'sequence of option strings'),
    (('a', 'a'), 'distinct'), (('a', 2), 'sequence of option strings'),
])
def test_bad_row_options_fail_before_any_compute(options, message):
    task = TaskContext(response=Choice(labels_column='options', raw_text=True))
    with pytest.raises(ValueError, match=message):
        trial_prompt(task, {'stimulus_id': 'a', 'text': 'x', 'options': options})
