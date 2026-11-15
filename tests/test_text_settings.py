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
    assert (choice.labels, choice.verbatim) == (('a', 'b'), False)
    assert choice.channels == ('behavior:choice', 'behavior:probabilities',
                               'behavior:log_probabilities')
    assert choice.directive() == 'Answer with exactly one of: a, b.'
    assert TaskContext(response=choice).label_set == ['a', 'b']


def test_a_choice_names_its_formats_and_has_no_probabilities_switch():
    # The session's requested outputs say which formats are answered.
    with pytest.raises(TypeError):
        Choice(['a', 'b'], probabilities=True)
    assert Choice(['a', 'b'], reaction_time='surprisal').channels == (
        'behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities', 'behavior:rt')


def test_options_from_a_column_become_the_rows_labels():
    choice = Choice('options', verbatim=True)
    assert choice.channels == ('behavior:choice', 'behavior:probabilities',
                               'behavior:log_probabilities')
    task = TaskContext(response=choice)
    assert task.label_set is None
    prompt = trial_prompt(task, {'stimulus_id': 'a', 'text': 'the cat', 'options': ('sat', 'ran')})
    # Raw text: the text exactly as given, no directive.
    assert prompt.parts == (('text', 'the cat'),)
    assert prompt.response.labels == ('sat', 'ran')
    assert prompt.response.verbatim


def test_a_column_choice_without_verbatim_phrases_the_rows_options():
    task = TaskContext(response=Choice('options'))
    prompt = trial_prompt(task, {'stimulus_id': 'a', 'sentence': 'x', 'options': ['p', 'q']})
    assert prompt.parts[-1] == ('text', 'Answer with exactly one of: p, q.')


@pytest.mark.parametrize('fields, message', [
    ({'labels': ''}, 'column name in labels must not be empty'),
    ({}, 'at least two labels'),
    ({'labels': ['a']}, 'at least two labels'),
    ({'labels': ['a', 'b'], 'verbatim': 'yes'}, 'True or False'),
])
def test_choice_fields_are_checked_when_the_task_is_made(fields, message):
    with pytest.raises(ValueError, match=message):
        Choice(**fields)


def test_the_old_name_raw_text_is_gone():
    with pytest.raises(TypeError):
        Choice(['1', '2'], raw_text=True)


def test_a_choice_has_no_raw_scores_switch():
    # Log probabilities are their own format, behavior:log_probabilities.
    with pytest.raises(TypeError):
        Choice(['a', 'b'], raw_scores=True)


def test_a_choice_has_no_confidence():
    with pytest.raises(TypeError):
        Choice(['a', 'b'], confidence=Scale(1, 5))
    assert not hasattr(Choice(['a', 'b']), 'parse_confidence')


def test_verbatim_refuses_an_instruction():
    with pytest.raises(ValueError, match='A verbatim response shows the stimulus text exactly as given'):
        TaskContext(instruction='Pick one.', response=Choice(['a', 'b'], verbatim=True))


@pytest.mark.parametrize('options, message', [
    (None, 'sequence of option strings'), ('ab', 'sequence of option strings'),
    (('a', 'a'), 'distinct'), (('a', 2), 'sequence of option strings'),
])
def test_bad_row_options_fail_before_any_compute(options, message):
    task = TaskContext(response=Choice('options', verbatim=True))
    with pytest.raises(ValueError, match=message):
        trial_prompt(task, {'stimulus_id': 'a', 'text': 'x', 'options': options})
