"""The generation settings of a written answer belong to the task (Contract Rule 3.9)."""
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Text, trial_prompt


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
