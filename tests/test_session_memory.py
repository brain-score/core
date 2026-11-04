"""Memory across trials: a declared trait, required by benchmarks, kept through a session."""
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.model_interface import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core._evaluation.compatibility import check_channel_compatibility

pytestmark = pytest.mark.unit

WORDS = ['cat', 'cat', 'dog', 'sun', 'sun', 'dog']
ONE_BACK = [None, 'yes', 'no', 'no', 'yes', 'no']        # same as the previous trial?
STIMULI = StimulusSet({'stimulus_id': [f's{i}' for i in range(6)], 'sentence': WORDS})
STIMULI.identifier = 'one-back'
TASK = TaskContext(instruction='Is this word the same as the previous one?',
                   response=Choice(['yes', 'no']))


def _word(prompt):
    return [content for kind, content in prompt.parts if kind == 'text'][1]


def one_back(prompt):
    """Answers from the session's history; with no history it can only say no."""
    previous = _word(prompt.history[-1][0]) if prompt.history else None
    return 'yes' if previous == _word(prompt) else 'no'


def _answers(model):
    session = BufferedSession(stimulus_set=STIMULI, task_context=TASK,
                              requested_output_channels=['behavior:choice'])
    model.participate(session)
    return list(session.collect('behavior:choice')['raw_answer'].values)


def test_a_model_with_memory_sees_earlier_trials():
    model = BrainScoreModel('remembers', supported_modalities={'text'}, reply=one_back, memory='session')
    assert model.traits == {'memory:session'}
    assert _answers(model)[1:] == ONE_BACK[1:]


def test_without_memory_every_trial_is_alone():
    model = BrainScoreModel('forgets', supported_modalities={'text'}, reply=one_back)
    assert model.traits == set()
    assert _answers(model) == ['no'] * 6


def test_memory_lasts_one_session():
    model = BrainScoreModel('remembers', supported_modalities={'text'}, reply=one_back, memory='session')
    _answers(model)
    assert len(model._behavioral.history) == 6
    model.reset()
    assert model._behavioral.history == []
    assert _answers(model)[1:] == ONE_BACK[1:]               # a new session starts fresh


class _MemoryBenchmark:
    identifier = 'one-back'
    required_input_channels = {'text'}
    requested_output_channels = {'behavior:choice'}
    required_traits = {'memory:session'}


def test_a_memory_benchmark_marks_a_model_without_memory_na():
    forgets = BrainScoreModel('forgets', supported_modalities={'text'}, reply=one_back)
    with pytest.raises(CompatibilityError, match='memory:session'):
        check_channel_compatibility(forgets, _MemoryBenchmark())
    remembers = BrainScoreModel('remembers', supported_modalities={'text'}, reply=one_back, memory='session')
    check_channel_compatibility(remembers, _MemoryBenchmark())


def test_memory_needs_a_part_that_can_see_history():
    model = BrainScoreModel('answers', supported_modalities={'text'}, memory='session',
                            answer=lambda stimulus_row, instruction, label_set: 'no')
    with pytest.raises(ValueError, match='reply'):
        _answers(model)
