import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.model_interface import TaskContext
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _answers(*replies):
    lookup = dict(zip(['a', 'b'], replies))
    model = BrainScoreModel(identifier='answer', supported_modalities={'text'},
                            answer=lambda stimulus_row, instruction, label_set:
                            lookup[stimulus_row.stimulus_id])
    model.start_task(TaskContext(task_type='probabilities', instruction='choose',
                                 label_set=['cat', 'dog']))
    return model.process(StimulusSet({'sentence': ['x', 'y'], 'stimulus_id': ['a', 'b']}))


def test_invalid_answer_credits_no_label():
    with pytest.warns(UserWarning, match='invalid'):
        result = _answers('neither', 'dog')
    assert np.isnan(result.values[0]).all()
    np.testing.assert_array_equal(result.values[1], [0, 1])
    assert list(result['raw_answer'].values) == ['neither', 'dog']
    assert list(result['answer_valid'].values) == [False, True]


def test_valid_answers_are_unchanged():
    result = _answers('cat', 'dog')
    np.testing.assert_array_equal(result.values, [[1, 0], [0, 1]])
    assert result['answer_valid'].values.all()
