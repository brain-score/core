"""Legacy behavioral benchmarks score an invalid generated answer at chance."""
import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.model_interface import TaskContext
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _look_at(replies):
    ids = [f's{i}' for i in range(len(replies))]
    lookup = dict(zip(ids, replies))
    model = BrainScoreModel(identifier='answers', supported_modalities={'text'},
                            answer=lambda stimulus_row, instruction, label_set:
                            lookup[stimulus_row.stimulus_id])
    model.start_task(TaskContext(task_type='probabilities', instruction='choose',
                                 label_set=['cat', 'dog']))
    return model.look_at(StimulusSet({'sentence': ['x'] * len(ids), 'stimulus_id': ids}))


def test_invalid_answers_become_seeded_valid_labels():
    replies = ['cat'] * 19 + ['neither']   # 5% invalid, under the limit
    with pytest.warns(UserWarning, match='random valid labels'):
        first = _look_at(replies)
    second = _look_at(replies)
    assert not np.isnan(first.values).any()
    np.testing.assert_array_equal(first.values.sum(axis=1), 1)  # every row is one label
    np.testing.assert_array_equal(first.values, second.values)  # seeded, so repeatable
    assert first.attrs['n_invalid_answers'] == 1
    assert list(first['answer_valid'].values) == [True] * 19 + [False]


def test_valid_answers_pass_through_unchanged():
    result = _look_at(['cat', 'dog'])
    np.testing.assert_array_equal(result.values, [[1, 0], [0, 1]])
    assert 'n_invalid_answers' not in result.attrs


def test_too_many_invalid_answers_raise():
    with pytest.raises(ValueError, match='outside the label set'):
        _look_at(['cat'] * 8 + ['neither'] * 2)   # 20% invalid
