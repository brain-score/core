"""TaskContext(response=...): choices, confidence, ratings and written answers.

Reported answers come from a reply and are parsed by the response; implied
measures come from a likelihood part or a readout. Each result says which.
"""
import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.model_interface import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c', 'd'], 'sentence': ['w', 'x', 'y', 'z'],
                       'label': ['cat', 'dog', 'cat', 'dog']})
STIMULI.identifier = 'responses-toy'


def _features(stimuli, layers, **kwargs):
    ids = list(stimuli['stimulus_id'])
    values = np.array([[1.0, 0.0] if i in 'ac' else [0.0, 1.0] for i in ids])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['f0', 'f1']),
        'layer': ('neuroid', [layers[0]] * 2)})


def _model(replies=None, scores=None, readout=False):
    prompts = []

    def reply(stimulus_row, prompt):
        prompts.append(prompt)
        return replies[stimulus_row['stimulus_id']]

    def likelihood(stimulus_row, prompt, candidates):
        return {c: scores[stimulus_row['stimulus_id']][c] for c in candidates}

    model = BrainScoreModel(identifier='toy', extractor={'text': _features},
                            reply=reply if replies else None,
                            likelihood=likelihood if scores else None,
                            behavioral_readout_layer='late' if readout else None)
    return model, prompts


def _ask(model, task, channels):
    session = BufferedSession(stimulus_set=STIMULI, task_context=task,
                              requested_output_channels=channels)
    model.participate(session)
    return session


def test_choice_reply_is_parsed_and_invalid_replies_are_kept_as_invalid():
    model, prompts = _model(replies={'a': 'Cat', 'b': 'a dog.', 'c': 'cat or dog', 'd': 'bird'})
    task = TaskContext(instruction='Which animal?', response=Choice(['cat', 'dog']))
    choice = _ask(model, task, ['behavior:choice']).collect('behavior:choice')
    np.testing.assert_array_equal(choice.values[:2], [[1, 0], [0, 1]])
    assert np.isnan(choice.values[2:]).all()
    assert list(choice['answer_valid'].values) == [True, True, False, False]
    assert list(choice['raw_answer'].values) == ['Cat', 'a dog.', 'cat or dog', 'bird']
    assert choice.attrs['measure'] == {'kind': 'reported', 'source': 'reply'}
    assert prompts[0] == 'Which animal?\n\nAnswer with exactly one of: cat, dog.'


def test_reported_confidence_comes_from_the_same_reply():
    model, _ = _model(replies={'a': 'cat\n4', 'b': 'dog\n9', 'c': 'cat', 'd': 'dog\n2'})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], confidence=Scale(1, 5, 1)))
    session = _ask(model, task, ['behavior:choice', 'behavior:confidence'])
    confidence = session.collect('behavior:confidence')
    np.testing.assert_array_equal(np.isnan(confidence.values), [False, True, True, False])
    assert confidence.values[0] == 4 and confidence.values[3] == 2
    assert 'outside' in confidence['parse_error'].values[1]


def test_rating_from_a_reply_and_from_a_readout():
    model, _ = _model(replies={'a': '6', 'b': 'about 2', 'c': '7.5', 'd': 'none'})
    rating = _ask(model, TaskContext(instruction='How cute?', response=Scale(1, 7, 1)),
                  ['behavior:rating']).collect('behavior:rating')
    assert list(rating.values[:2]) == [6, 2] and np.isnan(rating.values[2:]).all()

    fitting = STIMULI.assign(label=[6.0, 2.0, 6.0, 2.0])
    fitting.identifier = 'responses-ratings'
    readout, _ = _model(readout=True)
    task = TaskContext(response=Scale(1, 7), fitting_stimuli=fitting)
    predicted = _ask(readout, task, ['behavior:rating']).collect('behavior:rating')
    np.testing.assert_allclose(predicted.values, [6, 2, 6, 2], atol=0.5)
    assert predicted.attrs['measure'] == {'kind': 'implied', 'source': 'readout'}


def test_written_answers_are_kept_as_written():
    model, prompts = _model(replies={s: f'A story about {s}.' for s in 'abcd'})
    text = _ask(model, TaskContext(instruction='Describe it.', response=Text()),
                ['behavior:text']).collect('behavior:text')
    assert list(text.values) == [f'A story about {s}.' for s in 'abcd']
    assert prompts[0] == 'Describe it.'


def test_likelihood_gives_a_choice_and_its_implied_probabilities():
    scores = {s: {'cat': -1.0, 'dog': -3.0} if s in 'ac' else {'cat': -2.0, 'dog': -0.5}
              for s in 'abcd'}
    model, _ = _model(scores=scores)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True))
    session = _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    probabilities = session.collect('behavior:probabilities')
    np.testing.assert_allclose(probabilities.values[0], [1 / (1 + np.exp(-2)), 1 - 1 / (1 + np.exp(-2))],
                               rtol=1e-6)
    assert probabilities.attrs['measure'] == {'kind': 'implied', 'source': 'likelihood'}
    np.testing.assert_array_equal(session.collect('behavior:choice').values.argmax(1), [0, 1, 0, 1])


def test_readout_choice_with_probabilities():
    model, _ = _model(readout=True)
    task = TaskContext(response=Choice(['cat', 'dog'], probabilities=True), fitting_stimuli=STIMULI)
    session = _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    np.testing.assert_array_equal(session.collect('behavior:choice').values.argmax(1), [0, 1, 0, 1])
    assert session.collect('behavior:probabilities').values.shape == (4, 2)


@pytest.mark.parametrize('response, channel', [(Text(), 'behavior:text'),
                                               (Choice(['cat', 'dog'], confidence=Scale(1, 5)),
                                                'behavior:confidence')])
def test_reported_answers_from_a_readout_only_model_are_not_supported(response, channel):
    model, _ = _model(readout=True)
    assert channel not in model.out_channels
    with pytest.raises(NotImplementedError):
        _ask(model, TaskContext(response=response, fitting_stimuli=STIMULI), [channel])


def test_plain_behavior_request_returns_the_primary_answer():
    model, _ = _model(replies={s: 'dog' for s in 'abcd'})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    behavior = _ask(model, task, ['behavior']).collect('behavior')
    np.testing.assert_array_equal(behavior.values.argmax(1), [1, 1, 1, 1])


def test_task_type_and_response_must_agree():
    with pytest.raises(ValueError, match='contradicts'):
        TaskContext(task_type='label', response=Scale(1, 7))
    with pytest.raises(ValueError, match='passive'):
        TaskContext()
    assert TaskContext(response=Choice(['a', 'b'])).label_set == ['a', 'b']
