"""A requested answer the task's response does not give is refused before any compute."""
import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.model_interface import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c', 'd'], 'sentence': ['w', 'x', 'y', 'z'],
                       'label': ['cat', 'dog', 'cat', 'dog']})
STIMULI.identifier = 'response-preflight-toy'


class _CountingExtractor:
    def __init__(self):
        self.calls = 0

    def __call__(self, stimuli, layers, **kwargs):
        self.calls += 1
        ids = list(stimuli['stimulus_id'])
        values = np.array([[1.0, 0.0] if i in 'ac' else [0.0, 1.0] for i in ids])
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['f0', 'f1']),
            'layer': ('neuroid', [layers[0]] * 2)})


def _readout_model():
    extractor = _CountingExtractor()
    model = BrainScoreModel(identifier='toy-readout', extractor={'text': extractor},
                            behavioral_readout_layer='late')
    return model, extractor


def _likelihood_model():
    calls = []

    def likelihood(prompt, candidates):
        calls.append(prompt.stimulus_id)
        return {c: float(i) for i, c in enumerate(candidates)}

    return BrainScoreModel(identifier='toy-likelihood', likelihood=likelihood), calls


def _ask(model, task, channels):
    session = BufferedSession(stimulus_set=STIMULI, task_context=task,
                              requested_output_channels=channels)
    model.participate(session)
    return session


def test_readout_refuses_probabilities_the_response_does_not_give():
    model, extractor = _readout_model()
    fits = []
    model._behavioral.fit_behavioral_readout = lambda *a, **k: fits.append(1)  # never reached
    task = TaskContext(response=Choice(['cat', 'dog'], probabilities=False),
                       fitting_stimuli=STIMULI)
    assert 'behavior:probabilities' in model.out_channels
    with pytest.raises(ValueError, match=r"behavior:probabilities.*behavior:choice"):
        _ask(model, task, ['behavior:probabilities'])
    assert extractor.calls == 0
    assert fits == []
    assert model._behavioral.readout_classifier is None


def test_likelihood_refuses_probabilities_the_response_does_not_give():
    model, calls = _likelihood_model()
    task = TaskContext(instruction='Pick.', response=Choice(['cat', 'dog'], probabilities=False))
    with pytest.raises(ValueError, match='behavior:probabilities'):
        _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    assert calls == []


def test_readout_matched_request_is_unchanged():
    model, extractor = _readout_model()
    task = TaskContext(response=Choice(['cat', 'dog'], probabilities=True), fitting_stimuli=STIMULI)
    session = _ask(model, task, ['behavior:probabilities'])
    assert session.collect('behavior:probabilities').values.shape == (4, 2)
    assert extractor.calls > 0


def test_likelihood_matched_request_is_unchanged():
    model, calls = _likelihood_model()
    task = TaskContext(instruction='Pick.', response=Choice(['cat', 'dog'], probabilities=True))
    session = _ask(model, task, ['behavior', 'behavior:choice', 'behavior:probabilities'])
    np.testing.assert_array_equal(session.collect('behavior:choice').values.argmax(1), [1, 1, 1, 1])
    assert session.collect('behavior:probabilities').values.shape == (4, 2)
    assert session.collect('behavior').values.shape == (4, 2)
    assert len(calls) == 4


def test_driver_refuses_addressed_channels_without_a_response_before_task_setup():
    from types import SimpleNamespace
    from brainscore_core._sessions.drivers import drive_behavior_and_neural_session
    setup = []
    model = SimpleNamespace(start_task=lambda _: setup.append('start_task'),
                            _behavioral=SimpleNamespace(history=[]))
    session = BufferedSession(stimulus_set=STIMULI, task_context=TaskContext(task_type='probabilities'),
                              requested_output_channels=['behavior:probabilities'])
    with pytest.raises(ValueError, match='TaskContext'):
        drive_behavior_and_neural_session(model, session)
    assert setup == []


def test_readout_without_a_response_refuses_addressed_channels_before_fitting():
    model, extractor = _readout_model()
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'], fitting_stimuli=STIMULI)
    with pytest.raises(ValueError, match='TaskContext'):
        _ask(model, task, ['behavior:probabilities'])
    assert extractor.calls == 0
    assert model._behavioral.readout_classifier is None


def test_plain_behavior_without_a_response_is_unchanged():
    calls = []

    def answer(stimulus_row, instruction, label_set):
        calls.append(stimulus_row['stimulus_id'])
        return label_set[0]

    model = BrainScoreModel(identifier='toy-answer', answer=answer)
    task = TaskContext(task_type='probabilities', instruction='Pick.', label_set=['cat', 'dog'],
                       prefer_path='generation')
    session = _ask(model, task, ['behavior'])
    assert session.collect('behavior').sizes['presentation'] == 4
    assert len(calls) == 4
