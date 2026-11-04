"""One presentation, two measurements: behavior and neural:<region> together.

A toy network counts its forward passes, so these tests check that every
stimulus is presented once, and that the recorded activity is the activity of
the pass that produced the answer.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.model_interface import TaskContext
from brainscore_core.selection import LayerSelector
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

LAYERS = {'early': 3, 'late': 2}   # layer -> number of units


class ToyNet:
    """Activity depends on the stimulus and on what else is in the prompt."""

    def __init__(self):
        self.passes, self.hooks = 0, []

    def forward(self, stimulus_id, prompt=''):
        self.passes += 1
        base = float(sum(map(ord, stimulus_id)) + len(prompt))
        out = {layer: base + np.arange(n, dtype=float) * (i + 1)
               for i, (layer, n) in enumerate(LAYERS.items())}
        for hook in self.hooks:
            hook(out)
        return out


def _assembly(rows, stimulus_ids, layers):
    values = np.array([np.concatenate([row[layer] for layer in layers]) for row in rows])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(stimulus_ids)),
        'neuroid_id': ('neuroid', [f'{l}.{u}' for l in layers for u in range(LAYERS[l])]),
        'layer': ('neuroid', [l for l in layers for _ in range(LAYERS[l])])})


class ToyExtractor:
    identifier = backbone_id = 'toy'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return _assembly([self.net.forward(i) for i in ids], ids, layers)

    @contextmanager
    def record(self, layers):
        captured = []
        recording = type('Recording', (), {})()

        def take():
            first = captured[0]
            captured.clear()
            return _assembly([first], ['?'], layers)
        recording.take = take
        self.net.hooks.append(captured.append)
        try:
            yield recording
        finally:
            self.net.hooks.remove(captured.append)


class NoRecordExtractor(ToyExtractor):
    record = property(lambda self: (_ for _ in ()).throw(AttributeError('record')))


STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'y', 'z'],
                       'label': ['cat', 'dog', 'cat']})
STIMULI.identifier = 'joint-toy'


def _model(net, extractor_type=ToyExtractor, **kwargs):
    def answer(stimulus_row, instruction, label_set):
        out = net.forward(stimulus_row['stimulus_id'], prompt=instruction)
        return label_set[int(out['late'][0]) % 2]
    return BrainScoreModel(identifier='toy', extractor={'text': extractor_type(net)},
                           region_layer_map={'A': LayerSelector('early', modality='text'),
                                             'B': LayerSelector('late', modality='text')},
                           answer=answer, **kwargs)


def _run(model, task, channels):
    session = BufferedSession(stimulus_set=STIMULI, task_context=task,
                              requested_output_channels=channels)
    model.participate(session)
    return session


GENERATE = TaskContext(task_type='label', label_set=['cat', 'dog'],
                       instruction='Name the animal.', prefer_path='generation')


def test_answers_and_recording_come_from_one_presentation():
    net = ToyNet()
    model = _model(net)
    session = _run(model, GENERATE, ['behavior', 'neural:A', 'neural:B'])
    assert net.passes == len(STIMULI)                       # each stimulus shown once
    behavior = session.collect('behavior')
    model.reset()
    alone = _run(_model(ToyNet()), GENERATE, ['behavior']).collect('behavior')
    np.testing.assert_array_equal(behavior.values, alone.values)
    recorded = session.collect('neural:A')
    assert list(recorded['stimulus_id'].values) == ['a', 'b', 'c']
    assert set(recorded['region'].values) == {'A'}
    # The prompt is part of the pass: activity differs from passive viewing.
    expected = [ToyNet().forward(s, prompt='Name the animal.')['early'] for s in 'abc']
    np.testing.assert_array_equal(recorded.values, expected)


def test_readout_records_in_the_same_extraction():
    net = ToyNet()
    model = _model(net, behavioral_readout_layer='late')
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       fitting_stimuli=STIMULI, prefer_path='readout')
    model.start_task(task)
    fitted = net.passes
    session = _run(model, task, ['behavior', 'neural:A'])
    assert net.passes - fitted == len(STIMULI)              # one extraction for both
    model.reset()
    reference = _model(ToyNet(), behavioral_readout_layer='late')
    alone = _run(reference, task, ['behavior']).collect('behavior')
    np.testing.assert_allclose(session.collect('behavior').values, alone.values)
    recorded = session.collect('neural:A')
    np.testing.assert_array_equal(recorded.values,
                                  [ToyNet().forward(s)['early'] for s in 'abc'])


def test_extractor_that_cannot_record_fails_before_any_answer():
    net = ToyNet()
    model = _model(net, extractor_type=NoRecordExtractor)
    with pytest.raises(NotImplementedError, match='record'):
        _run(model, GENERATE, ['behavior', 'neural:A'])
    assert net.passes == 0


def test_undeclared_region_is_not_supported():
    with pytest.raises(NotImplementedError):
        _run(_model(ToyNet()), GENERATE, ['behavior', 'neural:IT'])


def test_neural_session_after_a_task_returns_activity():
    model = _model(ToyNet())
    model.start_task(GENERATE)
    session = BufferedSession(stimulus_set=STIMULI, requested_output_channels=['neural:A'])
    model.participate(session)
    assert isinstance(session.collect('neural:A'), NeuroidAssembly)
