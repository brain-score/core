"""Regions record on every task path: likelihood scoring, a reply with implied
probabilities, composite regions, streaming sessions, and extractors without
record() (through forward hooks on the model)."""
from contextlib import contextmanager

import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.model_interface import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.selection import CompositeSelector, LayerSelector
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['x', 'yy', 'zzz']})
STIMULI.identifier = 'everywhere'


class Net:
    """Each pass sees a text; 'early' has 3 units, 'late' 2. Passes are logged."""

    def __init__(self):
        self.passes, self.hooks = [], []

    def forward(self, text):
        self.passes.append(text)
        out = {'early': np.array([len(text), 2.0 * len(text), 3.0 * len(text)]),
               'late': np.array([-len(text), 10.0])}
        for hook in self.hooks:
            hook(out)
        return out


class Recordable:
    identifier = backbone_id = 'toy'

    def __init__(self, net):
        self.net = net

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('recorded during the task')

    @contextmanager
    def record(self, layers):
        seen = []

        class Take:
            @staticmethod
            def take():
                first = seen[0]
                seen.clear()
                values = np.concatenate([first[layer] for layer in layers])[None]
                units = [(layer, i) for layer in layers for i in range(len(first[layer]))]
                return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ['?']),
                    'neuroid_id': ('neuroid', [f'{l}.{i}' for l, i in units]),
                    'neuroid_num': ('neuroid', [i for _, i in units]),
                    'layer': ('neuroid', [l for l, _ in units])})
        self.net.hooks.append(seen.append)
        try:
            yield Take()
        finally:
            self.net.hooks.remove(seen.append)


def _model(net, region_map=None):
    def reply(prompt):
        net.forward(prompt.text())
        return 'cat'

    def likelihood(prompt, candidates):
        net.forward(prompt.text())                         # the prompt alone first
        return {c: -float(len(net.forward(prompt.text() + c)['late'])) - i
                for i, c in enumerate(candidates)}
    return BrainScoreModel('toy', extractor={'text': Recordable(net)},
                           region_layer_map=region_map or {'A': LayerSelector('early', modality='text')},
                           reply=reply, likelihood=likelihood)


def _run(model, task, channels, streaming=False):
    session = BufferedSession(stimulus_set=STIMULI, task_context=task,
                              requested_output_channels=channels, streaming=streaming)
    model.participate(session)
    return session


def _prompt_text(stimulus_text):
    return f'Which?\n\n{stimulus_text}\n\nAnswer with exactly one of: cat, dog.'


def test_recording_during_likelihood_scoring_is_the_prompt_pass():
    net = Net()
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True),
                       prefer_path='likelihood')
    recorded = _run(_model(net), task, ['behavior:probabilities', 'neural:A']).collect('neural:A')
    expected = [Net().forward(_prompt_text(s))['early'] for s in ('x', 'yy', 'zzz')]
    np.testing.assert_array_equal(recorded.values, expected)


def test_a_reply_with_implied_probabilities_records_the_prompt_pass():
    net = Net()
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True),
                       prefer_path='generation')
    session = _run(_model(net), task, ['behavior:choice', 'behavior:probabilities', 'neural:A'])
    expected = [Net().forward(_prompt_text(s))['early'] for s in ('x', 'yy', 'zzz')]
    np.testing.assert_array_equal(session.collect('neural:A').values, expected)
    assert session.collect('behavior:probabilities').attrs['measure']['source'] == 'likelihood'


def test_a_composite_region_records_its_chosen_units():
    net = Net()
    region = CompositeSelector(layers=(('early', (0, 2)), ('late', None)), modality='text')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    recorded = _run(_model(net, {'R': region}), task, ['behavior:choice', 'neural:R']).collect('neural:R')
    first = Net().forward(_prompt_text('x'))
    np.testing.assert_array_equal(recorded.values[0], [first['early'][0], first['early'][2], *first['late']])


def test_a_streaming_session_gets_each_trial_as_it_ends():
    net = Net()
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = _run(_model(net), task, ['behavior:choice', 'neural:A'], streaming=True)
    assert [e.t_ms for e in session.emitted if e.channel == 'neural:A'] == [0.0, 1.0, 2.0]
    assert session.collect('neural:A').sizes['presentation'] == 3


def test_an_extractor_without_record_is_recorded_through_the_model():
    torch = pytest.importorskip('torch')
    network = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))

    def extract(stimuli, layers, **kwargs):
        raise AssertionError('recorded during the task')

    def reply(prompt):
        network(torch.tensor([[float(len(prompt.text())), 1.0]]))
        return 'cat'
    model = BrainScoreModel('plain', model=network, extractor={'text': extract},
                            region_layer_map={'A': LayerSelector('0', modality='text')}, reply=reply)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    recorded = _run(model, task, ['behavior:choice', 'neural:A']).collect('neural:A')
    expected = network[0](torch.tensor([[float(len(_prompt_text('x'))), 1.0]])).detach().numpy()[0]
    np.testing.assert_allclose(recorded.values[0], expected, rtol=1e-6)
    assert not network[0]._forward_hooks
