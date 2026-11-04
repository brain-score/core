"""A fused extractor: one forward pass over several input types, split by input."""
import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.selection import LayerSelector
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


class Fused:
    """Sees the image and the sentence of each stimulus together; returns one step per input."""
    identifier = backbone_id = 'fused'

    def __init__(self):
        self.calls = []

    def __call__(self, stimuli, layers, **kwargs):
        self.calls.append(sorted(c for c in stimuli.columns if c != 'stimulus_id'))
        ids = list(stimuli['stimulus_id'])
        values = np.stack([[[1.0, 2.0], [3.0, 4.0]] for _ in ids])       # parts: image, text
        return NeuroidAssembly(values, dims=['presentation', 'part', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'part_name': ('part', ['image', 'text']),
            'part_index': ('part', [0, 1]),
            'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})


def test_a_fused_extractor_declares_each_input_and_runs_once():
    fused = Fused()
    model = BrainScoreModel('fused', extractor={('vision', 'text'): fused},
                            region_layer_map={'A': 'layers.3'})
    assert {'vision', 'text', 'vision:still'} <= model.in_channels
    both = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png'],
                        'sentence': ['one', 'two']})
    both.identifier = 'fused-both'
    session = BufferedSession(stimulus_set=both, requested_output_channels=['neural:A'])
    model.participate(session)
    recorded = session.collect('neural:A')
    assert len(fused.calls) == 1 and fused.calls[0] == ['image_path', 'sentence']
    assert recorded.dims == ('presentation', 'part', 'neuroid')
    assert list(recorded['part_name'].values) == ['image', 'text']


def test_single_input_extractors_still_take_single_input_stimuli():
    fused, vision_calls = Fused(), []

    def vision(stimuli, layers, **kwargs):
        vision_calls.append(1)
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 2)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['v0', 'v1']),
            'layer': ('neuroid', [layers[0]] * 2)})
    model = BrainScoreModel('mixed', extractor={'vision': vision, ('vision', 'text'): fused},
                            region_layer_map={'A': LayerSelector('layers.3', modality='vision')})
    images = StimulusSet({'stimulus_id': ['a'], 'image_path': ['a.png']})
    images.identifier = 'images-only'
    session = BufferedSession(stimulus_set=images, requested_output_channels=['neural:A'])
    model.participate(session)
    assert session.collect('neural:A').dims == ('presentation', 'neuroid')
