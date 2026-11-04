"""A subject declares the visual forms it takes natively: stills, clips, or both."""
import numpy as np
import pytest

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.selection import LayerSelector
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _extractor(value):
    def extract(stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.full((len(ids), 2), value), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', ids),
                                       'neuroid_id': ('neuroid', ['n0', 'n1']),
                                       'layer': ('neuroid', [layers[0]] * 2)})
    return extract


def _stimuli(column):
    stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], column: ['a', 'b']})
    stimuli.identifier = f'forms-{column}'
    return stimuli


@pytest.mark.parametrize('keys, forms', [(('vision',), {'vision:still'}),
                                         (('video',), {'vision:clip'}),
                                         (('vision', 'video'), {'vision:still', 'vision:clip'})])
def test_forms_follow_the_extractor_keys(keys, forms):
    model = BrainScoreModel('m', extractor={key: _extractor(1.0) for key in keys})
    assert 'vision' in model.in_channels
    assert {c for c in model.in_channels if c.startswith('vision:')} == forms


def test_a_subject_with_both_forms_takes_each_stimulus_natively():
    model = BrainScoreModel('m', extractor={'vision': _extractor(1.0), 'video': _extractor(2.0)},
                            region_layer_map={'IT': LayerSelector('blocks.1', modality='vision')})
    assert neural_response(model, _stimuli('image_path'), record='IT').values.max() == 1.0
    assert neural_response(model, _stimuli('video_path'), record='IT').values.max() == 2.0


def test_models_without_an_extractor_declare_the_inputs_they_accept():
    model = BrainScoreModel('judge', supported_modalities={'vision'},
                            answer=lambda stimulus_row, instruction, label_set: label_set[0])
    assert 'vision:still' in model.in_channels
