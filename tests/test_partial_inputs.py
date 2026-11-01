"""Contract v3 rule 5.5: streams a subject cannot take in never enter it."""
import numpy as np
import pandas as pd

from brainscore_core.compatibility import check_compatibility
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _VisionExtractor:
    def __init__(self):
        self.seen_columns = []

    def __call__(self, stimuli, layers=None, **kwargs):
        self.seen_columns.append(sorted(stimuli.columns))
        n = len(stimuli)
        return NeuroidAssembly(
            np.ones((n, 2)), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', [0, 1]),
                    'layer': ('neuroid', [layers[0]] * 2)})


class _Benchmark:
    identifier = 'movie-fmri'
    required_modalities = {'vision', 'text', 'audio'}
    region = 'IT'


def _movie():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['clip0', 'clip1'],
        'image_path': ['/tmp/0.png', '/tmp/1.png'],
        'sentence': ['subtitle zero', 'subtitle one'],
        'audio_path': ['/tmp/0.wav', '/tmp/1.wav'],
    }))
    stimuli.identifier = 'movie-clips'
    return stimuli


def test_vision_only_subject_runs_audiovisual_benchmark_on_vision_alone():
    extractor = _VisionExtractor()
    model = BrainScoreModel('vision-only', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor})

    match = check_compatibility(model, _Benchmark())
    assert match.kind == 'partial'
    assert match.dropped_inputs == {'text', 'audio'}

    response = neural_response(model, _movie(), record='IT')
    assert response.sizes['presentation'] == 2
    assert len(extractor.seen_columns) == 1  # one extraction, vision only
