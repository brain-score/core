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


class _Recorder:
    """record(layers) stand-in; a clip recording has two time steps."""

    def __init__(self, value, timed):
        self.value, self.timed, self.hooks = value, timed, []

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('a task presents through reply, not the extractor')

    def record(self, layers):
        from contextlib import contextmanager

        @contextmanager
        def recording():
            owner = self

            class Take:
                @staticmethod
                def take():
                    coords = {'stimulus_id': ('presentation', ['?']),
                              'neuroid_id': ('neuroid', ['n0', 'n1']),
                              'layer': ('neuroid', [layers[0]] * 2)}
                    if not owner.timed:
                        return NeuroidAssembly(np.full((1, 2), owner.value), coords=coords,
                                               dims=['presentation', 'neuroid'])
                    coords.update(time_bin_start_ms=('time_bin', [0.0, 1000.0]),
                                  time_bin_end_ms=('time_bin', [1000.0, 2000.0]))
                    values = np.stack([np.full(2, owner.value), np.full(2, owner.value + 1)])[None]
                    return NeuroidAssembly(values, coords=coords,
                                           dims=['presentation', 'time_bin', 'neuroid'])
            yield Take()
        return recording()


def test_a_clip_recorded_during_a_task_uses_the_clip_extractor_and_keeps_time():
    from brainscore_core.model_interface import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming import BufferedSession
    model = BrainScoreModel('m', extractor={'vision': _Recorder(1.0, False), 'video': _Recorder(5.0, True)},
                            region_layer_map={'IT': LayerSelector('blocks.1', modality='vision')},
                            reply=lambda prompt: 'left')
    task = TaskContext(instruction='Which way?', response=Choice(['left', 'right']))
    session = BufferedSession(stimulus_set=_stimuli('video_path'), task_context=task,
                              requested_output_channels=['behavior:choice', 'neural:IT'])
    model.participate(session)
    recorded = session.collect('neural:IT')
    assert recorded.dims == ('presentation', 'time_bin', 'neuroid')
    np.testing.assert_array_equal(recorded.values[0, :, 0], [5.0, 6.0])
    assert list(recorded['time_bin_start_ms'].values) == [0.0, 1000.0]


def test_a_multi_image_trial_is_recorded_per_image():
    from contextlib import contextmanager
    from brainscore_core.model_interface import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming import BufferedSession

    class PerImage:
        def __call__(self, stimuli, layers, **kwargs):
            raise AssertionError('recorded during the reply')

        @contextmanager
        def record(self, layers):
            class Take:
                @staticmethod
                def take():
                    values = np.arange(6, dtype=float).reshape(1, 3, 2)   # 3 images, 2 units
                    return NeuroidAssembly(values, dims=['presentation', 'part', 'neuroid'], coords={
                        'stimulus_id': ('presentation', ['?']), 'part_index': ('part', [0, 1, 2]),
                        'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})
            yield Take()

    model = BrainScoreModel('m', extractor={'vision': PerImage()},
                            region_layer_map={'IT': LayerSelector('blocks.1', modality='vision')},
                            reply=lambda prompt: 'yes')
    trials = StimulusSet({'stimulus_id': ['t1'], 'rules': ['r.png'], 'word_1': ['a.png'], 'word_2': ['b.png']})
    trials.identifier = 'sequence'
    task = TaskContext(response=Choice(['yes', 'no']), present=('image:rules', 'image:word_1', 'image:word_2'))
    session = BufferedSession(stimulus_set=trials, task_context=task,
                              requested_output_channels=['behavior:choice', 'neural:IT'])
    model.participate(session)
    recorded = session.collect('neural:IT')
    assert recorded.dims == ('presentation', 'part', 'neuroid')
    assert list(recorded['part_name'].values) == ['rules', 'word_1', 'word_2']
