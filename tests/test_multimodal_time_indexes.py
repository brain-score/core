"""Independent sensory clocks survive one public neural session."""
import numpy as np
import pandas as pd
import pytest

from brainscore_core import BrainScoreModel, LayerSelector, TaskContext
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def assembly(modality, starts, ends, *, rich=True):
    count = len(starts)
    values = np.arange(2 * count * 2, dtype=float).reshape(2, count, 2)
    if modality == 'audio':
        values += 100
        values[:, 0, :] = np.nan  # A real invalid time bin must remain, not be trimmed.
    coords = {'stimulus_id': ('presentation', ['clip-a', 'clip-b']),
              'stimulus_path': ('presentation', [f'a.{modality}', f'b.{modality}']),
              'time_bin_start_ms': ('time_bin', starts),
              'time_bin_end_ms': ('time_bin', ends),
              'neuroid_id': ('neuroid', [f'{modality}-0', f'{modality}-1']),
              'layer': ('neuroid', ['vision-layer' if modality == 'video' else 'audio-layer'] * 2)}
    if rich:
        coords['time_bin_center_ms'] = ('time_bin', [(a + b) / 2 for a, b in zip(starts, ends)])
        coords['time_bin_idx'] = ('time_bin', list(range(count)))
        coords['neuroid_num'] = ('neuroid', [0, 1])
        coords['model'] = ('neuroid', [modality] * 2)
    else:
        coords['time_bin_id'] = ('time_bin', list(range(count)))
    return NeuroidAssembly(values, dims=['presentation', 'time_bin', 'neuroid'], coords=coords)


def subject(video, audio):
    calls = []
    def extract(name, output):
        def run(stimuli, layers):
            calls.append((name, tuple(stimuli['stimulus_id'])))
            return output.copy(deep=True)
        return run
    model = BrainScoreModel('paired-temporal-fixture',
        extractor={'video': extract('video', video), 'audio': extract('audio', audio)},
        region_layer_map={'IT': LayerSelector('vision-layer', modality='video'),
                          'A1': LayerSelector('audio-layer', modality='audio')},
        required_modalities={'video', 'audio'})
    return model, calls


def stimuli():
    return StimulusSet({'stimulus_id': ['clip-a', 'clip-b'],
                        'video_path': ['a.mp4', 'b.mp4'], 'audio_path': ['a.wav', 'b.wav']})


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('shared_intervals', [False, True])
def test_neural_channels_keep_native_values_times_and_index_metadata(reverse, shared_intervals):
    starts, ends = [0., 40.], [40., 80.]
    if reverse:
        starts, ends = starts[::-1], ends[::-1]
    video = assembly('video', starts, ends)
    audio = assembly('audio', starts if shared_intervals else [0., 20., 40., 60.],
                     ends if shared_intervals else [20., 40., 60., 80.], rich=False)
    model, calls = subject(video, audio)
    session = BufferedSession(stimulus_set=stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:IT', 'neural:A1'])
    model.participate(session)
    assert calls == [('video', ('clip-a', 'clip-b')), ('audio', ('clip-a', 'clip-b'))]
    for channel, expected in [('neural:IT', video), ('neural:A1', audio)]:
        actual = session.collect(channel)
        np.testing.assert_equal(actual.values, expected.values)
        pd.testing.assert_index_equal(actual.indexes['time_bin'], expected.indexes['time_bin'])
        assert actual.sizes['time_bin'] == expected.sizes['time_bin']
        assert 'source_time_bin_index' not in actual.coords


def test_direct_combined_output_joins_real_intervals_without_interpolation():
    video = assembly('video', [0., 40.], [40., 80.])
    audio = assembly('audio', [0., 20., 40., 60.], [20., 40., 60., 80.], rich=False)
    model, _ = subject(video, audio)
    model.start_recording(['IT', 'A1'])
    combined = model.process(stimuli(), multi_modality=True)
    assert combined.sizes['time_bin'] == 6
    assert combined.indexes['time_bin'].names == ['time_bin_start_ms', 'time_bin_end_ms']
    for modality, expected in [('vision', video), ('audio', audio)]:
        selected = combined.isel(neuroid=np.flatnonzero(combined['modality'].values == modality))
        for index, interval in enumerate(zip(expected['time_bin_start_ms'].values, expected['time_bin_end_ms'].values)):
            np.testing.assert_equal(selected.sel(time_bin=interval).values, expected.isel(time_bin=index).values)
