"""Each requested neural channel equals the same channel requested alone.

Two extractors (video and audio, still and audio, ...) have their own time
axes and coordinates. A session that requests regions from both must not
merge their outputs: each channel keeps its own extractor's output exactly.
"""
import numpy as np
import pytest

from brainscore_core import BrainScoreModel, LayerSelector, TaskContext
from brainscore_core._data.assembly_builder import make_assembly, make_layer_neuroid_coords
from brainscore_core._sessions.perception import StreamingStimulusSetSession
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

IDS = ['clip-a', 'clip-b']
RNG = np.random.default_rng(0)


def video(n=4, clip_ms=3000.0, layer='blocks.16'):
    """VideoWrapper._package: four time levels, rich neuroid coords."""
    step = clip_ms / n
    starts = np.arange(n) * step
    return make_assembly(RNG.standard_normal((2, n, 3)).astype(np.float32), {
        'stimulus_id': ('presentation', IDS), 'video_path': ('presentation', ['a.mp4', 'b.mp4']),
        'time_bin_center_ms': ('time_bin', starts + step / 2),
        'time_bin_start_ms': ('time_bin', starts), 'time_bin_end_ms': ('time_bin', starts + step),
        'time_bin_idx': ('time_bin', list(range(n))),
        **make_layer_neuroid_coords('vid', layer, 3)}, ['presentation', 'time_bin', 'neuroid'])


def audio(n=12, step=250.0, bounds=True, layer='encoder.layers.6'):
    """AudioWrapper._pack_3d: three time levels (two without ``_step_ms``)."""
    coords = {'stimulus_id': ('presentation', IDS), 'stimulus_path': ('presentation', ['a.wav', 'b.wav']),
              'time_bin_id': ('time_bin', list(range(n))),
              **make_layer_neuroid_coords('aud', layer, 2, include=('neuroid_id', 'layer'))}
    if bounds:
        coords['time_bin_start_ms'] = ('time_bin', np.arange(n) * step)
        coords['time_bin_end_ms'] = ('time_bin', np.arange(n) * step + step)
    values = RNG.standard_normal((2, n, 2)).astype(np.float32)
    values[1, -2:, :] = np.nan  # genuine padding inside the native output
    return make_assembly(values, coords, ['presentation', 'time_bin', 'neuroid'])


def vlm_video(n=6, step=500.0, layer='blocks.20'):
    """VLMVideoWrapper._package: one presentation coord, two time levels."""
    return NeuroidAssembly(RNG.standard_normal((2, n, 2)).astype(np.float32),
        dims=['presentation', 'time_bin', 'neuroid'], coords={
            'stimulus_id': ('presentation', IDS),
            'time_bin_start_ms': ('time_bin', [k * step for k in range(n)]),
            'time_bin_end_ms': ('time_bin', [(k + 1) * step for k in range(n)]),
            'neuroid_id': ('neuroid', [f'vlm.{layer}.{u}' for u in range(2)]),
            'layer': ('neuroid', [layer] * 2)})


def text_per_token(n=5, layer='h.6'):
    """TextWrapper per_token: token positions, no time bounds."""
    assembly = make_assembly(RNG.standard_normal((2, n, 2)).astype(np.float32), {
        'stimulus_id': ('presentation', IDS), 'sentence': ('presentation', ['x y', 'z w']),
        **make_layer_neuroid_coords('txt', layer, 2)}, ['presentation', 'time_bin', 'neuroid'])
    return assembly.assign_coords(token_position=('time_bin', list(range(n))))


def still(layer='encoder.layers.10'):
    """PytorchWrapper still image: no time axis."""
    return make_assembly(RNG.standard_normal((2, 4)).astype(np.float32), {
        'stimulus_id': ('presentation', IDS), 'image_path': ('presentation', ['a.png', 'b.png']),
        **make_layer_neuroid_coords('img', layer, 4)}, ['presentation', 'neuroid'])


def stimuli():
    return StimulusSet({'stimulus_id': IDS, 'video_path': ['a.mp4', 'b.mp4'],
                        'audio_path': ['a.wav', 'b.wav'], 'image_path': ['a.png', 'b.png'],
                        'sentence': ['x y', 'z w']})


def subject(outputs, regions, calls):
    def extractor(modality, output):
        def run(stimuli, layers):
            calls.append(modality)
            return output.copy(deep=True)
        return run
    return BrainScoreModel('two-extractors', extractor={
        modality: extractor(modality, output) for modality, output in outputs.items()},
        region_layer_map={region: LayerSelector(layer, modality=modality)
                          for region, (layer, modality) in regions.items()})


def run(outputs, regions, channels, calls=None):
    session = BufferedSession(stimulus_set=stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=channels)
    subject(outputs, regions, [] if calls is None else calls).participate(session)
    return {channel: session.collect(channel) for channel in channels}


def assert_same_output(actual, expected):
    assert actual.dims == expected.dims
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.values, expected.values)
    assert set(actual.indexes) == set(expected.indexes)
    for dim, index in expected.indexes.items():
        assert list(actual.indexes[dim].names) == list(index.names), dim
        assert actual.indexes[dim].equals(index), dim
    assert set(actual.coords) == set(expected.coords)
    for name in expected.coords:
        assert actual[name].dims == expected[name].dims, name
        assert actual[name].dtype == expected[name].dtype, name
        np.testing.assert_array_equal(actual[name].values, expected[name].values)
    assert actual.attrs == expected.attrs
    assert actual.identical(expected)


CASES = {
    'video+audio': (lambda: {'video': video(), 'audio': audio()},
                    {'IT': ('blocks.16', 'video'), 'A1': ('encoder.layers.6', 'audio')}),
    'video+audio-without-bounds': (lambda: {'video': video(), 'audio': audio(bounds=False)},
                                   {'IT': ('blocks.16', 'video'), 'A1': ('encoder.layers.6', 'audio')}),
    'vlm-video+audio': (lambda: {'video': vlm_video(), 'audio': audio()},
                        {'IT': ('blocks.20', 'video'), 'A1': ('encoder.layers.6', 'audio')}),
    'still+audio': (lambda: {'vision': still(), 'audio': audio()},
                    {'IT': ('encoder.layers.10', 'vision'), 'A1': ('encoder.layers.6', 'audio')}),
    'video+text-per-token': (lambda: {'video': video(), 'text': text_per_token()},
                             {'IT': ('blocks.16', 'video'), 'language': ('h.6', 'text')}),
    'video+audio-same-grid': (lambda: {'video': video(n=4), 'audio': audio(n=4, step=750.0)},
                              {'IT': ('blocks.16', 'video'), 'A1': ('encoder.layers.6', 'audio')}),
}


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('reverse', [False, True])
def test_each_channel_equals_its_single_channel_output(case, reverse):
    make, regions = CASES[case]
    outputs = make()
    channels = [f'neural:{region}' for region in regions]
    if reverse:
        channels = channels[::-1]
    calls = []
    joint = run(outputs, regions, channels, calls)
    assert sorted(calls) == sorted(outputs), 'each extractor runs exactly once'
    for channel in channels:
        alone = run(outputs, regions, [channel])[channel]
        assert_same_output(joint[channel], alone)


def test_two_regions_of_one_extractor_run_per_channel():
    outputs = {'video': video(layer='blocks.16'), 'audio': audio()}
    regions = {'IT': ('blocks.16', 'video'), 'V4': ('blocks.16', 'video'),
               'A1': ('encoder.layers.6', 'audio')}
    calls = []
    channels = ['neural:IT', 'neural:V4', 'neural:A1']
    joint = run(outputs, regions, channels, calls)
    assert sorted(calls) == ['audio', 'video', 'video']       # one call per channel
    for channel in channels:
        assert_same_output(joint[channel], run(outputs, regions, [channel])[channel])


def test_fused_extractor_runs_per_channel():
    calls = []
    fused_output = video(layer='fused.0')
    def fused(stimuli, layers):
        calls.append(tuple(layers))
        return fused_output.copy(deep=True)
    model = BrainScoreModel('fused', extractor={('video', 'audio'): fused},
        region_layer_map={'IT': LayerSelector('fused.0', modality='video'),
                          'A1': LayerSelector('fused.0', modality='audio')})
    session = BufferedSession(stimulus_set=stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:IT', 'neural:A1'])
    model.participate(session)
    assert len(calls) == 2
    for channel in ('neural:IT', 'neural:A1'):
        np.testing.assert_array_equal(session.collect(channel).values, fused_output.values)


def test_streaming_session_records_each_extractor_separately():
    outputs = {'video': video(), 'audio': audio()}
    regions = {'IT': ('blocks.16', 'video'), 'A1': ('encoder.layers.6', 'audio')}
    table = stimuli()[['stimulus_id', 'video_path', 'audio_path']]
    session = StreamingStimulusSetSession(stimulus_set=table, record=['IT', 'A1'])
    subject(outputs, regions, []).participate(session)
    channels = [event.channel for event in session.emitted]
    assert channels and channels == ['neural:IT', 'neural:A1'] * (len(channels) // 2)
    for event in session.emitted:
        native = outputs['video' if event.channel == 'neural:IT' else 'audio']
        assert event.payload.indexes['time_bin'].equals(native.indexes['time_bin'])


@pytest.mark.parametrize('case', ['video+audio', 'still+audio', 'video+text-per-token',
                                  'video+audio-without-bounds'])
def test_direct_merge_of_different_time_axes_is_refused(case):
    make, regions = CASES[case]
    model = subject(make(), regions, [])
    model.start_recording(list(regions))
    with pytest.raises(ValueError, match='different time axes'):
        model.process(stimuli(), multi_modality=True)


def test_direct_merge_of_one_shared_time_axis_still_works():
    shared = audio(n=4, step=750.0)
    other = shared.copy(deep=True)
    regions = {'IT': ('encoder.layers.6', 'video'), 'A1': ('encoder.layers.6', 'audio')}
    model = subject({'video': other, 'audio': shared}, regions, [])
    model.start_recording(list(regions))
    combined = model.process(stimuli(), multi_modality=True)
    assert combined.sizes['time_bin'] == 4
    assert combined.sizes['neuroid'] == 4


def test_missing_input_is_refused_before_any_extractor_runs():
    calls = []
    outputs = {'video': video(), 'audio': audio()}
    regions = {'IT': ('blocks.16', 'video'), 'A1': ('encoder.layers.6', 'audio')}
    session = BufferedSession(stimulus_set=stimuli()[['stimulus_id', 'video_path']],
                              task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:IT', 'neural:A1'])
    with pytest.raises(ValueError, match='missing input modalities'):
        subject(outputs, regions, calls).participate(session)
    assert calls == []


# --- Layer-aware fixtures: several regions on one extractor -------------------

def layered(kind, calls, numbering='per_layer', label=None):
    """An extractor that returns one block per requested layer.

    ``numbering='across_layers'`` copies VideoWrapper._package, which numbers
    ``neuroid_num`` over all requested layers of one call.
    """
    def extract(stimuli, layers):
        calls.append((kind, tuple(layers)))
        ids = list(stimuli['stimulus_id'])
        units = 2
        values = np.concatenate([
            np.full((len(ids), units), float(sum(map(ord, layer))), dtype=np.float32)
            + np.arange(units, dtype=np.float32) for layer in layers], axis=-1)
        numbers = (list(range(units * len(layers))) if numbering == 'across_layers'
                   else [u for _ in layers for u in range(units)])
        coords = {'stimulus_id': ('presentation', ids),
                  'stimulus_path': ('presentation', [f'{i}.{kind}' for i in ids]),
                  'neuroid_id': ('neuroid', [f'{kind}.{layer}.{u}' for layer in layers for u in range(units)]),
                  'neuroid_num': ('neuroid', numbers),
                  'layer': ('neuroid', [label or layer for layer in layers for _ in range(units)])}
        if kind == 'audio':
            values = np.repeat(values[:, None, :], 3, axis=1)
            coords.update(time_bin_start_ms=('time_bin', [0., 20., 40.]),
                          time_bin_end_ms=('time_bin', [20., 40., 60.]))
            return NeuroidAssembly(values, dims=['presentation', 'time_bin', 'neuroid'], coords=coords)
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords=coords)
    return extract


def layered_subject(calls, numbering='per_layer', composite=False):
    region_map = {'V1': LayerSelector('early', modality='vision'),
                  'V4': LayerSelector('middle', modality='vision'),
                  'IT': LayerSelector('late', modality='vision'),
                  'A1': LayerSelector('auditory', modality='audio')}
    if composite:
        from brainscore_core.selection import CompositeSelector
        region_map['mix'] = CompositeSelector((('early', (1,)), ('late', (0,))), modality='vision')
    return BrainScoreModel('layered', extractor={
        'vision': layered('vision', calls, numbering), 'audio': layered('audio', calls)},
        region_layer_map=region_map)


def layered_stimuli():
    return StimulusSet({'stimulus_id': IDS, 'image_path': ['a.png', 'b.png'],
                        'audio_path': ['a.wav', 'b.wav']})


def layered_run(channels, calls=None, **kwargs):
    model = layered_subject([] if calls is None else calls, **kwargs)
    session = BufferedSession(stimulus_set=layered_stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=channels)
    model.participate(session)
    return {channel: session.collect(channel) for channel in channels}


@pytest.mark.parametrize('channels', [
    ['neural:V1', 'neural:V4', 'neural:IT'],                  # one extractor
    ['neural:V1', 'neural:V4', 'neural:IT', 'neural:A1'],     # whole brain, two extractors
])
def test_several_regions_on_one_extractor_equal_each_region_alone(channels):
    calls = []
    joint = layered_run(channels, calls)
    expected = ['vision'] * len([c for c in channels if c != 'neural:A1']) + (
        ['audio'] if 'neural:A1' in channels else [])
    assert [kind for kind, _ in calls] == expected
    for channel in channels:
        assert_same_output(joint[channel], layered_run([channel])[channel])
        assert 'region' not in joint[channel].coords


def test_default_runs_per_channel_so_request_dependent_numbering_stays_native():
    channels = ['neural:V1', 'neural:V4', 'neural:IT', 'neural:A1']
    joint = layered_run(channels, numbering='across_layers')
    for channel in channels:
        assert_same_output(joint[channel], layered_run([channel], numbering='across_layers')[channel])


def test_composite_regions_keep_base_calls_and_equal_each_region_alone():
    calls = []
    channels = ['neural:IT', 'neural:mix', 'neural:A1']
    joint = layered_run(channels, calls, composite=True)
    assert calls == [('vision', ('late',)), ('vision', ('early', 'late')), ('audio', ('auditory',))]
    for channel in channels:
        assert_same_output(joint[channel], layered_run([channel], composite=True)[channel])


def test_combined_recording_stays_active_after_the_session():
    model = layered_subject([])
    session = BufferedSession(stimulus_set=layered_stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:IT', 'neural:A1'])
    model.participate(session)
    assert model._recorder.recording_regions == ['IT', 'A1']


def test_combined_recording_is_restored_after_a_failure():
    calls = []
    model = layered_subject(calls)
    def broken(stimuli, layers):
        raise RuntimeError('extractor failed')
    model._extractors['audio'] = broken
    session = BufferedSession(stimulus_set=layered_stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:IT', 'neural:A1'])
    with pytest.raises(RuntimeError, match='extractor failed'):
        model.participate(session)
    assert model._recorder.recording_regions == ['IT', 'A1']


def test_request_order_does_not_change_extractor_order():
    calls = []
    channels = ['neural:A1', 'neural:IT']
    layered_run(channels, calls)
    assert [kind for kind, _ in calls] == ['vision', 'audio']


def test_overlapping_fused_extractors_use_the_extractor_dispatch_selects():
    calls = []
    def fused(name):
        def extract(stimuli, layers):
            calls.append(name)
            return still(layer=layers[0])
        return extract
    model = BrainScoreModel('fused-overlap', extractor={('vision', 'text'): fused('partial'),
                                                         ('vision', 'text', 'audio'): fused('full')},
                            region_layer_map={'IT': LayerSelector('visual', modality='vision'),
                                              'A1': LayerSelector('auditory', modality='audio')})
    session = BufferedSession(stimulus_set=stimuli()[['stimulus_id', 'image_path', 'sentence', 'audio_path']],
                              task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:IT', 'neural:A1'])
    model.participate(session)
    assert calls == ['full', 'full']      # per channel, the extractor that real dispatch selects


@pytest.mark.parametrize('channels', [['neural:IT', 'neural:A1'], ['neural:V1', 'neural:V4', 'neural:A1']])
def test_wrong_layer_labels_are_refused_in_a_several_channel_session(channels):
    calls = []
    model = layered_subject(calls)
    model._extractors['audio'] = layered('audio', calls, label='wrong')
    session = BufferedSession(stimulus_set=layered_stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=channels)
    with pytest.raises(ValueError, match="layers \\['wrong'\\]"):
        model.participate(session)


@pytest.mark.parametrize('fault', ['wrong', 'empty', 'presentation'])
def test_one_channel_session_checks_layer_provenance(fault):
    """Changed from base: a one-channel session now refuses outputs whose layer
    labels name no requested layer, that hold no neuroids, or whose layer
    coordinate is not on neuroid."""
    calls = []
    model = layered_subject(calls)
    good = layered('audio', calls, label='wrong' if fault == 'wrong' else None)
    def faulty(stimuli, layers):
        output = good(stimuli, layers)
        if fault == 'empty':
            return output.isel(neuroid=slice(0, 0))
        if fault == 'presentation':
            plain = NeuroidAssembly(output.values, dims=output.dims, coords={
                'stimulus_id': ('presentation', IDS), 'stimulus_path': ('presentation', ['a.wav', 'b.wav']),
                'neuroid_id': ('neuroid', ['x', 'y']),
                'time_bin_start_ms': ('time_bin', [0., 20., 40.]),
                'time_bin_end_ms': ('time_bin', [20., 40., 60.])})
            return plain.assign_coords(layer=('presentation', ['auditory'] * len(IDS)))
        return output
    model._extractors['audio'] = faulty
    session = BufferedSession(stimulus_set=layered_stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:A1'])
    with pytest.raises(ValueError, match='layer|neuroids'):
        model.participate(session)


@pytest.mark.parametrize('channels', [['neural:IT', 'neural:IT'], ['neural:IT', 'neural:A1', 'neural:IT']])
def test_duplicate_channels_are_refused_before_compute(channels):
    calls = []
    model = layered_subject(calls)
    session = BufferedSession(stimulus_set=layered_stimuli(), task_context=TaskContext(task_type='passive'),
                              requested_output_channels=channels)
    with pytest.raises(ValueError, match='Duplicate output channels'):
        model.participate(session)
    assert calls == []


def unindexed(offset, per_presentation):
    assembly = NeuroidAssembly(np.arange(8, dtype='float32').reshape(2, 2, 2),
        dims=['presentation', 'time_bin', 'neuroid'], coords={
            'stimulus_id': ('presentation', IDS),
            'neuroid_id': ('neuroid', [f'unit-{offset}-{i}' for i in range(2)]),
            'layer': ('neuroid', ['layer'] * 2)})
    if per_presentation:
        starts = np.array([[0., 10.], [0., 10.]]) + offset
        return assembly.assign_coords(time_bin_start_ms=(('presentation', 'time_bin'), starts),
                                      time_bin_end_ms=(('presentation', 'time_bin'), starts + 10))
    starts = np.array([0., 10.]) + offset
    assembly = assembly.assign_coords(time_bin_start_ms=('time_bin', starts),
                                      time_bin_end_ms=('time_bin', starts + 10))
    return assembly.reset_index('time_bin') if 'time_bin' in assembly.indexes else assembly


@pytest.mark.parametrize('per_presentation', [True, False])
@pytest.mark.parametrize('offset', [100, 0])
def test_direct_merge_compares_time_bounds_not_only_lengths(per_presentation, offset):
    model = BrainScoreModel('bounds', extractor={
        'text': lambda stimuli, layers: unindexed(0, per_presentation),
        'audio': lambda stimuli, layers: unindexed(offset, per_presentation)},
        region_layer_map={'language': LayerSelector('layer', modality='text'),
                          'A1': LayerSelector('layer', modality='audio')})
    model.start_recording('all')
    table = StimulusSet({'stimulus_id': IDS, 'sentence': ['x', 'y'], 'audio_path': ['a', 'b']})
    if offset:
        with pytest.raises(ValueError, match='different time axes'):
            model.process(table, multi_modality=True)
    else:
        assert model.process(table, multi_modality=True).sizes['time_bin'] == 2


# --- metadata, capabilities, composites, time axes --------------------------

def keyed(name, calls, region_coord=None):
    def extract(stimuli, layers):
        calls.append((name, tuple(layers)))
        ids = list(stimuli['stimulus_id'])
        output = NeuroidAssembly(
            np.concatenate([np.full((len(ids), 2), float(sum(map(ord, l))), dtype=np.float32)
                            for l in layers], axis=1), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', ids),
                    'neuroid_id': ('neuroid', [f'{name}.{l}.{u}' for l in layers for u in range(2)]),
                    'layer': ('neuroid', [l for l in layers for _ in range(2)])})
        if region_coord:
            output = output.assign_coords(region=('neuroid', [region_coord] * output.sizes['neuroid']))
        return output
    return extract


def keyed_session(model, table, regions):
    session = BufferedSession(stimulus_set=table, task_context=TaskContext(task_type='passive'),
                              requested_output_channels=['neural:' + r for r in regions])
    model.participate(session)
    return session


VISUAL = {'V1': LayerSelector('early', modality='vision'), 'IT': LayerSelector('late', modality='vision')}


def test_request_dependent_capability_gives_exact_standalone_values():
    from brainscore_core.capabilities import Capability, register_capability
    from brainscore_core.capabilities.registry import capability_registry
    from brainscore_core.capabilities.neural import NeuralEncodingCapability

    class Center(Capability):
        identifier = 'test-v8-center'
        order = 1
        def enabled_for(self, model): return model.identifier == 'center'
        def handles(self, model, event, **kwargs): return isinstance(event, StimulusSet)
        def process(self, model, event, **kwargs):
            result = NeuralEncodingCapability().process(model, event, **kwargs)
            return result - result.mean('neuroid')
    model = BrainScoreModel('center', extractor={'vision': keyed('v', [])}, region_layer_map=VISUAL)
    register_capability(Center())
    try:
        table = StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']})
        joint = keyed_session(model, table, ['V1', 'IT'])
        for region in ('V1', 'IT'):
            assert_same_output(joint.collect('neural:' + region),
                               keyed_session(model, table, [region]).collect('neural:' + region))
    finally:
        capability_registry.pop(Center.identifier)


def test_source_owned_region_coordinate_stays_exact():
    model = BrainScoreModel('owned', extractor={'vision': keyed('v', [], region_coord='source-owned')},
                            region_layer_map=VISUAL)
    table = StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']})
    joint = keyed_session(model, table, ['V1', 'IT'])
    for region in ('V1', 'IT'):
        assert_same_output(joint.collect('neural:' + region),
                           keyed_session(model, table, [region]).collect('neural:' + region))
        assert set(joint.collect('neural:' + region)['region'].values) == {'source-owned'}


@pytest.mark.parametrize('positions', [[0, 0], [0, 1]])
def test_external_events_with_position_hints_are_never_merged(positions):
    from pathlib import Path
    from brainscore_core.streaming import StreamEvent
    calls = []
    model = BrainScoreModel('ext', extractor={'vision': keyed('v', calls)},
                            region_layer_map={'IT': LayerSelector('late', modality='vision')})
    events = [StreamEvent('vision:still', Path(sid + '.png'), float(i),
                          meta={'column': 'image_path', 'stimulus_id': sid, 'stimulus_position': p,
                                'table_row': ('forged', p)})
              for i, (sid, p) in enumerate(zip(IDS, positions))]
    session = BufferedSession(inputs=events, requested_output_channels=['neural:IT'], streaming=True)
    model.participate(session)
    assert len(calls) == 2
    assert session.collect('neural:IT').sizes['presentation'] == 2


def test_composite_region_needs_every_selected_layer():
    from brainscore_core.selection import CompositeSelector
    model = BrainScoreModel('partial', extractor={'vision': lambda stimuli, layers: keyed('v', [])(stimuli, layers[:1])},
                            region_layer_map={'mix': CompositeSelector((('early', None), ('late', None)),
                                                                       modality='vision')})
    with pytest.raises(ValueError, match='every selected layer|no neuroids'):
        keyed_session(model, StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']}), ['mix'])


@pytest.mark.parametrize('strategy', ['manual', 'xarray'])
def test_time_guard_compares_bounds_by_stimulus_identity(strategy):
    from brainscore_core.assembly_builder import concat_neuroid_assemblies
    def block(ids, layer, starts):
        out = NeuroidAssembly(np.ones((2, 2, 1), dtype='float32'), dims=['presentation', 'time_bin', 'neuroid'],
                              coords={'stimulus_id': ('presentation', ids),
                                      'neuroid_id': ('neuroid', [layer]), 'layer': ('neuroid', [layer])})
        starts = np.asarray(starts)
        return out.assign_coords(time_bin_start_ms=(('presentation', 'time_bin'), starts),
                                 time_bin_end_ms=(('presentation', 'time_bin'), starts + 10))
    same_rows = [[0., 10.], [20., 30.]]
    with pytest.raises(ValueError, match='different time axes'):
        concat_neuroid_assemblies([block(['a', 'b'], 'early', same_rows),
                                   block(['b', 'a'], 'late', same_rows)], strategy=strategy)
    merged = concat_neuroid_assemblies([block(['a', 'b'], 'early', same_rows),
                                        block(['a', 'b'], 'late', same_rows)], strategy=strategy)
    assert merged.sizes['neuroid'] == 2


# --- repeated ids, empty subsets --------------------------------------------

@pytest.mark.parametrize('kind', ['replacement', 'state_dependent'])
def test_replaced_or_state_dependent_handler_gives_exact_standalone_values(kind):
    from brainscore_core.capabilities import Capability, register_capability
    from brainscore_core.capabilities.registry import capability_registry
    from brainscore_core.capabilities.neural import NeuralEncodingCapability

    identifier = NeuralEncodingCapability.identifier if kind == 'replacement' else 'test-v8-state'

    class Center(Capability):
        order = 1
        def enabled_for(self, model): return model.identifier == 'center'
        def handles(self, model, event, **kwargs):
            return isinstance(event, StimulusSet) and (
                kind == 'replacement' or not model._recorder.is_multi_region)
        def process(self, model, event, **kwargs):
            result = NeuralEncodingCapability().process(model, event, **kwargs)
            return result - result.mean('neuroid')
    Center.identifier = identifier
    previous = capability_registry.get(identifier)
    register_capability(Center(), replace=previous is not None)
    calls = []
    model = BrainScoreModel('center', extractor={'vision': keyed('v', calls)}, region_layer_map=VISUAL)
    try:
        table = StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']})
        joint = keyed_session(model, table, ['V1', 'IT'])
        assert len(calls) == 2
        for region in ('V1', 'IT'):
            assert_same_output(joint.collect('neural:' + region),
                               keyed_session(model, table, [region]).collect('neural:' + region))
    finally:
        capability_registry.pop(identifier)
        if previous is not None:
            capability_registry[identifier] = previous


@pytest.mark.parametrize('strategy', ['manual', 'xarray'])
def test_repeated_stimulus_ids_with_identical_presentation_index_merge(strategy):
    from brainscore_core.assembly_builder import concat_neuroid_assemblies
    def block(layer, values):
        out = NeuroidAssembly(np.repeat(np.asarray(values, dtype='float32')[:, None, None], 2, axis=1),
                              dims=['presentation', 'time_bin', 'neuroid'],
                              coords={'stimulus_id': ('presentation', ['same', 'same']),
                                      'trial': ('presentation', [0, 1]),
                                      'neuroid_id': ('neuroid', [layer]), 'layer': ('neuroid', [layer])})
        starts = np.array([[0., 10.], [20., 30.]])
        return out.assign_coords(time_bin_start_ms=(('presentation', 'time_bin'), starts),
                                 time_bin_end_ms=(('presentation', 'time_bin'), starts + 10))
    merged = concat_neuroid_assemblies([block('early', [1, 2]), block('late', [10, 20])], strategy=strategy)
    np.testing.assert_array_equal(merged.values[:, 0, :], [[1, 10], [2, 20]])


def test_composite_with_an_intentionally_empty_layer_subset_is_valid():
    from brainscore_core.selection import CompositeSelector
    model = BrainScoreModel('empty-subset', extractor={'vision': keyed('v', [])},
                            region_layer_map={'mix': CompositeSelector((('early', ()), ('late', None)),
                                                                       modality='vision')})
    result = keyed_session(model, StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']}),
                           ['mix']).collect('neural:mix')
    assert result.sizes['neuroid'] == 2
    assert set(result['layer'].values) == {'late'}


def test_composite_with_no_selected_units_is_refused():
    from brainscore_core.selection import CompositeSelector
    model = BrainScoreModel('no-units', extractor={'vision': keyed('v', [])},
                            region_layer_map={'mix': CompositeSelector((('early', ()), ('late', ())),
                                                                       modality='vision')})
    with pytest.raises(ValueError, match='no neuroids'):
        keyed_session(model, StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']}), ['mix'])



# --- v8: one selected-layer rule, index schema, base streaming -----------------

def test_optional_empty_layer_may_be_absent_from_the_extractor_output():
    from brainscore_core.selection import CompositeSelector
    def without_early(stimuli, layers):
        return keyed('v', [])(stimuli, [layer for layer in layers if layer != 'early'])
    model = BrainScoreModel('optional', extractor={'vision': without_early},
                            region_layer_map={'IT': LayerSelector('late', modality='vision'),
                                              'mix': CompositeSelector((('early', ()), ('late', None)),
                                                                       modality='vision')})
    table = StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']})
    alone = keyed_session(model, table, ['mix']).collect('neural:mix')
    assert alone.sizes['neuroid'] == 2
    assert_same_output(keyed_session(model, table, ['mix', 'IT']).collect('neural:mix'), alone)


def test_virtual_all_uses_the_same_selected_layer_rule():
    from brainscore_core.selection import CompositeSelector
    model = BrainScoreModel('all', extractor={'vision': keyed('v', [])},
                            region_layer_map={'mix': CompositeSelector((('early', ()), ('late', None)),
                                                                       modality='vision')})
    result = keyed_session(model, StimulusSet({'stimulus_id': IDS, 'image_path': ['a', 'b']}),
                           ['all']).collect('neural:all')
    assert set(result['layer'].values) == {'late'}


def test_time_guard_keeps_presentation_index_level_names():
    from brainscore_core._data.time_axis import require_one_time_axis
    def block(layer, trial_name):
        out = NeuroidAssembly(np.ones((2, 2, 1), dtype='float32'), dims=['presentation', 'time_bin', 'neuroid'],
                              coords={'stimulus_id': ('presentation', ['same', 'same']),
                                      trial_name: ('presentation', [0, 1]),
                                      'neuroid_id': ('neuroid', [layer]), 'layer': ('neuroid', [layer])})
        starts = np.array([[0., 10.], [20., 30.]])
        return out.assign_coords(time_bin_start_ms=(('presentation', 'time_bin'), starts),
                                 time_bin_end_ms=(('presentation', 'time_bin'), starts + 10))
    require_one_time_axis([block('early', 'trial'), block('late', 'trial')])
    with pytest.raises(ValueError, match='different time axes'):
        require_one_time_axis([block('early', 'trial'), block('late', 'other_trial')])


def test_streamed_table_with_two_input_columns_presents_each_row_once():
    """Each input column is its own event; the row is still presented once (rule 4.5).

    On base b296d09 this gave four calls and four presentations (A, A, B, B).
    tests/test_stream_rows_once.py covers the fix in full.
    """
    calls = []
    model = BrainScoreModel('stream', extractor={'text': keyed('t', calls)},
                            region_layer_map={'A': LayerSelector('late', modality='text')})
    table = StimulusSet({'stimulus_id': IDS, 'sentence': ['x', 'y'], 'text': ['x', 'y']})
    session = StreamingStimulusSetSession(stimulus_set=table, record='A')
    model.participate(session)
    assert len(calls) == 2
    assert session.collect('neural:A').sizes['presentation'] == 2
