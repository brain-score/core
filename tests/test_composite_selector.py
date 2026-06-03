"""Tests for the three layer-mapping types: standard, 'all' (whole-brain), and
the v1.5 CompositeSelector (units gathered across multiple layers as one region).
"""
import numpy as np
import xarray as xr

from brainscore_core.model_interface import BrainScoreModel, CompositeSelector


def make_stub_preprocessor(return_value='default_output'):
    """A 'vision' preprocessor placeholder; the activations_model does the
    real extraction, so this is only here to register the modality."""
    def preprocessor(model, stimuli, *, recording_layer=None, **kwargs):
        return return_value
    return preprocessor


class StubStimulusSet:
    """Minimal stand-in for StimulusSet with column names."""

    def __init__(self, columns):
        self.columns = columns


class _MultiUnitActivations:
    """Activations stub returning ``units_per_layer`` neuroids per layer, each
    tagged with its 'layer'. Lets composite tests index within a layer."""

    def __init__(self, units_per_layer=4):
        self.units_per_layer = units_per_layer
        self.calls = []

    def __call__(self, stimuli, layers=None, **kwargs):
        layers = list(layers) if layers else ['default']
        self.calls.append(list(layers))
        k = self.units_per_layer
        layer_coord = [L for L in layers for _ in range(k)]
        n = len(layer_coord)
        return xr.DataArray(
            np.zeros((2, n)),
            dims=('presentation', 'neuroid'),
            coords={'layer': ('neuroid', np.array(layer_coord)),
                    'neuroid_id': ('neuroid', np.arange(n))},
        )


def _model(region_layer_map, units_per_layer=4):
    return BrainScoreModel(
        identifier='composite-test', model=None,
        region_layer_map=region_layer_map,
        preprocessors={'vision': make_stub_preprocessor()},
        activations_model=_MultiUnitActivations(units_per_layer),
    )


def _stim():
    return StubStimulusSet(columns=['image_file_name'])


class TestStandardMapping:
    def test_single_region_one_layer(self):
        m = _model({'IT': 'layer.10'})
        m.start_recording('IT')
        a = m.process(_stim())
        assert 'region' not in a.coords            # single-region keeps legacy shape
        assert a.sizes['neuroid'] == 4             # all units of layer.10


class TestAllMapping:
    def test_whole_brain_all_expands_every_region(self):
        m = _model({'V4': 'layer.8', 'IT': 'layer.10'})
        m.start_recording('all')
        a = m.process(_stim())
        assert 'region' in a.coords
        assert set(a['region'].values.tolist()) == {'V4', 'IT'}
        assert a.sizes['neuroid'] == 8             # 4 per layer, both layers


class TestCompositeMapping:
    def test_recording_expands_composite_layers(self):
        sel = CompositeSelector(layers=(('layer.5', None), ('layer.8', None)))
        m = _model({'Vc': sel})
        m.start_recording('Vc')
        assert m._composite_recording is True
        assert m._recording_layers == ['layer.5', 'layer.8']

    def test_composite_all_units(self):
        sel = CompositeSelector(layers=(('layer.5', None), ('layer.8', None)))
        m = _model({'Vc': sel})
        m.start_recording('Vc')
        a = m.process(_stim())
        assert set(a['region'].values.tolist()) == {'Vc'}
        assert a.sizes['neuroid'] == 8
        assert set(a['layer'].values.tolist()) == {'layer.5', 'layer.8'}

    def test_composite_indexed_units(self):
        sel = CompositeSelector(layers=(('layer.5', (0, 1)), ('layer.8', (3,))))
        m = _model({'Vc': sel}, units_per_layer=4)
        m.start_recording('Vc')
        a = m.process(_stim())
        assert a.sizes['neuroid'] == 3
        layers = a['layer'].values.tolist()
        assert layers.count('layer.5') == 2
        assert layers.count('layer.8') == 1
        assert a['region'].values.tolist() == ['Vc', 'Vc', 'Vc']

    def test_composite_mixed_with_standard_region(self):
        m = _model({'IT': 'layer.10',
                    'Vc': CompositeSelector(layers=(('layer.5', (0,)), ('layer.8', (1,))))})
        m.start_recording(['IT', 'Vc'])
        a = m.process(_stim())
        regions = a['region'].values.tolist()
        assert set(regions) == {'IT', 'Vc'}
        assert regions.count('IT') == 4   # all units of layer.10
        assert regions.count('Vc') == 2   # one unit each from layer.5, layer.8
