"""Transfer the same configuration between extraction, routing and sessions."""
import numpy as np
import pytest
import xarray as xr
from brainscore_core.model_interface import BrainScoreModel, LayerSelector, CompositeSelector
from brainscore_core.streaming import InMemorySession, StreamEvent
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class Extractor:
    # No identifier attribute: extraction is an explicit role, not a name heuristic.
    def __init__(self, value=1):
        self.value = value
        self.calls = []

    def __call__(self, stimuli, *, layers):
        self.calls.append(layers)
        return xr.DataArray(
            np.full((len(stimuli), 4 * len(layers)), self.value),
            dims=('presentation', 'neuroid'),
            coords={'layer': ('neuroid', np.repeat(layers, 4)),
                    'neuroid_id': ('neuroid', np.arange(4 * len(layers)))},
        )


def test_composite_mapping_round_trip_preserves_units_and_routing():
    selector = CompositeSelector(layers=(('first', (1,)), ('second', (2,))), modality='text')
    model = BrainScoreModel(identifier='roundtrip', region_layer_map={'language': selector},
                            activations_model={'text': Extractor()})
    model.region_layer_map = model.region_layer_map
    assert model.region_layer_map['language'] == selector
    model.start_recording('language')
    result = model.process(StimulusSet({'sentence': ['hello']}))
    assert result.sizes['neuroid'] == 2
    assert result.layer.values.tolist() == ['first', 'second']
    assert result.unit_index.values.tolist() == [1, 2]


def test_mapping_replacement_drops_removed_routes_and_is_atomic():
    model = BrainScoreModel(identifier='routes', region_layer_map={'IT': 'x'},
                            region_modality_map={'IT': 'vision'},
                            activations_model={'vision': Extractor()})
    before = model.region_layer_map
    with pytest.raises(ValueError, match='no preprocessor or extractor'):
        model.region_layer_map = {'A1': LayerSelector(name='x', modality='audio')}
    assert model.region_layer_map == before
    model.region_layer_map = {'V4': 'y'}
    assert model._region_modality_map == {}


@pytest.mark.parametrize('modality,column', [('vision', 'image_path'), ('text', 'sentence'),
                                           ('audio', 'audio_path'), ('video', 'video_path')])
def test_every_modality_uses_same_extractor_signature(modality, column):
    extractor = Extractor()
    model = BrainScoreModel(identifier='same-api', region_layer_map={'region': 'layer'},
                            activations_model={modality: extractor})
    model.start_recording('region')
    assert model.process(StimulusSet({column: ['input']})).sizes['neuroid'] == 4
    assert extractor.calls == [['layer']]


def test_two_towers_with_same_relative_layer_route_by_selector():
    model = BrainScoreModel(identifier='towers', region_layer_map={
        'IT': LayerSelector(name='encoder.1', modality='vision'),
        'language': LayerSelector(name='encoder.1', modality='text'),
    }, activations_model={'vision': Extractor(1), 'text': Extractor(2)})
    model.start_recording(['IT', 'language'])
    result = model.process(StimulusSet({'image_path': ['image'], 'sentence': ['text']}), multi_modality=True)
    assert result.sel(neuroid=result.region == 'IT').values.tolist() == [[1] * 4]
    assert result.sel(neuroid=result.region == 'language').values.tolist() == [[2] * 4]


@pytest.mark.parametrize('target', ['raw', ['raw']])
def test_raw_layer_has_same_meaning_alone_or_in_list(target):
    model = BrainScoreModel(identifier='raw', activations_model={'text': Extractor()})
    with pytest.warns(UserWarning, match='raw layer'):
        model.start_recording(target)
    assert model.process(StimulusSet({'sentence': ['x']})).layer.values.tolist() == ['raw'] * 4


def test_common_session_collects_neural_output_without_helper_class():
    model = BrainScoreModel(identifier='session', region_layer_map={'IT': 'x'},
                            activations_model={'vision': Extractor()})
    session = InMemorySession(stimulus_set=StimulusSet({'image_path': ['x']}),
                              requested_output_channels=('neural:IT',))
    model.interact(session)
    assert session.collect().shape == (1, 4)


def test_common_session_keeps_custom_events_and_rejects_ambiguous_inputs():
    session = InMemorySession(inputs=[StreamEvent('text', 'hello', 0)])
    session.emit(StreamEvent('custom', 7, 0))
    assert session.collect()[0].payload == 7
    with pytest.raises(ValueError, match='not both'):
        InMemorySession(inputs=[StreamEvent('text', 'hello', 0)], stimulus_set=StimulusSet({'sentence': ['x']}))


def test_explicit_source_is_independent_of_behavioral_task_configuration():
    from brainscore_core.model_interface import TaskContext
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    seen = []
    def generate(model, stimuli, task_context):
        seen.extend(stimuli.sentence)
        return BehavioralAssembly(np.ones((len(stimuli),)), dims=['presentation'],
                                  coords={'stimulus_id': ('presentation', list(stimuli.stimulus_id))})
    task = TaskContext(task_type='label', instruction='choose', label_set=['yes'],
                       metadata={'stimulus_set': StimulusSet({'sentence': ['wrong']})})
    explicit = StimulusSet({'sentence': ['actual'], 'stimulus_id': ['1']})
    # Check the driver directly, so this test isolates task/source composition.
    from brainscore_core.streaming_helpers import _drive_behavior_session_via_process
    from types import SimpleNamespace
    subject = SimpleNamespace(start_task=lambda context: None,
                              process=lambda stimuli: generate(None, stimuli, task))
    session = InMemorySession(stimulus_set=explicit, task_context=task,
                              requested_output_channels=['behavior'])
    _drive_behavior_session_via_process(subject, session)
    assert seen == ['actual']
    assert session.collect().shape == (1,)


def test_shared_extractor_resets_once_across_modalities():
    extractor = Extractor()
    resets = []
    extractor.reset = lambda: resets.append(True)
    model = BrainScoreModel(identifier='shared', activations_model={'vision': extractor, 'text': extractor})
    model.reset()
    assert resets == [True]


def test_variadic_legacy_wrapper_is_normalized_without_identifier_discovery():
    class Wrapper:
        def __call__(self, *args, **kwargs):
            return Extractor()(*args, **kwargs)
    model = BrainScoreModel(identifier='compatibility', preprocessors={'vision': Wrapper()},
                            region_layer_map={'IT': 'layer'})
    model.start_recording('IT')
    assert model.process(StimulusSet({'image_path': ['x']})).shape == (1, 4)


def test_neural_events_work_without_a_stimulus_table():
    model = BrainScoreModel(identifier='events', region_layer_map={'language': 'x'},
                            activations_model={'text': Extractor()})
    session = InMemorySession(inputs=[StreamEvent(channel='text', payload='hello', t_ms=0)],
                              requested_output_channels=['neural:language'])
    model.interact(session)
    assert session.collect().shape == (1, 4)


def test_generation_declares_instruction_input_for_replay():
    model = BrainScoreModel(identifier='answers', generation_fn=lambda *args: 'yes')
    assert 'instruction' in model.in_channels


@pytest.mark.parametrize('streaming', [False, True])
def test_common_session_collects_batch_and_streamed_assemblies(streaming):
    model = BrainScoreModel(identifier='delivery', region_layer_map={'language': 'x'},
                            activations_model={'text': Extractor()})
    session = InMemorySession(
        stimulus_set=StimulusSet({'sentence': ['one', 'two'], 'stimulus_id': ['1', '2']}),
        requested_output_channels=['neural:language'],
        streaming=streaming,
    )
    model.interact(session)
    assert session.collect().shape == (2, 4)



def test_external_selector_routing_does_not_require_a_dataclass():
    from brainscore_core.model_interface import UnitSelector
    class ExternalSelector(UnitSelector):
        @property
        def layer_path(self):
            return 'custom.layer'
    selector = ExternalSelector()
    model = BrainScoreModel(
        identifier='external',
        region_layer_map={'region': selector},
        region_modality_map={'region': 'text'},
        activations_model={'text': Extractor()},
    )
    assert selector.modality is None
    assert isinstance(model.region_layer_map['region'], ExternalSelector)
    model.region_layer_map = model.region_layer_map
    assert model._region_modality_map == {'region': 'text'}
    model.start_recording('region')
    assert model.process(StimulusSet({'sentence': ['x']})).shape == (1, 4)
