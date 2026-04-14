import pytest
from unittest.mock import MagicMock
from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import (
    TaskContext,
    UnifiedModel,
    BrainScoreModel,
)


# -- Helpers ------------------------------------------------------------------

def make_stub_preprocessor(return_value='default_output'):
    """Create a stub preprocessor callable that records its call args."""
    def preprocessor(model, stimuli, *, recording_layer=None, **kwargs):
        preprocessor.call_args = {
            'model': model,
            'stimuli': stimuli,
            'recording_layer': recording_layer,
        }
        return return_value
    preprocessor.call_args = None
    return preprocessor


class StubActivationsModel:
    """Stub activations_model that records calls and returns a fixed value."""

    def __init__(self, return_value='activations_output'):
        self._return_value = return_value
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        self.call_args = {
            'stimuli': stimuli,
            'layers': layers,
        }
        return self._return_value


class StubStimulusSet:
    """Minimal stand-in for StimulusSet with column names."""

    def __init__(self, columns):
        self.columns = columns


# -- TaskContext tests --------------------------------------------------------

class TestTaskContext:

    def test_required_field(self):
        ctx = TaskContext(task_type='classification')
        assert ctx.task_type == 'classification'

    def test_defaults(self):
        ctx = TaskContext(task_type='odd_one_out')
        assert ctx.label_set is None
        assert ctx.fitting_stimuli is None
        assert ctx.instruction is None
        assert ctx.metadata == {}

    def test_all_fields(self):
        fitting = [1, 2, 3]
        ctx = TaskContext(
            task_type='classification',
            label_set=['cat', 'dog'],
            fitting_stimuli=fitting,
            instruction='Pick the animal',
            metadata={'difficulty': 'hard'},
        )
        assert ctx.task_type == 'classification'
        assert ctx.label_set == ['cat', 'dog']
        assert ctx.fitting_stimuli is fitting
        assert ctx.instruction == 'Pick the animal'
        assert ctx.metadata == {'difficulty': 'hard'}

    def test_metadata_default_is_independent(self):
        ctx1 = TaskContext(task_type='a')
        ctx2 = TaskContext(task_type='b')
        ctx1.metadata['key'] = 'value'
        assert 'key' not in ctx2.metadata


# -- UnifiedModel tests ------------------------------------------------------

class ConcreteModel(UnifiedModel):
    """Minimal concrete subclass for testing the ABC."""

    def __init__(self):
        self._task_context = None

    @property
    def identifier(self) -> str:
        return 'test-model'

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return {'V1': 'layer1'}

    @property
    def supported_modalities(self) -> Set[str]:
        return {'vision'}

    def process(self, stimuli):
        return 'processed'


class TestUnifiedModel:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            UnifiedModel()

    def test_concrete_identity(self):
        m = ConcreteModel()
        assert m.identifier == 'test-model'
        assert m.region_layer_map == {'V1': 'layer1'}
        assert m.supported_modalities == {'vision'}

    def test_process(self):
        m = ConcreteModel()
        assert m.process('anything') == 'processed'

    def test_start_task_stores_context(self):
        m = ConcreteModel()
        ctx = TaskContext(task_type='classification')
        m.start_task(ctx)
        assert m._task_context is ctx

    def test_start_recording_is_noop(self):
        m = ConcreteModel()
        m.start_recording('V1')  # should not raise

    def test_reset_is_noop(self):
        m = ConcreteModel()
        m.reset()  # should not raise


# -- BrainScoreModel construction tests ---------------------------------------

class TestBrainScoreModelConstruction:

    def test_basic_construction(self):
        m = BrainScoreModel(
            identifier='resnet-50',
            model='fake_model',
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            preprocessors={'vision': lambda m, s, **kw: None},
        )
        assert m.identifier == 'resnet-50'
        assert m.region_layer_map == {'V1': 'layer1', 'IT': 'layer4'}
        assert m.supported_modalities == {'vision'}

    def test_region_layer_map_returns_copy(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1'},
            preprocessors={'vision': lambda m, s, **kw: None},
        )
        rlm = m.region_layer_map
        rlm['V1'] = 'hacked'
        assert m.region_layer_map['V1'] == 'layer1'

    def test_multiple_modalities(self):
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={},
            preprocessors={
                'vision': lambda m, s, **kw: None,
                'text': lambda m, s, **kw: None,
            },
        )
        assert m.supported_modalities == {'vision', 'text'}

    def test_with_activations_model(self):
        act_model = StubActivationsModel()
        m = BrainScoreModel(
            identifier='resnet',
            model=None,
            region_layer_map={'IT': 'layer4'},
            preprocessors={'vision': lambda m, s, **kw: None},
            activations_model=act_model,
        )
        assert m._activations_model is act_model

    def test_visual_degrees_default(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            preprocessors={'vision': lambda m, s, **kw: None},
        )
        assert m.visual_degrees() == 8

    def test_visual_degrees_custom(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            preprocessors={'vision': lambda m, s, **kw: None},
            visual_degrees=12,
        )
        assert m.visual_degrees() == 12


# -- BrainScoreModel dispatch tests ------------------------------------------

class TestBrainScoreModelDispatch:

    def test_vision_with_activations_model(self):
        """Vision stimuli route through activations_model when available."""
        act_model = StubActivationsModel(return_value='vision_assembly')
        m = BrainScoreModel(
            identifier='resnet',
            model='torch_model',
            region_layer_map={'IT': 'layer4'},
            preprocessors={'vision': make_stub_preprocessor()},
            activations_model=act_model,
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])
        result = m.process(stimuli)
        assert result == 'vision_assembly'
        assert act_model.call_args['stimuli'] is stimuli
        assert act_model.call_args['layers'] == ['layer4']

    def test_vision_without_activations_model_falls_to_preprocessor(self):
        """Vision stimuli fall through to preprocessor if no activations_model."""
        proc = make_stub_preprocessor(return_value='vision_via_preprocessor')
        m = BrainScoreModel(
            identifier='test',
            model='torch_model',
            region_layer_map={'IT': 'layer4'},
            preprocessors={'vision': proc},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        result = m.process(stimuli)
        assert result == 'vision_via_preprocessor'
        assert proc.call_args['model'] == 'torch_model'
        assert proc.call_args['recording_layer'] == 'layer4'

    def test_text_routes_to_preprocessor(self):
        """Text stimuli always route to the text preprocessor callable."""
        text_proc = make_stub_preprocessor(return_value='text_assembly')
        m = BrainScoreModel(
            identifier='clip',
            model='clip_model',
            region_layer_map={'language_system': 'text_model.encoder.layers.10'},
            preprocessors={
                'vision': make_stub_preprocessor(),
                'text': text_proc,
            },
            activations_model=StubActivationsModel(),
        )
        m.start_recording('language_system')
        stimuli = StubStimulusSet(columns=['sentence', 'stimulus_id'])
        result = m.process(stimuli)
        assert result == 'text_assembly'
        assert text_proc.call_args['model'] == 'clip_model'
        assert text_proc.call_args['recording_layer'] == 'text_model.encoder.layers.10'

    def test_recording_layer_resolved_from_region_map(self):
        """start_recording resolves brain region to model layer."""
        act_model = StubActivationsModel()
        m = BrainScoreModel(
            identifier='resnet',
            model='m',
            region_layer_map={'IT': 'layer4'},
            preprocessors={'vision': make_stub_preprocessor()},
            activations_model=act_model,
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        m.process(stimuli)
        assert act_model.call_args['layers'] == ['layer4']

    def test_no_modality_columns_raises(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            preprocessors={'vision': make_stub_preprocessor()},
        )
        stimuli = StubStimulusSet(columns=['unknown_column'])
        with pytest.raises(ValueError, match="No recognized modality columns"):
            m.process(stimuli)

    def test_single_modality_detected_from_multi_preprocessor_model(self):
        """If only one modality is in the stimuli, route to that modality."""
        act_model = StubActivationsModel(return_value='vis_only')
        text_proc = make_stub_preprocessor()
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={},
            preprocessors={
                'vision': make_stub_preprocessor(),
                'text': text_proc,
            },
            activations_model=act_model,
        )
        stimuli = StubStimulusSet(columns=['image_file_name'])
        result = m.process(stimuli)
        assert result == 'vis_only'
        assert text_proc.call_args is None


# -- BrainScoreModel state tests ---------------------------------------------

class TestBrainScoreModelState:

    def test_start_recording_resolves_region(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            preprocessors={'vision': make_stub_preprocessor()},
        )
        m.start_recording('IT')
        assert m._recording_layer == 'layer4'

    def test_start_recording_passes_through_unknown_region(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1'},
            preprocessors={'vision': make_stub_preprocessor()},
        )
        m.start_recording('layer3.2')
        assert m._recording_layer == 'layer3.2'

    def test_reset_clears_state(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': 'layer4'},
            preprocessors={'vision': make_stub_preprocessor()},
        )
        m.start_recording('IT')
        m.start_task(TaskContext(task_type='classification'))
        m.reset()
        assert m._recording_layer is None
        assert m._task_context is None


# -- BrainScoreModel column detection tests -----------------------------------

class TestBrainScoreModelColumnDetection:

    def _make_model(self, *modalities):
        preprocessors = {m: make_stub_preprocessor(f'{m}_out') for m in modalities}
        return BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            preprocessors=preprocessors,
        )

    def test_image_file_name(self):
        m = self._make_model('vision')
        assert m._detect_modalities(StubStimulusSet(['image_file_name'])) == {'vision'}

    def test_image_path(self):
        m = self._make_model('vision')
        assert m._detect_modalities(StubStimulusSet(['image_path'])) == {'vision'}

    def test_filename(self):
        m = self._make_model('vision')
        assert m._detect_modalities(StubStimulusSet(['filename'])) == {'vision'}

    def test_sentence(self):
        m = self._make_model('text')
        assert m._detect_modalities(StubStimulusSet(['sentence'])) == {'text'}

    def test_text_column(self):
        m = self._make_model('text')
        assert m._detect_modalities(StubStimulusSet(['text'])) == {'text'}

    def test_audio_path(self):
        m = self._make_model('audio')
        assert m._detect_modalities(StubStimulusSet(['audio_path'])) == {'audio'}

    def test_video_path(self):
        m = self._make_model('video')
        assert m._detect_modalities(StubStimulusSet(['video_path'])) == {'video'}

    def test_ignores_columns_without_matching_preprocessor(self):
        m = self._make_model('vision')
        detected = m._detect_modalities(StubStimulusSet(['image_file_name', 'sentence']))
        assert detected == {'vision'}

    def test_unknown_columns_ignored(self):
        m = self._make_model('vision')
        detected = m._detect_modalities(StubStimulusSet(['mystery_col']))
        assert detected == set()
