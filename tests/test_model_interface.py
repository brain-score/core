import pytest
from unittest.mock import MagicMock
from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import (
    TaskContext,
    UnifiedModel,
    ModalityProcessor,
    ModalityIntegrator,
    BrainScoreModel,
)


# ── Helpers ──────────────────────────────────────────────────────────

class StubProcessor(ModalityProcessor):
    """Minimal processor for testing dispatch logic."""

    def __init__(self, modality_name: str, return_value: Any = 'default_output'):
        self._modality_name = modality_name
        self._return_value = return_value
        self.call_args = None

    @property
    def modality(self) -> str:
        return self._modality_name

    def __call__(self, model, stimuli, *, recording_layer=None,
                 task_context=None, **kwargs):
        self.call_args = {
            'model': model,
            'stimuli': stimuli,
            'recording_layer': recording_layer,
            'task_context': task_context,
        }
        return self._return_value


class StubIntegrator(ModalityIntegrator):
    """Minimal integrator for testing dispatch logic."""

    def __init__(self, return_value: Any = 'integrated_output'):
        self._return_value = return_value
        self.call_args = None

    def integrate(self, modality_features, *, recording_layer=None,
                  task_context=None):
        self.call_args = {
            'modality_features': modality_features,
            'recording_layer': recording_layer,
            'task_context': task_context,
        }
        return self._return_value


class StubStimulusSet:
    """Minimal stand-in for StimulusSet with column names."""

    def __init__(self, columns):
        self.columns = columns


# ── TaskContext tests ────────────────────────────────────────────────

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


# ── UnifiedModel tests ──────────────────────────────────────────────

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


# ── ModalityProcessor tests ─────────────────────────────────────────

class TestModalityProcessor:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ModalityProcessor()

    def test_stub_processor(self):
        p = StubProcessor('vision', return_value='vis_out')
        assert p.modality == 'vision'
        result = p(model='m', stimuli='s', recording_layer='l1')
        assert result == 'vis_out'
        assert p.call_args['model'] == 'm'
        assert p.call_args['recording_layer'] == 'l1'


# ── ModalityIntegrator tests ────────────────────────────────────────

class TestModalityIntegrator:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ModalityIntegrator()

    def test_stub_integrator(self):
        i = StubIntegrator(return_value='fused')
        result = i.integrate({'vision': 'v', 'text': 't'}, recording_layer='l5')
        assert result == 'fused'
        assert i.call_args['modality_features'] == {'vision': 'v', 'text': 't'}
        assert i.call_args['recording_layer'] == 'l5'


# ── BrainScoreModel tests ───────────────────────────────────────────

class TestBrainScoreModelConstruction:

    def test_basic_construction(self):
        p = StubProcessor('vision')
        m = BrainScoreModel(
            identifier='resnet-50',
            model='fake_model',
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            processors=[p],
        )
        assert m.identifier == 'resnet-50'
        assert m.region_layer_map == {'V1': 'layer1', 'IT': 'layer4'}
        assert m.supported_modalities == {'vision'}

    def test_region_layer_map_returns_copy(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1'},
            processors=[StubProcessor('vision')],
        )
        rlm = m.region_layer_map
        rlm['V1'] = 'hacked'
        assert m.region_layer_map['V1'] == 'layer1'

    def test_multiple_processors(self):
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={},
            processors=[StubProcessor('vision'), StubProcessor('text')],
            primary_processor='vision',
        )
        assert m.supported_modalities == {'vision', 'text'}

    def test_invalid_primary_processor_raises(self):
        with pytest.raises(ValueError, match="primary_processor 'audio'"):
            BrainScoreModel(
                identifier='bad',
                model=None,
                region_layer_map={},
                processors=[StubProcessor('vision')],
                primary_processor='audio',
            )

    def test_primary_and_integrator_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            BrainScoreModel(
                identifier='bad',
                model=None,
                region_layer_map={},
                processors=[StubProcessor('vision'), StubProcessor('text')],
                integrator=StubIntegrator(),
                primary_processor='vision',
            )


class TestBrainScoreModelDispatch:

    def test_case1_single_modality_vision(self):
        """Case 1: single modality dispatches to the one processor."""
        proc = StubProcessor('vision', return_value='vision_assembly')
        m = BrainScoreModel(
            identifier='resnet',
            model='torch_model',
            region_layer_map={'IT': 'layer4'},
            processors=[proc],
        )
        stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])
        result = m.process(stimuli)
        assert result == 'vision_assembly'
        assert proc.call_args['model'] == 'torch_model'
        assert proc.call_args['stimuli'] is stimuli

    def test_case1_single_modality_text(self):
        proc = StubProcessor('text', return_value='text_assembly')
        m = BrainScoreModel(
            identifier='gpt2',
            model='hf_model',
            region_layer_map={},
            processors=[proc],
        )
        stimuli = StubStimulusSet(columns=['sentence', 'stimulus_id'])
        result = m.process(stimuli)
        assert result == 'text_assembly'

    def test_case1_passes_recording_layer(self):
        proc = StubProcessor('vision')
        m = BrainScoreModel(
            identifier='resnet',
            model='m',
            region_layer_map={'IT': 'layer4'},
            processors=[proc],
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        m.process(stimuli)
        assert proc.call_args['recording_layer'] == 'layer4'

    def test_case1_passes_task_context(self):
        proc = StubProcessor('vision')
        m = BrainScoreModel(
            identifier='resnet',
            model='m',
            region_layer_map={},
            processors=[proc],
        )
        ctx = TaskContext(task_type='classification', label_set=['a', 'b'])
        m.start_task(ctx)
        stimuli = StubStimulusSet(columns=['image_file_name'])
        m.process(stimuli)
        assert proc.call_args['task_context'] is ctx

    def test_case2_integrator(self):
        """Case 2: multiple modalities + integrator -> extract then fuse."""
        vis_proc = StubProcessor('vision', return_value='vis_features')
        txt_proc = StubProcessor('text', return_value='txt_features')
        integrator = StubIntegrator(return_value='integrated')
        m = BrainScoreModel(
            identifier='tribev2',
            model='backbone',
            region_layer_map={'IT': 'integration.layer4'},
            processors=[vis_proc, txt_proc],
            integrator=integrator,
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        result = m.process(stimuli)
        assert result == 'integrated'
        # Processors should NOT get recording_layer (they use backbone defaults)
        assert vis_proc.call_args['recording_layer'] is None
        assert txt_proc.call_args['recording_layer'] is None
        # Integrator SHOULD get recording_layer
        assert integrator.call_args['recording_layer'] == 'integration.layer4'
        assert set(integrator.call_args['modality_features'].keys()) == {'vision', 'text'}

    def test_case3_primary_processor_vlm(self):
        """Case 3: multiple modalities + primary_processor -> dispatch to primary."""
        vis_proc = StubProcessor('vision', return_value='vlm_output')
        txt_proc = StubProcessor('text', return_value='unused')
        m = BrainScoreModel(
            identifier='qwen-vl',
            model='vlm_model',
            region_layer_map={'IT': 'visual.blocks.26'},
            processors=[vis_proc, txt_proc],
            primary_processor='vision',
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        result = m.process(stimuli)
        assert result == 'vlm_output'
        # Primary processor gets recording_layer
        assert vis_proc.call_args['recording_layer'] == 'visual.blocks.26'
        # Text processor should NOT have been called
        assert txt_proc.call_args is None

    def test_no_modality_columns_raises(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            processors=[StubProcessor('vision')],
        )
        stimuli = StubStimulusSet(columns=['unknown_column'])
        with pytest.raises(ValueError, match="No recognized modality columns"):
            m.process(stimuli)

    def test_multi_modality_no_integrator_no_primary_raises(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            processors=[StubProcessor('vision'), StubProcessor('text')],
        )
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        with pytest.raises(ValueError, match="neither an integrator nor a primary_processor"):
            m.process(stimuli)

    def test_single_modality_detected_from_multi_processor_model(self):
        """If only one modality is in the stimuli, use Case 1 even with multiple processors."""
        vis_proc = StubProcessor('vision', return_value='vis_only')
        txt_proc = StubProcessor('text')
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={},
            processors=[vis_proc, txt_proc],
            primary_processor='vision',
        )
        stimuli = StubStimulusSet(columns=['image_file_name'])
        result = m.process(stimuli)
        assert result == 'vis_only'
        assert txt_proc.call_args is None


class TestBrainScoreModelState:

    def test_start_recording_resolves_region(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            processors=[StubProcessor('vision')],
        )
        m.start_recording('IT')
        assert m._recording_layer == 'layer4'

    def test_start_recording_passes_through_unknown_region(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1'},
            processors=[StubProcessor('vision')],
        )
        m.start_recording('layer3.2')
        assert m._recording_layer == 'layer3.2'

    def test_reset_clears_state(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': 'layer4'},
            processors=[StubProcessor('vision')],
        )
        m.start_recording('IT')
        m.start_task(TaskContext(task_type='classification'))
        m.reset()
        assert m._recording_layer is None
        assert m._task_context is None


class TestBrainScoreModelColumnDetection:

    def _make_model(self, *modalities):
        procs = [StubProcessor(m, return_value=f'{m}_out') for m in modalities]
        return BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            processors=procs,
            primary_processor=modalities[0] if len(modalities) > 1 else None,
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

    def test_ignores_columns_without_matching_processor(self):
        m = self._make_model('vision')
        detected = m._detect_modalities(StubStimulusSet(['image_file_name', 'sentence']))
        assert detected == {'vision'}

    def test_unknown_columns_ignored(self):
        m = self._make_model('vision')
        detected = m._detect_modalities(StubStimulusSet(['mystery_col']))
        assert detected == set()
