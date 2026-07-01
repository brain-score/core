"""Capability registry tests for BrainScoreModel."""

import importlib
import sys
import textwrap

from brainscore_core.capabilities import capability_registry
from brainscore_core.model_interface import (
    BrainScoreModel,
    EnvironmentResponse,
    EnvironmentStep,
    Message,
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
    TaskContext,
)


def make_stub_preprocessor(return_value='default_output'):
    def preprocessor(model, stimuli, *, recording_layer=None, **kwargs):
        del kwargs
        preprocessor.call_args = {
            'model': model,
            'stimuli': stimuli,
            'recording_layer': recording_layer,
        }
        return return_value
    preprocessor.call_args = None
    return preprocessor


class StubActivationsModel:
    def __init__(self, return_value='activations_output'):
        self._return_value = return_value
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        del kwargs
        self.call_args = {
            'stimuli': stimuli,
            'layers': layers,
        }
        return self._return_value


class StubStimulusSet:
    def __init__(self, columns):
        self.columns = columns


def test_neural_encoding_capability_is_registered_dispatch_path():
    act_model = StubActivationsModel(return_value='vision_assembly')
    model = BrainScoreModel(
        identifier='resnet',
        model='torch_model',
        region_layer_map={'IT': 'layer4'},
        preprocessors={'vision': make_stub_preprocessor()},
        activations_model=act_model,
    )
    model.start_recording('IT')
    stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])

    result = model.process(stimuli)

    assert 'neural-encoding' in capability_registry
    assert result == 'vision_assembly'
    assert act_model.call_args['stimuli'] is stimuli
    assert act_model.call_args['layers'] == ['layer4']
    assert 'neural-encoding' in model._capability_state


def test_behavioral_readout_capability_precedes_neural_dispatch():
    act_model = StubActivationsModel(return_value='vision_assembly')
    model = BrainScoreModel(
        identifier='behavioral',
        model='torch_model',
        region_layer_map={'IT': 'layer4'},
        preprocessors={'vision': make_stub_preprocessor()},
        activations_model=act_model,
    )
    model._task_context = TaskContext(task_type='probabilities')
    model._readout_classifier = object()
    stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])

    calls = []

    def fake_predict_probabilities(input_stimuli):
        calls.append(input_stimuli)
        return 'behavioral_assembly'

    model._predict_probabilities = fake_predict_probabilities

    result = model.process(stimuli)

    assert 'behavioral-readout' in capability_registry
    assert result == 'behavioral_assembly'
    assert calls == [stimuli]
    assert act_model.call_args is None
    assert 'behavioral-readout' in model._capability_state


def test_behavioral_generation_capability_precedes_readout_and_neural():
    act_model = StubActivationsModel(return_value='vision_assembly')
    model = BrainScoreModel(
        identifier='generation',
        model='torch_model',
        region_layer_map={'IT': 'layer4'},
        preprocessors={'vision': make_stub_preprocessor()},
        activations_model=act_model,
    )
    model._task_context = TaskContext(task_type='probabilities')
    model._use_generation_for_task = True
    model._readout_classifier = object()
    stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])

    calls = []

    def fake_generate_predictions(input_stimuli):
        calls.append(input_stimuli)
        return 'generated_behavioral_assembly'

    model._generate_predictions = fake_generate_predictions
    model._predict_probabilities = lambda input_stimuli: 'readout_assembly'

    result = model.process(stimuli)

    assert 'behavioral-generation' in capability_registry
    assert result == 'generated_behavioral_assembly'
    assert calls == [stimuli]
    assert act_model.call_args is None
    assert 'behavioral-generation' in model._capability_state
    assert 'behavioral-readout' not in model._capability_state


def test_embodied_action_capability_dispatches_environment_step():
    calls = []

    def action_fn(step):
        calls.append(step)
        return EnvironmentResponse(action=[0.0, 1.0])

    model = BrainScoreModel(
        identifier='embodied',
        model=None,
        region_layer_map={},
        preprocessors={},
        action_fn=action_fn,
    )
    step = EnvironmentStep(step_num=7)

    result = model.process(step)

    assert 'embodied-action' in capability_registry
    assert isinstance(result, EnvironmentResponse)
    assert result.action == [0.0, 1.0]
    assert calls == [step]
    assert 'embodied-action' in model._capability_state


def test_embodied_action_capability_dispatches_message():
    calls = []

    def action_fn(step):
        calls.append(step)
        return Message(content='pong')

    model = BrainScoreModel(
        identifier='agent',
        model=None,
        region_layer_map={},
        preprocessors={},
        action_fn=action_fn,
    )
    message = Message(content='ping')

    result = model.process(message)

    assert isinstance(result, Message)
    assert result.content == 'pong'
    assert len(calls) == 1
    assert calls[0].observation is message


def test_state_change_capability_dispatches_and_tracks_cleanup():
    cleanup_calls = []

    def state_change_fn(state_change):
        applied = PerturbationApplied(
            handle_id='capability-handle',
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup():
            cleanup_calls.append(applied.handle_id)

        return applied, cleanup

    model = BrainScoreModel(
        identifier='perturbed',
        model=None,
        region_layer_map={},
        preprocessors={},
        state_change_fn=state_change_fn,
    )
    apply_event = StateChange(
        kind='ablation',
        target=Selection(layer='layer4'),
        perturbation=Perturbation(kind='zero'),
    )

    result = model.process(apply_event)

    assert 'state-change' in capability_registry
    assert isinstance(result, PerturbationApplied)
    assert result.handle_id == 'capability-handle'
    assert 'capability-handle' in model._active_perturbations
    assert 'state-change' in model._capability_state

    reset_result = model.process(
        StateChange(kind='reset', handle_id='capability-handle')
    )

    assert reset_result is None
    assert cleanup_calls == ['capability-handle']
    assert model._active_perturbations == {}


def test_capability_config_supplies_builtin_capability_slots():
    calls = []

    def action_fn(step):
        calls.append(step)
        return EnvironmentResponse(action='configured')

    model = BrainScoreModel(
        identifier='configured-agent',
        model=None,
        region_layer_map={},
        preprocessors={},
        capability_config={'action_fn': action_fn},
    )
    step = EnvironmentStep(step_num=2)

    result = model.process(step)

    assert isinstance(result, EnvironmentResponse)
    assert result.action == 'configured'
    assert calls == [step]


def test_legacy_positional_capability_tail_remains_supported():
    def action_fn(step):
        return EnvironmentResponse(action=step.step_num)

    model = BrainScoreModel(
        'legacy-agent',
        None,
        {},
        {},
        None,
        8,
        None,
        None,
        action_fn,
    )

    result = model.process(EnvironmentStep(step_num=5))

    assert isinstance(result, EnvironmentResponse)
    assert result.action == 5


def test_self_registering_capability_file_dispatches_without_host_edits(tmp_path):
    module_path = tmp_path / 'demo_capability_plugin.py'
    module_path.write_text(textwrap.dedent("""
        from brainscore_core.capabilities import Capability, register_capability


        class DemoCapability(Capability):
            identifier = 'test-demo-capability'
            order = 10

            def setup(self, model):
                return {'model_identifier': model.identifier}

            def handles(self, model, event, **kwargs):
                return type(event).__name__ == 'DemoCapabilityEvent'

            def process(self, model, event, **kwargs):
                return model._capability_state[self.identifier]['model_identifier']


        register_capability(DemoCapability())
    """), encoding='utf-8')

    sys.path.insert(0, str(tmp_path))
    try:
        importlib.import_module('demo_capability_plugin')

        class DemoCapabilityEvent:
            pass

        model = BrainScoreModel(
            identifier='demo-model',
            model=None,
            region_layer_map={},
            preprocessors={},
        )

        assert model.process(DemoCapabilityEvent()) == 'demo-model'
    finally:
        capability_registry.pop('test-demo-capability', None)
        sys.path.remove(str(tmp_path))
        sys.modules.pop('demo_capability_plugin', None)
