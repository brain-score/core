"""Public API guard for ``brainscore_core.model_interface``.

The model-interface module is a compatibility facade: collaborators can import
these names from here even as the implementation moves into smaller modules.
This test snapshots the pre-decomposition public surface.
"""

import brainscore_core.model_interface as model_interface


EXPECTED_PUBLIC_NAMES = {
    'ABC',
    'Any',
    'BrainScoreModel',
    'Callable',
    'CameraFrame',
    'CompositeSelector',
    'Dict',
    'EnvironmentResponse',
    'EnvironmentStep',
    'Extractor',
    'FunctionalSelection',
    'IndexSelection',
    'InputEvent',
    'LayerSelector',
    'List',
    'Message',
    'Optional',
    'OutputEvent',
    'Perturbation',
    'PerturbationApplied',
    'Proprioception',
    'RandomSelection',
    'Selection',
    'Set',
    'StateChange',
    'Subject',
    'TYPE_CHECKING',
    'TaskContext',
    'Tuple',
    'UnifiedModel',
    'Union',
    'UnitSelection',
    'UnitSelector',
    'abstractmethod',
    'dataclass',
    'dispatch_metric',
    'field',
    'output_event_kind',
    'replace',
    'warnings',
}


def test_model_interface_public_surface_snapshot():
    public_names = {name for name in dir(model_interface)
                    if not name.startswith('_')}

    assert public_names == EXPECTED_PUBLIC_NAMES
    assert issubclass(model_interface.UnifiedModel, model_interface.Subject)


def test_brainscore_model_dispatch_order_snapshot():
    assert [
        (event_type.__name__, handler_name)
        for event_type, handler_name in model_interface.BrainScoreModel._INPUT_HANDLERS
    ] == [
        ('StateChange', '_dispatch_state_change'),
        ('EnvironmentStep', '_handle_environment_step'),
        ('Message', '_handle_message'),
    ]


# -- Public observation surface: ``extractor`` + ``detect_modalities`` --------
# Core-only fakes (no framework). Observation tools read these instead of the
# private ``_extractors`` / ``_detect_modalities``; neither is a Subject member.

import subprocess
import sys
from types import MappingProxyType

import pytest

from brainscore_core.model_interface import BrainScoreModel, Subject


class _Columns:
    def __init__(self, columns):
        self.columns = columns


class _FakeExtractor:
    """Extractor with the layer-aware signature and an episode reset."""

    def __init__(self, output):
        self.output, self.calls, self.resets = output, [], 0

    def __call__(self, stimuli, layers):
        self.calls.append((stimuli, list(layers)))
        return self.output

    def reset(self):
        self.resets += 1


def _forms():
    vision, text = _FakeExtractor('v-out'), _FakeExtractor('t-out')
    return {
        'extractor': (BrainScoreModel('m', extractor={'vision': vision, 'text': text}),
                      {'vision': vision, 'text': text}),
        'activations_model_mapping': (
            BrainScoreModel('m', activations_model={'vision': vision, 'text': text}),
            {'vision': vision, 'text': text}),
        'shared_activations_model': (
            BrainScoreModel('m', preprocessors={'vision': lambda x: x},
                            activations_model=vision),
            {'vision': vision}),
    }


@pytest.mark.parametrize('form', ['extractor', 'activations_model_mapping',
                                  'shared_activations_model'])
def test_extractor_is_read_only_normalized_mapping(form):
    model, expected = _forms()[form]
    mapping = model.extractor
    assert isinstance(mapping, MappingProxyType)
    assert set(mapping) == set(expected)
    assert all(mapping[k] is expected[k] for k in expected)  # same objects, no copies
    with pytest.raises(TypeError):
        mapping['audio'] = _FakeExtractor('a')
    with pytest.raises(TypeError):
        del mapping['vision']
    with pytest.raises(AttributeError):
        model.extractor = {}
    assert dict(model.extractor) == dict(mapping)  # unchanged by attempts


def test_extractor_is_not_a_subject_member():
    assert not hasattr(Subject, 'extractor')
    assert not hasattr(Subject, 'detect_modalities')


def test_canonical_and_old_forms_declare_same_channels():
    forms = _forms()
    new, old = forms['extractor'][0], forms['activations_model_mapping'][0]
    before = (new.in_channels, new.out_channels, new.supported_modalities)
    new.extractor  # reading the mapping must not change declarations
    assert (new.in_channels, new.out_channels, new.supported_modalities) == before
    assert before == (old.in_channels, old.out_channels, old.supported_modalities)
    assert new.supported_modalities == {'vision', 'text'}


@pytest.mark.parametrize('columns', [
    ['image_file_name'], ['image_path'], ['filename'], ['sentence'], ['text'],
    ['video_path'], ['audio_path'], ['mystery_col'], ['image_file_name', 'sentence'],
])
@pytest.mark.parametrize('form', ['extractor', 'activations_model_mapping',
                                  'shared_activations_model'])
def test_detect_modalities_matches_private_alias(form, columns):
    model = _forms()[form][0]
    stimuli = _Columns(columns)
    assert model.detect_modalities(stimuli) == model._detect_modalities(stimuli)
    assert model.detect_modalities(stimuli) == model._dispatcher.detect_modalities(stimuli)


def test_detect_modalities_private_alias_is_same_function():
    assert BrainScoreModel._detect_modalities is BrainScoreModel.detect_modalities


def test_extractor_mapping_survives_reset_and_reuse():
    vision = _FakeExtractor('v-out')
    model = BrainScoreModel('m', extractor={'vision': vision})
    mapping = model.extractor
    stimuli = _Columns(['image_path'])
    assert model.process(stimuli) == 'v-out'
    model.reset()
    assert vision.resets == 1
    assert model.extractor['vision'] is vision and dict(mapping) == {'vision': vision}
    assert model.process(stimuli) == 'v-out'
    assert [c[0] for c in vision.calls] == [stimuli, stimuli]


def test_extractor_access_imports_no_framework():
    code = (
        "import sys\n"
        "from brainscore_core.model_interface import BrainScoreModel\n"
        "m = BrainScoreModel('m', extractor={'text': lambda stimuli, layers: None})\n"
        "dict(m.extractor); m.detect_modalities(type('S', (), {'columns': ['sentence']})())\n"
        "bad = [n for n in ('torch', 'transformers', 'tensorflow', 'jax') if n in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


# Keep the existing imports usable as implementation files move into packages.
import importlib


_COMPONENT_MODULES = {
    'behavior': '_model.behavior',
    'behavioral': '_model.behavioral',
    'capability_config': '_model.capability_config',
    'dispatch': '_model.dispatch',
    'perturbation': '_model.perturbation',
    'recording': '_model.recording',
    'temporal': '_sessions.temporal',
    'streaming_helpers': '_sessions.helpers',
    'session_view': '_sessions.view',
    'compatibility': '_evaluation.compatibility',
    'execution_plan': '_evaluation.execution_plan',
    'memory': '_evaluation.memory',
    'preflight': '_evaluation.preflight',
    'score_metadata': '_evaluation.score_metadata',
    'score_status': '_evaluation.score_status',
    'legacy_facade': '_evaluation.legacy_facade',
    'nulls': '_evaluation.nulls',
    'assembly_builder': '_data.assembly_builder',
    'clock': '_data.clock',
    'multimodal': '_data.multimodal',
    'text': '_data.text',
    'environment_check': '_runtime.environment_check',
    'extraction_cache': '_runtime.extraction_cache',
    'hf_compat': '_runtime.hf_compat',
    'extensions': '_runtime.extensions',
    '_env_check': '_runtime.environment_check',
}


@pytest.mark.parametrize('name,implementation', _COMPONENT_MODULES.items())
def test_component_imports_share_the_same_module(name, implementation):
    public = importlib.import_module('brainscore_core.' + name)
    organized = importlib.import_module('brainscore_core.' + implementation)
    assert public is organized


@pytest.mark.parametrize('name,implementation', _COMPONENT_MODULES.items())
@pytest.mark.parametrize('first', ['public', 'implementation'])
def test_component_import_order_stays_framework_free(name, implementation, first):
    paths = ['brainscore_core.' + name, 'brainscore_core.' + implementation]
    if first == 'implementation':
        paths.reverse()
    code = (
        'import importlib, sys\n'
        f'first = importlib.import_module({paths[0]!r})\n'
        f'second = importlib.import_module({paths[1]!r})\n'
        'assert first is second\n'
        "assert not {'torch', 'transformers', 'tensorflow', 'jax'} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, '-I', '-c', code], check=True)


def test_moved_dispatcher_shares_nested_cache_state(monkeypatch):
    public = importlib.import_module('brainscore_core.dispatch')
    organized = importlib.import_module('brainscore_core._model.dispatch')
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    with public._activation_cache_disabled():
        with organized._activation_cache_disabled():
            assert public._cache_disable_depth == 2
            assert organized._cache_disable_depth == 2
        assert public._cache_disable_depth == 1
    assert public._cache_disable_depth == 0
    import os
    assert 'RESULTCACHING_DISABLE' not in os.environ


def test_session_facade_preserves_implementation_identity():
    from brainscore_core import streaming_helpers as public
    from brainscore_core._sessions import buffered, drivers, environment, payloads, perception

    pairs = [
        (public.StimulusSetSession, buffered.StimulusSetSession),
        (public.BehavioralSession, buffered.BehavioralSession),
        (public.StateChangeSession, buffered.StateChangeSession),
        (public.neural_response, buffered.neural_response),
        (public.behavioral_response, buffered.behavioral_response),
        (public.apply_state_change, buffered.apply_state_change),
        (public.EnvironmentSession, environment.EnvironmentSession),
        (public.run_environment, environment.run_environment),
        (public.StreamingStimulusSetSession, perception.StreamingStimulusSetSession),
        (public.WindowedStreamSession, perception.WindowedStreamSession),
        (public.RealTimeStreamSession, perception.RealTimeStreamSession),
        (public.drive_neural_session, drivers.drive_neural_session),
        (public.drive_behavior_session, drivers.drive_behavior_session),
        (public.drive_typed_session, drivers.drive_typed_session),
        (public.drain_session_inputs, payloads.drain_session_inputs),
        (public.session_stimuli, payloads.session_stimuli),
    ]
    assert all(left is right for left, right in pairs)
    assert public._drain_stream_events is public.drain_session_inputs
    assert public._stimuli_from_stream_session is public.session_stimuli
    assert public._drive_neural_session_via_process is public.drive_neural_session
    assert public._drive_behavior_session_via_process is public.drive_behavior_session


def test_layout_keeps_recorded_contract_type_names():
    from brainscore_core.contract import TaskContext
    from brainscore_core.events import EnvironmentResponse, EnvironmentStep, StateChange
    from brainscore_core.streaming import StreamEvent

    assert BrainScoreModel.__module__ == 'brainscore_core.brainscore_model'
    assert Subject.__module__ == TaskContext.__module__ == 'brainscore_core.contract'
    assert StreamEvent.__module__ == 'brainscore_core.streaming'
    assert EnvironmentStep.__module__ == 'brainscore_core.events'
    assert EnvironmentResponse.__module__ == 'brainscore_core.events'
    assert StateChange.__module__ == 'brainscore_core.events'


@pytest.mark.parametrize('module_name,type_name', [
    ('behavior', 'ProbabilitiesClassifier'),
    ('behavioral', 'BehavioralReadout'),
    ('compatibility', 'CompatibilityMatch'),
    ('execution_plan', 'ExecutionPlan'),
    ('extraction_cache', 'FileContent'),
    ('legacy_facade', 'LegacyBenchmarkFacade'),
    ('memory', 'MemoryError'),
    ('multimodal', 'MultimodalStimulusSet'),
    ('temporal', 'Window'),
])
def test_moved_types_keep_public_pickle_references(module_name, type_name):
    import pickle

    module = importlib.import_module('brainscore_core.' + module_name)
    definition = vars(module)[type_name]
    assert definition.__module__ == 'brainscore_core.' + module_name
    assert pickle.loads(pickle.dumps(definition)) is definition
