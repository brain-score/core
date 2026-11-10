"""Public API guard for ``brainscore_core.model_interface``.

The model-interface module names the subject contract and its events. The
concrete BrainScoreModel and its unit selectors live in the unified
``brainscore`` package. This test snapshots the public surface.
"""

import brainscore_core.model_interface as model_interface


EXPECTED_PUBLIC_NAMES = {
    'ABC',
    'Any',
    'Callable',
    'CameraFrame',
    'Dict',
    'EnvironmentResponse',
    'EnvironmentStep',
    'InputEvent',
    'List',
    'Message',
    'Optional',
    'OutputEvent',
    'Perturbation',
    'PerturbationApplied',
    'Proprioception',
    'Selection',
    'Set',
    'StateChange',
    'Subject',
    'TYPE_CHECKING',
    'TaskContext',
    'Tuple',
    'UnifiedModel',
    'Union',
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


# -- ``extractor`` and ``detect_modalities`` are not Subject members -----------

import subprocess
import sys

import pytest

from brainscore_core.model_interface import Subject


def test_extractor_is_not_a_subject_member():
    assert not hasattr(Subject, 'extractor')
    assert not hasattr(Subject, 'detect_modalities')


# Keep the existing imports usable as implementation files move into packages.
import importlib


_COMPONENT_MODULES = {
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
    'time_axis': '_data.time_axis',
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

    assert Subject.__module__ == TaskContext.__module__ == 'brainscore_core.contract'
    assert StreamEvent.__module__ == 'brainscore_core.streaming'
    assert EnvironmentStep.__module__ == 'brainscore_core.events'
    assert EnvironmentResponse.__module__ == 'brainscore_core.events'
    assert StateChange.__module__ == 'brainscore_core.events'


@pytest.mark.parametrize('module_name,type_name', [
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
