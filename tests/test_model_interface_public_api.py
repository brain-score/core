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
    'Dict',
    'InputEvent',
    'List',
    'Optional',
    'OutputEvent',
    'Perturbation',
    'PerturbationApplied',
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
import json


_COMPONENT_MODULES = {
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


_IMPORT_ORDER_SCRIPT = """
import importlib, json, sys
modules = json.loads(sys.argv[1])
frameworks = {'torch', 'transformers', 'tensorflow', 'jax'}
failures = []
for name, implementation in modules.items():
    for first in ('public', 'implementation'):
        # Forget the package so each case imports in a clean state.
        for loaded in [m for m in sys.modules if m == 'brainscore_core' or m.startswith('brainscore_core.')]:
            del sys.modules[loaded]
        paths = ['brainscore_core.' + name, 'brainscore_core.' + implementation]
        if first == 'implementation':
            paths.reverse()
        try:
            one = importlib.import_module(paths[0])
            two = importlib.import_module(paths[1])
            assert one is two, 'public and implementation modules differ'
            assert not frameworks & sys.modules.keys(), 'a framework was imported'
        except Exception as error:
            failures.append(f'{name} ({first} first): {type(error).__name__}: {error}')
print(json.dumps(failures))
"""


def test_component_import_order_stays_framework_free():
    # One interpreter checks every module and order; a failure names each one.
    run = subprocess.run([sys.executable, '-I', '-c', _IMPORT_ORDER_SCRIPT, json.dumps(_COMPONENT_MODULES)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    failures = json.loads(run.stdout.strip().splitlines()[-1])
    assert not failures, '\n'.join(failures)


def test_session_facade_preserves_implementation_identity():
    from brainscore_core import streaming_helpers as public
    from brainscore_core._sessions import buffered, drivers, environment, payloads, perception

    pairs = [
        (public.stimulus_session, buffered.stimulus_session),
        (public.behavior_session, buffered.behavior_session),
        (public.neural_response, buffered.neural_response),
        (public.behavioral_response, buffered.behavioral_response),
        (public.apply_state_change, buffered.apply_state_change),
        (public.EnvironmentSource, environment.EnvironmentSource),
        (public.run_environment, environment.run_environment),
        (public.windowed_stream, perception.windowed_stream),
        (public.drive_neural_session, drivers.drive_neural_session),
        (public.drive_behavior_session, drivers.drive_behavior_session),
        (public.drive_typed_session, drivers.drive_typed_session),
        (public.drain_session_inputs, payloads.drain_session_inputs),
        (public.session_stimuli, payloads.session_stimuli),
    ]
    assert all(left is right for left, right in pairs)
    assert public.environment_session is environment.environment_session


def test_removed_session_builders_are_not_public():
    from brainscore_core import streaming_helpers as public
    for name in ('table_session', 'streaming_stimulus_session', 'state_change_session'):
        assert not hasattr(public, name), name
    assert public._drain_stream_events is public.drain_session_inputs
    assert public._stimuli_from_stream_session is public.session_stimuli
    assert public._drive_neural_session_via_process is public.drive_neural_session
    assert public._drive_behavior_session_via_process is public.drive_behavior_session


def test_layout_keeps_recorded_contract_type_names():
    from brainscore_core.contract import TaskContext
    from brainscore_core.events import StateChange
    from brainscore_core.streaming import StreamEvent

    assert Subject.__module__ == TaskContext.__module__ == 'brainscore_core.contract'
    assert StreamEvent.__module__ == 'brainscore_core.streaming'
    assert StateChange.__module__ == 'brainscore_core.events'


@pytest.mark.parametrize('module_name,type_name', [
    ('compatibility', 'CompatibilityMatch'),
    ('execution_plan', 'ExecutionPlan'),
    ('extraction_cache', 'FileContent'),
    ('legacy_facade', 'LegacyBenchmarkFacade'),
    ('memory', 'MemoryError'),
    ('multimodal', 'MultimodalStimulusSet'),
])
def test_moved_types_keep_public_pickle_references(module_name, type_name):
    import pickle

    module = importlib.import_module('brainscore_core.' + module_name)
    definition = vars(module)[type_name]
    assert definition.__module__ == 'brainscore_core.' + module_name
    assert pickle.loads(pickle.dumps(definition)) is definition
