"""Public API guard for ``brainscore_core``.

The concrete BrainScoreModel and its unit selectors live in the unified
``brainscore`` package. These tests check what a user or a saved record
depends on: Subject members, framework-free imports, and the module names
of recorded types (cache keys and pickles carry them).
"""

# -- ``extractor`` and ``detect_modalities`` are not Subject members -----------

import subprocess
import sys

import pytest

from brainscore_core.contract import Subject


def test_extractor_is_not_a_subject_member():
    assert not hasattr(Subject, 'extractor')
    assert not hasattr(Subject, 'detect_modalities')


# Each module has one import path, its public name, and imports no framework.
import importlib
import json


_PUBLIC_MODULES = [
    'streaming_helpers', 'compatibility', 'execution_plan', 'memory', 'preflight',
    'score_metadata', 'score_status', 'legacy_facade', 'assembly_builder', 'text',
    'time_axis', 'environment_check', 'extraction_cache', 'hf_compat', 'extensions',
]


_IMPORT_SCRIPT = """
import importlib, json, sys
frameworks = {'torch', 'transformers', 'tensorflow', 'jax'}
failures = []
for name in json.loads(sys.argv[1]):
    # Forget the package so each module imports in a clean state.
    for loaded in [m for m in sys.modules if m == 'brainscore_core' or m.startswith('brainscore_core.')]:
        del sys.modules[loaded]
    try:
        importlib.import_module('brainscore_core.' + name)
        assert not frameworks & sys.modules.keys(), 'a framework was imported'
    except Exception as error:
        failures.append(f'{name}: {type(error).__name__}: {error}')
print(json.dumps(failures))
"""


def test_public_modules_import_framework_free():
    # One interpreter checks every module; a failure names each one.
    # -P -s, not -I: -I also drops PYTHONPATH and then tests an installed copy
    # instead of the code under test.
    run = subprocess.run([sys.executable, '-P', '-s', '-c', _IMPORT_SCRIPT,
                          json.dumps(_PUBLIC_MODULES)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    failures = json.loads(run.stdout.strip().splitlines()[-1])
    assert not failures, '\n'.join(failures)


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
])
def test_types_keep_public_pickle_references(module_name, type_name):
    import pickle

    module = importlib.import_module('brainscore_core.' + module_name)
    definition = vars(module)[type_name]
    assert definition.__module__ == 'brainscore_core.' + module_name
    assert pickle.loads(pickle.dumps(definition)) is definition
