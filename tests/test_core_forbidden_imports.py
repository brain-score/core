"""Enforcement check E1: core imports no model library and no plugin package.

Contract rule 11.1; Architecture and Ownership Map, section 6, E1.

Signal: every ``brainscore_core/**/*.py`` file is parsed with ``ast`` (never
imported). A file fails when it imports, outside a function body, one of
``FRAMEWORKS``; or imports, at any level, one of ``PACKAGES``. An import inside
a function is an optional dependency and is allowed for the frameworks.

Cannot detect: imports made through ``importlib.import_module``, ``__import__``
or ``exec``; a framework reached through a third package that imports it; an
import inside a function that runs at module load.

The allow-list holds today's violations. It may only shrink: an entry that no
longer violates fails the test, so it must be removed.
"""
import ast
from pathlib import Path

import pytest

import brainscore_core

FRAMEWORKS = {'torch', 'transformers', 'tensorflow', 'jax', 'cv2', 'sklearn'}
PACKAGES = {'brainscore', 'brainscore_vision', 'brainscore_language'}

# (path inside brainscore_core, imported module): reason
ALLOWED = {
    ('plugin_management/generate_benchmark_metadata.py', 'torch.nn'):
        'legacy metadata generator counts torch layers; known violation of rule 11.1',
}


def forbidden_imports(source):
    """Set of imported module names that break E1 in one source file."""
    found = set()

    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                modules = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                modules = [child.module]
            else:
                modules = []
            for module in modules:
                root = module.split('.')[0]
                if root in PACKAGES or (root in FRAMEWORKS and not in_function):
                    found.add(module)
            visit(child, in_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))

    visit(ast.parse(source), False)
    return found


def violations(package_root):
    return {(path.relative_to(package_root).as_posix(), module)
            for path in sorted(Path(package_root).rglob('*.py'))
            for module in forbidden_imports(path.read_text(encoding='utf-8'))}


def check(package_root, allowed):
    found = violations(package_root)
    new = sorted(found - set(allowed))
    stale = sorted(set(allowed) - found)
    assert not new, f'new forbidden imports in core: {new}'
    assert not stale, f'allow-list entries that no longer violate; remove them: {stale}'


def test_core_imports_no_model_library():
    check(Path(brainscore_core.__file__).parent, ALLOWED)


def test_check_fails_on_a_new_violation(tmp_path):
    (tmp_path / 'plugin_management').mkdir()
    (tmp_path / 'plugin_management' / 'generate_benchmark_metadata.py').write_text('import torch.nn as nn\n')
    (tmp_path / 'new.py').write_text('from sklearn.linear_model import Ridge\n')
    with pytest.raises(AssertionError, match='new forbidden imports'):
        check(tmp_path, ALLOWED)


def test_check_fails_on_a_stale_entry(tmp_path):
    (tmp_path / 'clean.py').write_text('import numpy\n')
    with pytest.raises(AssertionError, match='no longer violate'):
        check(tmp_path, ALLOWED)


@pytest.mark.parametrize('source', [
    'import torch',
    'import transformers.models',
    'from sklearn import linear_model',
    'if True:\n    import cv2',
    'def f():\n    import brainscore',
    'from brainscore_vision.models import x',
])
def test_signal_catches(source):
    assert forbidden_imports(source)


@pytest.mark.parametrize('source', [
    'def f():\n    import torch',
    'import brainscore_core.streaming',
    'from . import torch',
    'import numpy',
])
def test_signal_allows(source):
    assert not forbidden_imports(source)
