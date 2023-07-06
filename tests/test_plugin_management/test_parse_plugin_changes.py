import shutil
import sys
import tempfile
from pathlib import Path

from brainscore_core.plugin_management.parse_plugin_changes import get_all_changed_files, separate_plugin_files, get_plugin_paths, get_plugin_ids

DUMMY_FILES_CHANGED = ['brainscore_core/models/dummy_model/model.py', 
                'brainscore_core/models/dummy_model/test.py', 
                'brainscore_core/models/dummy_model/__init__.py',
                'brainscore_core/benchmarks/dummy_benchmark/__init__.py',
                'brainscore_core/__init__.py',
                'brainscore_core/README.md']


def test_branch_git_access():
    import subprocess
    cmd = f'git diff --name-only 1ee0923234bd40126cff0d995d56c608a4a803a1 b55f3f3c5b4f30c0d1963e59f4a65432dfc90c31'
    files_changed_bytes = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.splitlines()
    files_changed = [f.decode() for f in files_changed_bytes]
    assert set(['.travis.yml', 'README.md', 'pyproject.toml', 'setup.py']) == set(files_changed)
    core_dir = Path(__file__).parents[2]
    cmd = f'git diff --name-only 1ee0923234bd40126cff0d995d56c608a4a803a1 b55f3f3c5b4f30c0d1963e59f4a65432dfc90c31 -C {core_dir}'
    files_changed_bytes = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.splitlines()
    files_changed = [f.decode() for f in files_changed_bytes]
    print(core_dir)
    assert set(['.travis.yml', 'README.md', 'pyproject.toml', 'setup.py']) == set(files_changed)

def test_get_all_changed_files():

    commit_sha = '1ee0923234bd40126cff0d995d56c608a4a803a1'
    comparison_branch = 'b55f3f3c5b4f30c0d1963e59f4a65432dfc90c31'
    files_changed = get_all_changed_files(commit_sha, comparison_branch)
    assert set(['.travis.yml', 'README.md', 'pyproject.toml', 'setup.py']) == set(files_changed)


def test_separate_plugin_files():
    plugin_files, non_plugin_files = separate_plugin_files(DUMMY_FILES_CHANGED)
    assert set(['brainscore_core/models/dummy_model/model.py', 
        'brainscore_core/models/dummy_model/test.py', 
        'brainscore_core/models/dummy_model/__init__.py', 
        'brainscore_core/benchmarks/dummy_benchmark/__init__.py']) == set(plugin_files)
    assert set(['brainscore_core/__init__.py', 
        'brainscore_core/README.md']) == set(non_plugin_files)


def test_get_plugin_paths():
    changed_plugins = get_plugin_paths(DUMMY_FILES_CHANGED, 'brainscore_core')
    assert changed_plugins['models'][0] == 'dummy_model'
    assert changed_plugins['benchmarks'][0] == 'dummy_benchmark'
    assert len(changed_plugins['data']) + len(changed_plugins['metrics']) == 0


def test_get_plugin_ids():

    dummy_root = str(Path(__file__).parent / 'test_parse_plugin_changes__brainscore_dummy')
    plugin_types = ['models', 'benchmarks']
    for plugin_type in plugin_types:
        plugin_id = f'dummy_{plugin_type}'.strip('s')
        plugin_ids = get_plugin_ids(plugin_type, [plugin_id], dummy_root)
        assert plugin_id == plugin_id



