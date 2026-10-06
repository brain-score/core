"""Exercise plugin command boundaries without creating Conda environments."""
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

import pytest

from brainscore_core.plugin_management.environment_manager import EnvironmentManager
from brainscore_core.plugin_management.test_plugins import PluginTestRunner


@pytest.mark.parametrize('value', ['space in value', "quote's value", '$(touch SHOULD_NOT_EXIST)', 'x; echo unexpected'])
def test_argument_lists_are_passed_literally(tmp_path, monkeypatch, value):
    monkeypatch.setattr(EnvironmentManager, 'get_conda_base', lambda _: str(tmp_path))
    output = tmp_path / 'arguments.json'
    program = 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))'
    result = EnvironmentManager().run_in_env([sys.executable, '-c', program, str(output), value])
    assert result.returncode == 0
    assert json.loads(output.read_text()) == [value]


def test_legacy_shell_commands_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(EnvironmentManager, 'get_conda_base', lambda _: str(tmp_path))
    assert EnvironmentManager().run_in_env('exit 7').returncode == 7


@pytest.fixture
def plugin_environment(tmp_path, monkeypatch):
    # Keep metacharacters in paths to catch accidental shell interpretation.
    library = tmp_path / "checkout space's $literal"
    plugin = library / 'brainscore_dummy' / 'models' / 'dummy'
    plugin.mkdir(parents=True)
    generic = library / 'brainscore_dummy' / 'model_helpers' / 'generic_plugin_tests.py'
    generic.parent.mkdir()
    generic.touch()
    for name in ('test.py', 'setup.py', 'requirements.txt', 'environment.yml'):
        (plugin / name).touch()
    commands = tmp_path / 'commands'
    commands.mkdir()
    log = tmp_path / 'commands.jsonl'
    # Fake external commands record argv; the real shell script still executes.
    program = '''import json, os, sys
from pathlib import Path
name = sys.argv.pop(1)
with open(os.environ['PLUGIN_TEST_COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'python' and sys.argv[1:2] == ['-c']:
    print('3.11.15')
if name == 'pytest':
    sys.exit(int(os.environ.get('PLUGIN_TEST_EXIT', '0')))
'''
    stub = commands / 'record_command.py'
    stub.write_text(program)
    for name in ('conda', 'python', 'pip', 'pytest', 'junitparser'):
        path = commands / name
        path.write_text(
            '#!/bin/sh\nexec ' + shlex.join([sys.executable, str(stub), name]) + ' "$@"\n'
        )
        path.chmod(0o755)
    monkeypatch.setenv('PATH', str(commands) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PLUGIN_TEST_COMMAND_LOG', str(log))
    monkeypatch.setenv('BRAINSCORE_KEEP_PLUGIN_ENV', '1')
    monkeypatch.setenv('PYTEST_SETTINGS', 'not slow and not memory_intense')
    monkeypatch.setenv('XML_FILE', str(library / 'results file.xml'))
    for name in ('TRAVIS', 'OPENMIND', 'PRIVATE_ACCESS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(EnvironmentManager, 'get_conda_base', lambda _: str(tmp_path))
    return plugin, generic, log


@pytest.mark.parametrize('mode', ['default', 'single', 'travis-public', 'travis-private', 'openmind'])
def test_plugin_shell_preserves_paths_and_test_expression(plugin_environment, monkeypatch, mode):
    plugin, generic, log = plugin_environment
    expression = 'test_one or test_two'
    if mode.startswith('travis'):
        monkeypatch.setenv('TRAVIS', '1')
        monkeypatch.setenv('PRIVATE_ACCESS', '1' if mode == 'travis-private' else '0')
    if mode == 'openmind':
        monkeypatch.setenv('OPENMIND', '1')
    runner = PluginTestRunner(plugin, test=expression if mode == 'single' else False)
    runner.run_tests()
    assert runner.returncode == 0
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert ['conda', 'env', 'update', '--file', str(plugin / 'environment.yml')] in calls
    assert ['pip', 'install', str(plugin), '--default-timeout=600', '--retries=5'] in calls
    assert ['pip', 'install', '-r', str(plugin / 'requirements.txt'), '--default-timeout=600', '--retries=5'] in calls
    tests = [call for call in calls if call[0] == 'pytest']
    assert len(tests) == 2
    assert str(generic) in tests[0]
    assert tests[0][tests[0].index('--plugin_directory') + 1] == str(plugin)
    assert str(plugin / 'test.py') in tests[1]
    if mode == 'single':
        assert tests[1][tests[1].index('-k') + 1] == expression
    if mode == 'openmind':
        merged = next(call for call in calls if call[0] == 'junitparser')
        assert merged[2] == os.environ['XML_FILE'] == merged[4]
    if mode == 'travis-public':
        assert tests[1][2].startswith('not private_access and ')
    if mode == 'travis-private':
        assert tests[1][2].startswith('private_access and ')


def test_plugin_shell_preserves_failing_exit_code(plugin_environment, monkeypatch):
    plugin, _, _ = plugin_environment
    monkeypatch.setenv('PLUGIN_TEST_EXIT', '17')
    runner = PluginTestRunner(plugin)
    result = subprocess.run([
        'bash', str(runner.script_path), str(plugin), runner.plugin_name,
        'False', str(runner.library_path), 'False',
    ])
    assert result.returncode == 17
