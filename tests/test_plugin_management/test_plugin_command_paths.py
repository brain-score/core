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
    assert all(call[-2:] == ['-p', 'required_plugin_tests'] for call in tests)
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


def test_plugin_shell_runs_requested_file(plugin_environment):
    plugin, _, log = plugin_environment
    selected = plugin / 'test_second.py'
    selected.touch()
    runner = PluginTestRunner(plugin, test_file=selected)
    runner.run_tests()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert str(selected) in [call for call in calls if call[0] == 'pytest'][-1]


def test_empty_plugin_selection_is_reported_as_failure(plugin_environment, monkeypatch):
    plugin, _, _ = plugin_environment
    monkeypatch.setenv('PLUGIN_TEST_EXIT', '5')
    runner = PluginTestRunner(plugin)
    result = subprocess.run([
        'bash', str(runner.script_path), str(plugin), runner.plugin_name,
        'False', str(runner.library_path), 'False',
    ], capture_output=True, text=True)
    assert result.returncode == 5
    assert 'Plugin-specific tests failed, return code 5' in result.stdout


@pytest.mark.parametrize('codes', [[5], [1, 0], [0, 5], [1, 0, 0]])
def test_plugin_accounting_preserves_earlier_failures(tmp_path, monkeypatch, codes):
    from brainscore_core.plugin_management import test_plugins
    paths = []
    for index in range(len(codes)):
        path = tmp_path / f'test_{index}.py'
        path.touch()
        paths.append(str(path))
    pending = iter(codes)
    monkeypatch.setattr(test_plugins, 'run_specified_tests',
                        lambda **kwargs: {'models__same_plugin': next(pending)})
    with pytest.raises(RuntimeError, match='plugin test runs failed'):
        test_plugins.run_args(tmp_path, test_files=paths)


def test_plugin_accounting_returns_every_successful_run(tmp_path, monkeypatch):
    from brainscore_core.plugin_management import test_plugins
    path = tmp_path / 'test.py'
    path.touch()
    monkeypatch.setattr(test_plugins, 'run_specified_tests', lambda **kwargs: {'models__same': 0})
    result = test_plugins.run_args(tmp_path, test_files=[str(path), str(path)])
    assert len(result) == 2
    assert list(result.values()) == [0, 0]


@pytest.mark.parametrize('selection', [None, []])
def test_plugin_accounting_rejects_no_runs(tmp_path, monkeypatch, selection):
    from brainscore_core.plugin_management import test_plugins
    monkeypatch.setattr(test_plugins, 'run_all_tests', lambda **kwargs: {})
    with pytest.raises(RuntimeError, match='No plugin tests'):
        test_plugins.run_args(tmp_path, test_files=selection)


@pytest.mark.parametrize('body, succeeds', [
    ('pass', True),
    ('pytest.skip("missing dataset")', False),
    ('pytest.xfail("not implemented")', False),
    ('assert False', False),
])
def test_plugin_guard_requires_an_executed_passing_test(tmp_path, body, succeeds):
    from brainscore_core.plugin_management import required_plugin_tests

    (tmp_path / 'test_probe.py').write_text(f'import pytest\ndef test_one():\n    {body}\n')
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               PYTHONPATH=str(Path(required_plugin_tests.__file__).parent))
    env.pop('PYTEST_ADDOPTS', None)
    run = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', 'test_probe.py', '-p', 'required_plugin_tests'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert (run.returncode == 0) is succeeds, run.stdout + run.stderr
