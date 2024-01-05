import contextlib
import io
import json
from collections import namedtuple
from pathlib import Path

import pytest

from brainscore_core.plugin_management.parse_plugin_changes import separate_plugin_files, get_plugin_paths, \
    plugin_types_to_test_all, get_plugin_ids, parse_plugin_changes, get_scoring_info, get_testing_info, is_plugin_only, \
    run_changed_plugin_tests

DUMMY_FILES_CHANGED = ['brainscore_core/models/dummy_model/model.py',
                       'brainscore_core/models/dummy_model/test.py',
                       'brainscore_core/models/dummy_model/__init__.py',
                       'brainscore_core/models/__init__.py',
                       'brainscore_core/benchmarks/dummy_benchmark/__init__.py',
                       'brainscore_core/data_helpers/dummy_helper.py',
                       'brainscore_core/__init__.py',
                       'brainscore_core/README.md']

DUMMY_FILES_CHANGED_AUTOMERGEABLE = ['brainscore_core/data/dummy_data/__init__.py',
                                     'brainscore_core/data/dummy_data/data.py']

DUMMY_FILES_CHANGED_NO_PLUGINS = ['brainscore_core/__init__.py',
                                  'brainscore_core/README.md']


class TestSeparatePluginFiles:
    def test_model_benchmark(self):
        plugin_files, non_plugin_files, plugin_related_files = separate_plugin_files(DUMMY_FILES_CHANGED)
        assert {'brainscore_core/models/dummy_model/model.py',
                'brainscore_core/models/dummy_model/test.py',
                'brainscore_core/models/dummy_model/__init__.py',
                'brainscore_core/benchmarks/dummy_benchmark/__init__.py'} == set(plugin_files)
        assert {'brainscore_core/__init__.py', 'brainscore_core/README.md'} == set(non_plugin_files)
        assert {'brainscore_core/models/__init__.py',
                'brainscore_core/data_helpers/dummy_helper.py'} == set(plugin_related_files)

    def test_get_plugin_paths(self):
        plugin_files, non_plugin_files, plugin_related_files = separate_plugin_files(DUMMY_FILES_CHANGED)
        changed_plugins = get_plugin_paths(plugin_files, 'brainscore_core')
        assert changed_plugins['models'][0] == 'dummy_model'
        assert changed_plugins['benchmarks'][0] == 'dummy_benchmark'
        assert len(changed_plugins['data']) + len(changed_plugins['metrics']) == 0

    def test_plugin_types_to_test_all(self):
        plugin_files, non_plugin_files, plugin_related_files = separate_plugin_files(DUMMY_FILES_CHANGED)
        run_all_plugin_tests = plugin_types_to_test_all(plugin_related_files)
        assert {'data', 'benchmarks', 'models'} == set(run_all_plugin_tests)

    def test_plugin_types_to_test_all_special_case(self):
        run_all_plugin_tests = plugin_types_to_test_all(['brainscore_vision/model_interface.py'])
        assert {'models', 'benchmarks', 'data', 'metrics'} == set(run_all_plugin_tests)

    def test_plugin_types_to_test_all_none(self):
        plugin_files, non_plugin_files, plugin_related_files = separate_plugin_files([])
        run_all_plugin_tests = plugin_types_to_test_all(plugin_related_files)
        assert set([]) == set(run_all_plugin_tests)


def test_get_plugin_ids():
    dummy_root = str(Path(__file__).parent / 'test_parse_plugin_changes__brainscore_dummy')
    plugin_types = ['models', 'benchmarks']
    for plugin_type in plugin_types:
        plugin_id = f'dummy_{plugin_type}'.strip('s')
        plugin_ids = get_plugin_ids(plugin_type, [plugin_id], dummy_root)
        assert plugin_ids == [plugin_id.replace("_", "-")]


class TestParsePluginChanges:
    def test_not_automergeable(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED)
        plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')
        assert plugin_info_dict["modifies_plugins"] is True
        assert len(plugin_info_dict["changed_plugins"]["models"]) == 1
        assert plugin_info_dict["changed_plugins"]["models"][0] == "dummy_model"
        assert len(plugin_info_dict["changed_plugins"]["benchmarks"]) == 1
        assert plugin_info_dict["changed_plugins"]["benchmarks"][0] == "dummy_benchmark"
        assert len(plugin_info_dict["changed_plugins"]["data"]) == 0
        assert len(plugin_info_dict["changed_plugins"]["metrics"]) == 0
        assert plugin_info_dict["is_automergeable"] is False

    def test_automergeable(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_AUTOMERGEABLE)
        plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')
        assert plugin_info_dict["modifies_plugins"] is True
        assert len(plugin_info_dict["changed_plugins"]["data"]) == 1
        assert plugin_info_dict["changed_plugins"]["data"][0] == "dummy_data"
        assert plugin_info_dict["is_automergeable"] is True

    def test_no_files_changed(self):
        changed_files = ""
        with pytest.raises(AssertionError):
            parse_plugin_changes(changed_files, 'brainscore_core')

    def test_parse_plugin_changes_no_plugins_changed(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)
        plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')
        all_plugins_changed = [len(plugin_list) for plugin_list in plugin_info_dict["changed_plugins"].values()]
        assert sum(all_plugins_changed) == 0
        assert plugin_info_dict["modifies_plugins"] is False
        assert plugin_info_dict["is_automergeable"] is False


class TestGetScoringInfo:
    def test_scoring_needed(self, mocker):
        changed_files = " ".join(DUMMY_FILES_CHANGED)

        mocked_plugin_ids = ["dummy_plugin1", "dummy_plugin2"]
        get_plugin_ids_mock = mocker.patch("brainscore_core.plugin_management.parse_plugin_changes.get_plugin_ids")
        get_plugin_ids_mock.return_value = mocked_plugin_ids

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            get_scoring_info(changed_files, 'brainscore_core')
        plugin_info_dict = json.loads(f.getvalue())
        print(plugin_info_dict)

        assert plugin_info_dict["run_score"] == str(True)
        assert plugin_info_dict["new_models"] == "dummy_plugin1 dummy_plugin2"

    def test_scoring_not_needed(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            get_scoring_info(changed_files, 'brainscore_core')
        plugin_info_dict = json.loads(f.getvalue())
        print(plugin_info_dict)

        assert plugin_info_dict["run_score"] == str(False)

    def test_testing_needed(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_AUTOMERGEABLE)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            get_testing_info(changed_files, 'brainscore_core')
        return_values = (f.getvalue())

        # First value: modifies_plugins
        # Second value: is_automergeable
        assert return_values == "True True"

    def test_testing_not_needed(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            get_testing_info(changed_files, 'brainscore_core')
        return_values = (f.getvalue())

        # First value: modifies_plugins
        # Second value: is_automergeable
        assert return_values == "False False"


class TestRunChangedPlugins:
    domain_root = str(Path(__file__).parent / 'test_parse_plugin_changes__brainscore_dummy')

    def test_run_changed_plugin_tests_one_benchmark(self, mocker):
        plugin_info_dict_mock = mocker.patch(
            "brainscore_core.plugin_management.parse_plugin_changes.parse_plugin_changes")
        plugin_info_dict_mock.return_value = {
            'modifies_plugins': True, 'test_all_plugins': [],
            'changed_plugins': {'models': [], 'benchmarks': ['dummy_benchmark_2'], 'data': [], 'metrics': []},
            'is_automergeable': False, 'run_score': 'True'}

        run_args_mock = mocker.patch("brainscore_core.plugin_management.parse_plugin_changes.run_args")
        run_args_mock.return_value = "Mock test run"

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            run_changed_plugin_tests('mocked_changed_files',
                                     self.domain_root)
            output = f.getvalue()

        assert "Running tests for new or modified plugins: [" \
               f"'{self.domain_root}/benchmarks/dummy_benchmark_2/test.py']" in output

    def test_run_changed_plugin_tests_one_model(self, mocker):
        plugin_info_dict_mock = mocker.patch(
            "brainscore_core.plugin_management.parse_plugin_changes.parse_plugin_changes")
        plugin_info_dict_mock.return_value = {
            'modifies_plugins': True, 'test_all_plugins': [],
            'changed_plugins': {'models': ['dummy_model'], 'benchmarks': [], 'data': [], 'metrics': []},
            'is_automergeable': False, 'run_score': 'True'}

        run_mock = mocker.patch("brainscore_core.plugin_management.environment_manager.EnvironmentManager.run_in_env")
        MockReturn = namedtuple("MockRunInEnv", field_names=['returncode'])
        run_mock.return_value = MockReturn(returncode=0)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            run_changed_plugin_tests('mocked_changed_files',
                                     self.domain_root)
            output = f.getvalue()

        assert "Running tests for new or modified plugins: [" \
               f"'{self.domain_root}/models/dummy_model/test.py']" in output
        # check generic testing
        assert len(run_mock.call_args[0]) == 1, "expected exactly one positional argument"
        assert len(run_mock.call_args[1]) == 0, "expected no keyword arguments"
        command = run_mock.call_args[0][0]
        print(command)
        command_parts = command.split()
        script_path = str(Path(__file__).parent.parent.parent /
                          'brainscore_core' / 'plugin_management' / 'test_plugin.sh')
        plugin_directory = str(Path(self.domain_root) / 'models' / 'dummy_model')
        plugin_name = 'models__dummy_model'
        single_test = 'False'
        library_path = str(Path(self.domain_root).parent)
        generic_plugin_test = str(Path(self.domain_root) / 'model_helpers' / 'generic_plugin_tests.py')
        assert command_parts == ['bash', script_path,
                                 plugin_directory, plugin_name, single_test, library_path, generic_plugin_test]

    def test_run_changed_plugin_tests_all_models_benchmarks_data(self, mocker):
        plugin_info_dict_mock = mocker.patch(
            "brainscore_core.plugin_management.parse_plugin_changes.parse_plugin_changes")
        plugin_info_dict_mock.return_value = {
            'modifies_plugins': True, 'test_all_plugins': ['data', 'benchmarks', 'models'],
            'changed_plugins': {'models': [], 'benchmarks': [], 'data': [], 'metrics': []},
            'is_automergeable': False, 'run_score': 'True'}

        run_args_mock = mocker.patch("brainscore_core.plugin_management.parse_plugin_changes.run_args")
        run_args_mock.return_value = "Mock test run"

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            run_changed_plugin_tests('mocked_changed_files',
                                     self.domain_root)
            output = f.getvalue()

        assert "Running tests for new or modified plugins: [" \
               f"'{self.domain_root}/models/dummy_model/test.py', " \
               f"'{self.domain_root}/benchmarks/dummy_benchmark/test.py', " \
               f"'{self.domain_root}/benchmarks/dummy_benchmark_2/test.py', " \
               f"'{self.domain_root}/data/dummy_data/test.py']" in output

    def test_model_generic_test_plugin(self):
        """ Test the pytest call to generic model plugin testing,
        but without running the full `test_plugin.sh` file which includes environment installation and other tests """
        plugin_directory = str(Path(self.domain_root) / 'models' / 'dummy_model')
        generic_plugin_test = str(Path(self.domain_root) / 'model_helpers' / 'generic_plugin_tests.py')
        pytest_settings = "not slow"
        command = [
            # this is the command that is built for the generic plugin test inside `test_plugin.sh`
            # to keep things simple and stay in the same python environment, we invoke pytest directly
            # rather than using the shell (and subprocess)
            'pytest', '-m', pytest_settings, "-vv",
            generic_plugin_test, "--plugin_directory", plugin_directory,
            "--log-cli-level=INFO"
        ]
        stdout_stream, stderr_stream = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
            returncode = pytest.main(command[1:] + ["-s"])  # print directly to console
            stdout, stderr = stdout_stream.getvalue(), stderr_stream.getvalue()
        assert returncode == 0, f"stderr: {stderr}"
        # make sure identifier was resolved
        dummy_model_identifier = 'dummy-model'  # inside self.domain_root/models/dummy_model/__init__.py
        assert f"Testing model {dummy_model_identifier}" in stdout


class TestIsPluginOnly:
    def test_true(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_AUTOMERGEABLE)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            is_plugin_only(changed_files, 'brainscore_core')
        return_values = (f.getvalue())

        # First value: modifies_plugins
        assert return_values == "True"

    def test_false(self):
        changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            is_plugin_only(changed_files, 'brainscore_core')
        return_values = (f.getvalue())

        # First value: modifies_plugins
        assert return_values == "False"
