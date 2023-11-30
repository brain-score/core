import json
import contextlib
import io
from pathlib import Path
import pytest

from brainscore_core.plugin_management.parse_plugin_changes import separate_plugin_files, get_plugin_paths, get_plugin_ids, parse_plugin_changes, get_scoring_info, get_testing_info, is_plugin_only, run_changed_plugin_tests

DUMMY_FILES_CHANGED = ['brainscore_core/models/dummy_model/model.py', 
                'brainscore_core/models/dummy_model/test.py', 
                'brainscore_core/models/dummy_model/__init__.py',
                'brainscore_core/benchmarks/dummy_benchmark/__init__.py',
                'brainscore_core/__init__.py',
                'brainscore_core/README.md']

DUMMY_FILES_CHANGED_AUTOMERGEABLE = ['brainscore_core/data/dummy_data/__init__.py',
                                     'brainscore_core/data/dummy_data/data.py']

DUMMY_FILES_CHANGED_NO_PLUGINS = ['brainscore_core/__init__.py',
                                'brainscore_core/README.md']


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
        assert plugin_ids == [plugin_id.replace("_", "-")]


def test_parse_plugin_changes_not_automergeable():
    changed_files = " ".join(DUMMY_FILES_CHANGED)
    plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')
    assert plugin_info_dict["modifies_plugins"] == True
    assert len(plugin_info_dict["changed_plugins"]["models"]) == 1
    assert plugin_info_dict["changed_plugins"]["models"][0] == "dummy_model"
    assert len(plugin_info_dict["changed_plugins"]["benchmarks"]) == 1
    assert plugin_info_dict["changed_plugins"]["benchmarks"][0] == "dummy_benchmark"
    assert len(plugin_info_dict["changed_plugins"]["data"]) == 0
    assert len(plugin_info_dict["changed_plugins"]["metrics"]) == 0
    assert plugin_info_dict["is_automergeable"] == False


def test_parse_plugin_changes_automergeable():
    changed_files = " ".join(DUMMY_FILES_CHANGED_AUTOMERGEABLE)
    plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')
    assert plugin_info_dict["modifies_plugins"] == True
    assert len(plugin_info_dict["changed_plugins"]["data"]) == 1
    assert plugin_info_dict["changed_plugins"]["data"][0] == "dummy_data"
    assert plugin_info_dict["is_automergeable"] == True


def test_parse_plugin_changes_no_files_changed():
    changed_files = ""
    with pytest.raises(AssertionError):
        plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')


def test_parse_plugin_changes_no_plugins_changed():
    changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)
    plugin_info_dict = parse_plugin_changes(changed_files, 'brainscore_core')
    all_plugins_changed = [len(plugin_list) for plugin_list in plugin_info_dict["changed_plugins"].values()]
    assert sum(all_plugins_changed) == 0
    assert plugin_info_dict["modifies_plugins"] == False
    assert plugin_info_dict["is_automergeable"] == False


def test_get_scoring_info_scoring_needed(mocker):
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


def test_get_scoring_info_scoring_not_needed():
    changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        get_scoring_info(changed_files, 'brainscore_core')
    plugin_info_dict = json.loads(f.getvalue())
    print(plugin_info_dict)

    assert plugin_info_dict["run_score"] == str(False)


def test_get_testing_info_testing_needed():
    changed_files = " ".join(DUMMY_FILES_CHANGED_AUTOMERGEABLE)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        get_testing_info(changed_files, 'brainscore_core')
    return_values = (f.getvalue())

    # First value: modifies_plugins
    # Second value: is_automergeable
    assert return_values == "True True"


def test_get_testing_info_testing_not_needed():
    changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        get_testing_info(changed_files, 'brainscore_core')
    return_values = (f.getvalue())

    # First value: modifies_plugins
    # Second value: is_automergeable
    assert return_values == "False False"


def test_run_changed_plugin_tests(mocker):
    changed_files = " ".join(DUMMY_FILES_CHANGED)

    plugin_info_dict_mock = mocker.patch("brainscore_core.plugin_management.parse_plugin_changes.parse_plugin_changes")
    plugin_info_dict_mock.return_value = {'modifies_plugins': True, 'changed_plugins': {'models': ['dummy_model'], 'benchmarks': ['dummy_benchmark'], 'data': [], 'metrics': []}, 'is_automergeable': False, 'run_score': 'True'}

    run_args_mock = mocker.patch("brainscore_core.plugin_management.parse_plugin_changes.run_args")
    run_args_mock.return_value = "Mock test run"
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        run_changed_plugin_tests(changed_files, 'tests/test_plugin_management/test_parse_plugin_changes__brainscore_dummy')
        output = f.getvalue()

    assert "Running tests for new or modified plugins: ['tests/test_plugin_management/test_parse_plugin_changes__brainscore_dummy/models/dummy_model/test.py', 'tests/test_plugin_management/test_parse_plugin_changes__brainscore_dummy/benchmarks/dummy_benchmark/test.py']" in output

def test_is_plugin_only_true():
    changed_files = " ".join(DUMMY_FILES_CHANGED_AUTOMERGEABLE)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        is_plugin_only(changed_files, 'brainscore_core')
    return_values = (f.getvalue())

    # First value: modifies_plugins
    assert return_values == "True"

def test_is_plugin_only_false():
    changed_files = " ".join(DUMMY_FILES_CHANGED_NO_PLUGINS)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        is_plugin_only(changed_files, 'brainscore_core')
    return_values = (f.getvalue())

    # First value: modifies_plugins
    assert return_values == "False"