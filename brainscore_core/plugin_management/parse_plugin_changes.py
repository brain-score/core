import json
import os
from pathlib import Path
import re
from typing import List, Tuple, Dict

from .test_plugins import run_args, MODEL_SUBSET

PLUGIN_DIRS = ['models', 'benchmarks', 'data', 'metrics']
SPECIAL_PLUGIN_FILES = ['brainscore_vision/model_interface.py', 'brainscore_language/artificial_subject.py']



def separate_plugin_files(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    :return: one list of files that are located inside a plugin, e.g. `['models/mymodel/__init__.py', 'models/mymodel/model.py', 'models/mymodel/test.py']`,
        one list of files that are not plugin-related, e.g. `['README.md', 'pyproject.toml']`,
        and one list of files that are not inside a plugin but will trigger scoring of all related plugins, e.g. `['model_helpers/make_model_brainlike.py', 'data/__init__.py']`
    """
    plugin_files = []
    non_plugin_files = []
    plugin_related_files = []

    for f in files:
        subdir = f.split('/')[1] if len(f.split('/')) > 1 else None
        if any(plugin_dir == subdir for plugin_dir in PLUGIN_DIRS):
            if len(f.split('/')) == 3 and not os.path.isdir(f):
                plugin_related_files.append(f)
            else:
                plugin_files.append(f)
        elif any(f'{plugin_dir.strip("s")}_helpers' == subdir for plugin_dir in PLUGIN_DIRS):
            plugin_related_files.append(f)
        elif any(special_plugin_file == f for special_plugin_file in SPECIAL_PLUGIN_FILES):
            plugin_related_files.append(f)
        else:
            non_plugin_files.append(f)

    return plugin_files, non_plugin_files, plugin_related_files


def _plugin_name_from_path(path_relative_to_library: str) -> str:
    """
    Returns the name of the plugin from the given path. 
    E.g. `_plugin_name_from_path("brainscore_vision/models/mymodel")` will return `"mymodel"`.
    """
    return path_relative_to_library.split('/')[2]


def get_plugin_paths(plugin_files: List[str], domain_root: str) -> Dict[str, List[str]]:
    """
    Returns a dictionary `plugin_type -> plugin names` with the full names of all plugin directories for each plugin_type
    """
    plugins = {}
    for plugin_type in PLUGIN_DIRS:
        plugin_type_path = f'{domain_root}/{plugin_type}/'
        plugin_paths = [fpath for fpath in plugin_files if fpath.startswith(plugin_type_path)]
        plugins[plugin_type] = list(set([_plugin_name_from_path(fname)
                                         for fname in plugin_paths if f'/{plugin_type}/' in fname]))
    return plugins


def plugin_types_to_test_all(plugin_related_files: List[str]) -> List[str]:
    """
    Returns a list of plugin types for which all plugins should be tested.
    If any of SPECIAL_PLUGIN_FILES is changed, all plugin tests will be run.
    """
    if any(f == special_plugin_file for special_plugin_file in SPECIAL_PLUGIN_FILES for f in plugin_related_files):
        return PLUGIN_DIRS

    plugin_types_changed = []
    for f in plugin_related_files:
        plugin_type = [plugin_dir for plugin_dir in PLUGIN_DIRS if plugin_dir.strip('s') in f]
        assert len(plugin_type) == 1, f"Expected exactly one plugin type to be associated with file {f}"
        plugin_types_changed.append(plugin_type[0])

    # if metric- or data-related files are changed, run all benchmark plugin tests
    if any(plugin_type in plugin_types_changed for plugin_type in ['metrics', 'data']) and (
            'benchmarks' not in plugin_types_changed):
        plugin_types_changed.append('benchmarks')

    return list(set(plugin_types_changed))


def get_plugin_ids(plugin_type: str, new_plugin_dirs: List[str], domain_root: str) -> List[str]:
    """
    Searches all __init.py__ files in `new_plugin_dirs` of `plugin_type` for registered plugins.
    Returns list of identifiers for each registered plugin.

    :param plugin_type: e.g. `models`
    :param new_plugin_dirs: e.g. `['alexnet', 'resnets']`
    :param domain_root: e.g. `/home/me/vision/brainscore_vision`
    """
    plugin_ids = []

    for plugin_dirname in new_plugin_dirs:
        init_file = Path(f'{domain_root}/{plugin_type}/{plugin_dirname}/__init__.py')
        if not init_file.exists():
            print(f"Warning: {init_file} does not exist. Skipping.")
            continue
        with open(init_file) as f:
            registry_name = plugin_type.strip(
                's') + '_registry'  # remove plural and determine variable name, e.g. "models" -> "model_registry"
            plugin_registrations = [line for line in f if f"{registry_name}["
                                    in line.replace('\"', '\'')]
            for line in plugin_registrations:
                result = re.search(f'{registry_name}\[.*\]', line)
                identifier = result.group(0)[len(registry_name) + 2:-2]  # remove brackets and quotes
                plugin_ids.append(identifier)

    return plugin_ids


def parse_plugin_changes(changed_files: str, domain_root: str) -> dict:
    """
    Return information about which files changed by the invoking PR (compared against main) belong to plugins

    :param changed_files: changed file path(s), separated by white space.
    :param domain_root: the root package directory of the repo where the PR originates, either 'brainscore' (vision) or 'brainscore_language' (language)
    """
    assert changed_files, "No files changed"
    assert "fatal" not in changed_files, "Unable to retrieve changed files"
    changed_files_list = changed_files.split()
    changed_plugin_files, changed_non_plugin_files, changed_plugin_related_files = separate_plugin_files(
        changed_files_list)

    plugin_info_dict = {}
    plugin_info_dict["modifies_plugins"] = False if (len(changed_plugin_files) + len(
        changed_plugin_related_files)) == 0 else True
    plugin_info_dict["changed_plugins"] = get_plugin_paths(changed_plugin_files, domain_root)
    plugin_info_dict["test_all_plugins"] = plugin_types_to_test_all(changed_plugin_related_files)
    plugin_info_dict["is_automergeable"] = (len(changed_non_plugin_files) + len(changed_plugin_related_files)) == 0

    return plugin_info_dict


def get_scoring_info(changed_files: str, domain_root: str):
    """
    If any model or benchmark files changed, get plugin ids and set run_score to "True".
    Otherwise set to "False".
    Print all collected information about plugins.
    """
    plugin_info_dict = parse_plugin_changes(changed_files, domain_root)

    scoring_plugin_types = ("models", "benchmarks")
    plugins_to_score = [plugin_info_dict["changed_plugins"][plugin_type] for plugin_type in scoring_plugin_types]

    if any(plugins_to_score):
        plugin_info_dict["run_score"] = "True"
        for plugin_type in scoring_plugin_types:
            scoring_plugin_ids = get_plugin_ids(plugin_type, plugin_info_dict["changed_plugins"][plugin_type],
                                                domain_root)
            plugin_info_dict[f'new_{plugin_type}'] = ' '.join(scoring_plugin_ids)
    else:
        plugin_info_dict["run_score"] = "False"

    plugin_info_json = json.dumps(plugin_info_dict)
    print(plugin_info_json, end="")  # output is accessed via print!


def get_testing_info(changed_files: str, domain_root: str):
    """
    1. Print "True" if PR changes ANY plugin files, else print "False"
    2. Print "True" if PR ONLY changes plugin files, else print "False"
    """
    plugin_info_dict = parse_plugin_changes(changed_files, domain_root)

    print(f'{plugin_info_dict["modifies_plugins"]} {plugin_info_dict["is_automergeable"]}',
          end="")  # output is accessed via print!


def is_plugin_only(changed_files: str, domain_root: str):
    """
    Print "True" if PR ONLY changes plugin files, else print "False"
    """
    plugin_info_dict = parse_plugin_changes(changed_files, domain_root)

    print(f'{plugin_info_dict["is_automergeable"]}', end="")  # output is accessed via print!


def get_test_file_paths(dir_to_search: Path) -> List[str]:
    """
    Returns list of paths to all test files in dir_to_search
    """
    assert dir_to_search.is_dir(), f"Plugin directory {dir_to_search} does not exist"
    return [str(filepath) for filepath in dir_to_search.rglob(r'test*.py')]


def run_changed_plugin_tests(changed_files: str, domain_root: str):
    """
    Initiates run of all tests in each changed plugin directory
    """
    plugin_info_dict = parse_plugin_changes(changed_files, domain_root)
    assert plugin_info_dict["modifies_plugins"], "Expected at least one plugin changed or added, none found."

    tests_to_run = []
    for plugin_type in PLUGIN_DIRS:
        if plugin_type in plugin_info_dict["test_all_plugins"]:
            plugin_type_dir = Path(f'{domain_root}/{plugin_type}')
            for plugin_dir in plugin_type_dir.iterdir():
                if plugin_dir.is_dir():
                    if str(plugin_type_dir) == 'brainscore_vision/models' and plugin_dir.name not in MODEL_SUBSET:  # run subset of models to decrease test time
                        continue
                    tests_to_run.extend(get_test_file_paths(plugin_dir))
        else:
            changed_plugins = plugin_info_dict["changed_plugins"][plugin_type]
            for plugin_dirname in changed_plugins:
                plugin_dir = Path(f'{domain_root}/{plugin_type}/{plugin_dirname}')
                if plugin_dir.is_dir(): tests_to_run.extend(get_test_file_paths(plugin_dir))

    print(f"Running tests for new or modified plugins: {tests_to_run}")
    run_args(domain_root, tests_to_run)
