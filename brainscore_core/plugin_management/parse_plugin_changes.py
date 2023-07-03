import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from .test_plugins import run_args

_logger = logging.getLogger(__name__)

PLUGIN_DIRS = ['models', 'benchmarks', 'data', 'metrics']


def get_all_changed_files(commit_SHA: str, comparison_branch='main') -> List[str]:
	"""
	:return: a list of file paths, relative to the library root directory, e.g. `['models/mymodel/__init__.py', 'models/mymodel/model.py', 'models/mymodel/test.py']`
	"""
	core_dir = Path(__file__).parents[2]
	cmd = f'git diff --name-only {comparison_branch} {commit_SHA} -C {core_dir}'
	files_changed_bytes = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.splitlines()
	files_changed = [f.decode() for f in files_changed_bytes]

	return files_changed


def separate_plugin_files(files: List[str]) -> Tuple[List[str], List[str]]:
	"""
	:return: one list of files that are located inside a plugin, and one list of files that are located outside of all plugins, 
		e.g. `['models/mymodel/__init__.py', 'models/mymodel/model.py', 'models/mymodel/test.py'], ['model_helpers/make_model_brainlike.py']`
	"""
	plugin_files = []
	non_plugin_files = []

	for f in files:
		subdir = f.split('/')[1] if len(f.split('/')) > 1 else None
		if not any(plugin_dir == subdir for plugin_dir in PLUGIN_DIRS):
			non_plugin_files.append(f)
		else:
			plugin_files.append(f)

	return plugin_files, non_plugin_files


def _plugin_name_from_path(path_relative_to_library: str) -> str:
    """
    Returns the name of the plugin from the given path. 
    E.g. `_plugin_name_from_path("brainscore_vision/models/mymodel")` will return `"mymodel"`.
    """
    return path_relative_to_library.split('/')[2]


def get_changed_plugin_paths(changed_plugin_files: List[str], domain_root: str) -> Dict[str, List[str]]:
	"""
	Returns a dictionary `plugin_type -> plugin names` with the full names of all changed plugin directories for each plugin_type
	"""
	changed_plugins = {}
	for plugin_type in PLUGIN_DIRS:
		plugin_type_path = f'{domain_root}/{plugin_type}/'
		changed_plugin_paths = [fpath for fpath in changed_plugin_files if fpath.startswith(plugin_type_path)]
		changed_plugins[plugin_type] = list(set([_plugin_name_from_path(fname) 
			for fname in changed_plugin_paths if f'/{plugin_type}/' in fname]))
	return changed_plugins


def get_plugin_ids(plugin_type: str, new_plugin_dirs: List[str], domain_root: str) -> List[str]:
	"""
	Searches all __init.py__ files in `new_plugin_dirs` of `plugin_type` for registered plugins.
	Returns list of identifiers for each registered plugin.
	"""
	plugin_ids = []

	for plugin_dirname in new_plugin_dirs:
		init_file = Path(f'{domain_root}/{plugin_type}/{plugin_dirname}/__init__.py')
		with open(init_file) as f:
			registry_name = plugin_type.strip(
				's') + '_registry'  # remove plural and determine variable name, e.g. "models" -> "model_registry"
			plugin_registrations = [line for line in f if f"{registry_name}["
									in line.replace('\"', '\'')]
			for line in plugin_registrations:
				result = re.search(f'{registry_name}\[.*\]', line)
				identifier = result.group(0)[len(registry_name) + 2:-2] # remove brackets and quotes
				plugin_ids.append(identifier)

	return plugin_ids


def parse_plugin_changes(commit_SHA: str, domain_root: str) -> dict:
	"""
	Return information about which files changed by the invoking PR (compared against main) belong to plugins

	:param commit_SHA: SHA of the invoking PR
	:param domain_root: the root package directory of the repo where the PR originates, either 'brainscore' (vision) or 'brainscore_language' (language)
	"""
	plugin_info_dict = {}
	changed_files = get_all_changed_files(commit_SHA)
	changed_plugin_files, changed_non_plugin_files = get_changed_plugin_files(changed_files)	

	plugin_info_dict["changed_plugins"] = get_changed_plugin_paths(changed_plugin_files, domain_root)
	plugin_info_dict["is_automergeable"] = str(num_changed_non_plugin_files > 0)

	return plugin_info_dict


def get_plugin_info(commit_SHA: str, domain_root: str):
	"""
	If any model or benchmark files changed, get plugin ids and set run_score to "True".
	Otherwise set to "False".
	Print all collected information about plugins.
	"""
	plugin_info_dict = parse_plugin_changes(commit_SHA, domain_root)

	scoring_plugin_types = ("models", "benchmarks")
	plugins_to_score = [plugin_info_dict["changed_plugins"][plugin_type] for plugin_type in scoring_plugin_types]

	if len(plugins_to_score) > 0:
		plugin_info_dict["run_score"] = "True"
		for plugin_type in scoring_plugin_types:
			scoring_plugin_ids = get_plugin_ids(plugin_type, plugin_info_dict["changed_plugins"][plugin_type], domain_root)
			plugin_info_dict[f'new_{plugin_type}'] = ' '.join(scoring_plugin_ids)
	else:
		plugin_info_dict["run_score"] = "False"

	print(plugin_info_dict)


def run_changed_plugin_tests(commit_SHA: str, domain_root: str):
	"""
	Initiates run of all tests in each changed plugin directory
	"""
	plugin_info_dict = parse_plugin_changes(commit_SHA, domain_root)

	tests_to_run = []
	for plugin_type in plugin_info_dict["changed_plugins"]:
		changed_plugins = plugin_info_dict["changed_plugins"][plugin_type]
		for plugin_dirname in changed_plugins:
			root = Path(f'{domain_root}/{plugin_type}/{plugin_dirname}')
			for filepath in root.rglob(r'test*.py'):
				tests_to_run.append(str(filepath))

	print("Running tests for new or modified plugins...")
	run_args('brainscore_language', tests_to_run)
