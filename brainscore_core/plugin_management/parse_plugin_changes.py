import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from .test_plugins import run_args

PLUGIN_DIRS = ['models', 'benchmarks', 'data', 'metrics']


def get_all_changed_files(commit_SHA: str) -> List[str]:
	cmd = f'git diff --name-only main {commit_SHA}'
	files_changed_bytes = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.splitlines()
	files_changed = [f.decode() for f in files_changed_bytes]

	return files_changed


def get_changed_plugin_files(changed_files: str) -> Tuple[List[str], List[str]]:
	changed_files_list = changed_files.split() if type(changed_files) == str else changed_files

	changed_plugin_files = []
	changed_non_plugin_files = []

	for f in changed_files_list:
		subdir = f.split('/')[1] if len(f.split('/')) > 1 else None
		if not any(plugin_dir == subdir for plugin_dir in PLUGIN_DIRS):
			changed_non_plugin_files.append(f)
		else:
			changed_plugin_files.append(f)

	return changed_plugin_files, changed_non_plugin_files


def is_automergeable(plugin_info_dict: dict, num_changed_non_plugin_files: int):
	""" 
	Stores `plugin_info_dict['is_plugin_only']` `"True"` or `"False"` 
	depending on whether there are any changed non-plugin files.
	"""
	plugin_info_dict["is_plugin_only"] = "False" if num_changed_non_plugin_files > 0 else "True"


def _plugin_name_from_path(path_relative_to_library: str) -> str:
    """
    Returns the name of the plugin from the given path. 
    E.g. `_plugin_name_from_path("brainscore_vision/models/mymodel")` will return `"mymodel"`.
    """
    return path_relative_to_library.split('/')[2]


def get_changed_plugin_paths(plugin_info_dict: dict, changed_plugin_files: List[str], domain_root: str):
	"""
	Adds full path (rel. to library) of all changed plugin directories for each plugin_type to plugin_info_dict
	"""
	plugin_info_dict["changed_plugins"] = {}
	for plugin_type in PLUGIN_DIRS:
		plugin_type_path = f'{domain_root}/{plugin_type}/'
		changed_plugin_paths = [fpath for fpath in changed_plugin_files if fpath.startswith(plugin_type_path)]
		plugin_info_dict["changed_plugins"][plugin_type] = list(set([_plugin_name_from_path(fname) 
			for fname in changed_plugin_paths if f'/{plugin_type}/' in fname]))


def _get_plugin_ids(plugin_type: str, new_plugin_dirs: List[str], domain_root: str) -> List[str]:
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
	is_automergeable(plugin_info_dict, len(changed_non_plugin_files))
	get_changed_plugin_paths(plugin_info_dict, changed_plugin_files, domain_root)

	return plugin_info_dict

def get_scoring_info(commit_SHA: str, domain_root: str):
	"""
	Get 
	If any model or benchmark files changed, get plugin ids and set run_score to "True"
	Otherwise set else "False"
	"""
	plugin_info_dict = parse_plugin_changes(commit_SHA, domain_root)

	scoring_plugin_types = ("models", "benchmarks")
	plugins_to_score = [plugin_info_dict["changed_plugins"][plugin_type] for plugin_type in scoring_plugin_types]

	if len(plugins_to_score) > 0:
		plugin_info_dict["run_score"] = "True"
		for plugin_type in scoring_plugin_types:
			scoring_plugin_ids = _get_plugin_ids(plugin_type, plugin_info_dict["changed_plugins"][plugin_type], domain_root)
			plugin_info_dict[f'new_{plugin_type}'] = ' '.join(scoring_plugin_ids)
	else:
		plugin_info_dict["run_score"] = "False"


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

	run_args('brainscore_language', tests_to_run)
