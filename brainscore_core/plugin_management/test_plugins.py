import warnings
from pathlib import Path
from typing import Dict, Union

import pytest_check as check
from .environment_manager import EnvironmentManager

PLUGIN_TYPES = ['benchmarks', 'data', 'metrics', 'models']


class PluginTestRunner(EnvironmentManager):
    """Runs plugin tests (requires "test.py" for each plugin)
    
    Usage examples (run `test_plugins.py` file in domain-specific brain-score library, e.g. `brainscore_language`):

    # Run all tests for futrell2018 benchmark:
    python brainscore_language/plugin_management/test_plugins.py brainscore_language/benchmarks/futrell2018/test.py

    # Run only tests with names matching specified pattern (test_exact):
    python brainscore_language/plugin_management/test_plugins.py brainscore_language/benchmarks/futrell2018/test.py --test=test_exact

    # Run all tests for all plugins:
    python brainscore_language/plugin_management/test_plugins.py 
    """

    def __init__(self, plugin_directory: Path, results: Dict, test: Union[bool, str] = False):
        super(PluginTestRunner, self).__init__()

        self.plugin_directory = plugin_directory
        self.plugin_type = Path(self.plugin_directory).parent.name
        self.plugin_name = self.plugin_type + '_' + Path(self.plugin_directory).name
        self.env_name = self.plugin_name
        self.has_requirements = (self.plugin_directory / 'requirements.txt').is_file()
        self.test = test if test else False
        self.results = results
        self.script_path = Path(__file__).parent / 'test_plugin.sh'
        assert self.script_path.is_file(), f"bash file {self.script_path} does not exist"

    def __call__(self):
        self.validate_plugin()
        self.run_tests()
        self.teardown()

    def validate_plugin(self):
        """ requires "test.py" file in plugin directory """
        assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

    def run_tests(self):
        """ 
        calls bash script to create conda environment, then
        runs all tests or selected test for specified plugin
        """
        run_command = f"bash {self.script_path} \
            {self.plugin_directory} {self.plugin_name} \
            {str(self.has_requirements).lower()} {self.test}"

        completed_process = self.run_in_env(run_command)
        check.equal(completed_process.returncode, 0)  # use check to register any errors, but let tests continue

        self.results[self.plugin_name] = completed_process.returncode


def run_specified_tests(root_directory: Path, test_file: str, results: Dict, test: str):
    """ Runs either a single test or all tests in a specified test.py """
    plugin_type, plugin_dirname, filename = test_file.split('/')[-3:]
    plugin = root_directory / plugin_type / plugin_dirname
    assert filename == "test.py", "Filepath not recognized as test file, must be 'test.py'."
    assert plugin_type in PLUGIN_TYPES, "Filepath not recognized as plugin test file."
    plugin_test_runner = PluginTestRunner(plugin, results, test=test)
    plugin_test_runner()


def run_all_tests(root_directory: Path, results: Dict):
    """ Runs tests for all plugins """
    for plugin_type in PLUGIN_TYPES:
        plugins_dir = root_directory / plugin_type
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                plugin_test_runner = PluginTestRunner(plugin, results)
                plugin_test_runner()


def run_args(root_directory: Union[Path, str], test_file: Union[None, str] = None, test: Union[None, str] = None):
    """
    Run single specified test or all tests for each plugin.

    :param root_directory: the directory containing all plugin types, e.g. `/local/brain-score_language/`
    :param test_file: path of target test file (optional)
    :param test: name of test to run (optional)
    """
    results = {}
    if not test_file:
        run_all_tests(root_directory=Path(root_directory), results=results)
    elif test_file and Path(test_file).exists():
        run_specified_tests(root_directory=Path(root_directory), test_file=test_file, results=results, test=test)
    else:
        warnings.warn("Test file not found.")

    plugins_with_errors = {k: v for k, v in results.items() if v == 1}
    num_plugins_failed = len(plugins_with_errors)
    assert num_plugins_failed == 0, f"\n{num_plugins_failed} plugin tests failed\n{plugins_with_errors}"
