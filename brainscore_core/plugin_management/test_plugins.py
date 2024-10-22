import os
import re
from pathlib import Path
from typing import Dict, Union, List

import pytest_check as check
import yaml

from .environment_manager import EnvironmentManager

PLUGIN_TYPES = ['benchmarks', 'data', 'metrics', 'models']
RECOGNIZED_TEST_FILES = r'test.*\.py'
GENERIC_PLUGIN_TEST_FILENAME = "generic_plugin_tests.py"
MODEL_SUBSET = ['hmax', 'alexnet', 'CORnet-S', 'resnet-50-robust', 'voneresnet-50-non_stochastic', 
                'resnet18-local_aggregation', 'grcnn_robust_v1', 'custom_model_cv_18_dagger_408', 
                'ViT_L_32_imagenet1k', 'mobilenet_v2_1.4_224', 'pixels', 'cvt_cvt-w24-384-in22k_finetuned-in1k_4', 
                'effnetb1_cutmixpatch_augmix_robust32_avge4e7_manylayers_324x288']


class PluginTestRunner(EnvironmentManager):
    """Runs plugin tests (requires "test.*\.py" for each plugin)
    
    Usage examples (run `test_plugins.py` file in domain-specific brain-score library, e.g. `brainscore_language`):

    # Run all tests for futrell2018 benchmark:
    python brainscore_core/plugin_management/test_plugins.py brainscore_language/benchmarks/futrell2018/test.py

    # Run only tests with names matching specified pattern (test_exact):
    python brainscore_core/plugin_management/test_plugins.py brainscore_language/benchmarks/futrell2018/test.py --test=test_exact

    # Run all tests for all plugins:
    python brainscore_core/plugin_management/test_plugins.py
    """

    def __init__(self, plugin_directory: Path, test: Union[bool, str] = False):
        super(PluginTestRunner, self).__init__()

        self.plugin_directory = plugin_directory
        self.plugin_type = Path(self.plugin_directory).parent.name
        self.library_path = Path(self.plugin_directory).parents[2]
        self.plugin_name = self.plugin_type + '__' + Path(self.plugin_directory).name
        self.generic_plugin_test = self._resolve_generic_plugin_test()
        self.env_name = self.plugin_name
        self.test = test if test else False
        self.returncode = 0
        self.script_path = Path(__file__).parent / 'test_plugin.sh'
        assert self.script_path.is_file(), f"bash file {self.script_path} does not exist"

    def __call__(self) -> Dict:
        self.validate_plugin()
        self.run_tests()
        self.teardown()

    def validate_plugin(self):
        self._validate_test_files()
        self._validate_environment_yml()

    def _validate_test_files(self):
        """
        requires at least one file matching the RECOGNIZED_TEST_FILES pattern in plugin directory,
        e.g. test.py, test_data.py.
        """
        test_files = [test_file for test_file in self.plugin_directory.iterdir()
                      if re.match(RECOGNIZED_TEST_FILES, test_file.name)]
        assert len(test_files) > 0, f"No test files matching '{RECOGNIZED_TEST_FILES}' found"

    def _validate_environment_yml(self):
        # if environment.yml is present, ensure no dependency conflicts
        # checks that environment.yml does not include env name or unsupported python versions
        conda_yml_path = self.plugin_directory / 'environment.yml'
        if conda_yml_path.is_file():
            with open(conda_yml_path, "r") as f:
                env = yaml.dump(yaml.safe_load(f))
                # ensure that name is not set so as to not override our assigned env name
                assert 'name' not in env, f"\nenvironment.yml must not specify 'name'"
                python_specs = [line for line in env.split("\n") if 'python=' in line]
                if len(python_specs) == 1:
                    python_spec = python_specs[0]
                    python_version = python_spec.split('python=')[1]
                    assert python_version.startswith('3.11')
                elif len(python_specs) > 1:
                    raise yaml.YAMLError('multiple versions of python found in environment.yml')
                # (else) no python specifications, ignore

    def run_tests(self):
        """ 
        Calls bash script to create conda environment, then
        runs all tests or selected test for specified plugin.
        If generic tests for the plugin type are defined by the domain library, those are run first.
        """

        run_command = f"bash {self.script_path} \
            {self.plugin_directory} {self.plugin_name} {self.test} {self.library_path} {self.generic_plugin_test}"

        completed_process = self.run_in_env(run_command)
        check.equal(completed_process.returncode, 0)  # use check to register any errors, but let tests continue

        self.returncode = completed_process.returncode

    def _resolve_generic_plugin_test(self) -> Union[bool, Path]:
        # remove plural and determine variable name, e.g. "models" -> "model"
        singular_prefix = self.plugin_type.strip('s')
        plugin_type_helper = f"{singular_prefix}_helpers"
        domain_root = Path(self.plugin_directory).parents[1]
        generic_plugin_test = domain_root / plugin_type_helper / GENERIC_PLUGIN_TEST_FILENAME
        if not generic_plugin_test.is_file():
            return False
        return generic_plugin_test


def run_specified_tests(root_directory: Path, test_file: str, test: str) -> Dict:
    """ Runs either a single test or all tests in the specified test file """

    results = {}
    plugin_type, plugin_dirname, filename = test_file.split('/')[-3:]
    plugin = root_directory / plugin_type / plugin_dirname
    assert re.match(RECOGNIZED_TEST_FILES, filename), \
        f"Test file {filename} not recognized as test file, must match '{RECOGNIZED_TEST_FILES}'."
    assert plugin_type in PLUGIN_TYPES, "Filepath not recognized as plugin test file."
    plugin_test_runner = PluginTestRunner(plugin, test=test)
    plugin_test_runner()
    results[plugin_test_runner.plugin_name] = plugin_test_runner.returncode

    return results


def run_all_tests(root_directory: Path) -> Dict:
    """ Runs tests for all plugins """
    results = {}
    for plugin_type in PLUGIN_TYPES:
        plugins_dir = root_directory / plugin_type
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                if str(plugins_dir) == 'brainscore_vision/models' and plugin.name not in MODEL_SUBSET:  # run subset of models to decrease test time
                    continue
                plugin_test_runner = PluginTestRunner(plugin)
                plugin_test_runner()
                results[plugin_test_runner.plugin_name] = plugin_test_runner.returncode
    return results


def run_args(root_directory: Union[Path, str], test_files: Union[None, List[str]] = None,
             test: Union[None, str] = None):
    """
    Run single specified test or all tests for each plugin.

    :param root_directory: the directory containing all plugin types, e.g. `/local/brain-score_language/`
    :param test_files: List of paths of target test files. If this is `None`, run all tests
    :param test: name of test to run (optional)
    """
    results = {}
    if not test_files:
        results = run_all_tests(root_directory=Path(root_directory))
    else:
        for test_file in test_files:
            assert Path(test_file).exists()
            results = run_specified_tests(root_directory=Path(root_directory), test_file=test_file, test=test)
    plugins_with_errors = {k: v for k, v in results.items() if (v != 0) and (v != 5)}
    num_plugins_failed = len(plugins_with_errors)
    assert num_plugins_failed == 0, f"\n{num_plugins_failed} plugin tests failed\n{plugins_with_errors}"
