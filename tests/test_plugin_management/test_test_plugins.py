import shutil
import tempfile
from pathlib import Path

import pytest

from brainscore_core.plugin_management.test_plugins import PluginTestRunner

DUMMY_LIBRARY_PATH = Path(tempfile.mkdtemp("plugin-library"))
DUMMY_PLUGIN_PATH = DUMMY_LIBRARY_PATH / 'brainscore_dummy' / 'plugintype' / 'pluginname'


@pytest.mark.requires_conda
class TestPluginTestRunner:
    def setup_method(self):
        local_resource = Path(__file__).parent / 'test_test_plugins__brainscore_dummy'
        shutil.copytree(local_resource, DUMMY_LIBRARY_PATH, dirs_exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(DUMMY_LIBRARY_PATH)

    def test_plugin_name(self):
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, {})
        assert plugin_test_runner.plugin_name == 'plugintype__pluginname'

    def test_has_testfile(self):
        test_file = DUMMY_PLUGIN_PATH / 'test.py'
        test_file.unlink()
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, {})
        with pytest.raises(Exception):
            plugin_test_runner.validate_plugin()

    def test_has_requirements(self):
        requirements_file = DUMMY_PLUGIN_PATH / 'requirements.txt'
        requirements_file.unlink()
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, {})
        assert not plugin_test_runner.has_requirements

    def test_run_tests(self):
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, {})
        plugin_test_runner.run_tests()
        assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0
