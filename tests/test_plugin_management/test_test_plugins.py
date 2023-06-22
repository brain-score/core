import pytest
import shutil
import subprocess
import tempfile
from pathlib import Path

from brainscore_core.plugin_management.test_plugins import PluginTestRunner


class TestPluginTestRunner:
    def setup_method(self):
        self.library_path = Path(tempfile.mkdtemp("plugin-library"))
        local_resource = Path(__file__).parent / 'test_test_plugins__brainscore_dummy'
        # `shutil.copytree(..., dirs_exist_ok=True)` would be preferable here but is not available in python 3.7
        subprocess.run(f"cp -r {local_resource}/* {self.library_path}", shell=True, text=True, check=True)

    def teardown_method(self):
        shutil.rmtree(self.library_path)

    def test_plugin_name(self):
        dummy_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'dummy_plugin'
        plugin_test_runner = PluginTestRunner(dummy_plugin_path, {})
        assert plugin_test_runner.plugin_name == 'plugintype__dummy_plugin'

    def test_has_no_testfile(self):
        dummy_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'dummy_plugin'
        test_file = dummy_plugin_path / 'test.py'
        test_file.unlink()
        plugin_test_runner = PluginTestRunner(dummy_plugin_path, {})
        with pytest.raises(Exception):
            plugin_test_runner.validate_plugin()

    def test_has_testfile_underscore_prefix(self):
        dummy_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'dummy_plugin'
        test_file = dummy_plugin_path / 'test.py'
        test_file.rename(dummy_plugin_path / 'test_plugin.py')
        plugin_test_runner = PluginTestRunner(dummy_plugin_path, {})
        plugin_test_runner.validate_plugin()

    def test_has_testfile_no_underscore_prefix(self):
        dummy_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'dummy_plugin'
        test_file = dummy_plugin_path / 'test.py'
        test_file.rename(dummy_plugin_path / 'testplugin.py')
        plugin_test_runner = PluginTestRunner(dummy_plugin_path, {})
        plugin_test_runner.validate_plugin()

    def test_has_testfile_valid_python(self):
        dummy_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'dummy_plugin'
        test_file = dummy_plugin_path / 'test.py'
        test_file.rename(dummy_plugin_path / 'testpluginpy')
        plugin_test_runner = PluginTestRunner(dummy_plugin_path, {})
        with pytest.raises(Exception):
            plugin_test_runner.validate_plugin()

    def test_run_tests(self):
        dummy_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'dummy_plugin'
        plugin_test_runner = PluginTestRunner(dummy_plugin_path, {})
        plugin_test_runner.run_tests()
        assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0

    @pytest.mark.travis_slow
    def test_run_tests_with_r(self):
        r_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'r_plugin'
        plugin_test_runner = PluginTestRunner(r_plugin_path, {})
        plugin_test_runner.run_tests()
        assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0
