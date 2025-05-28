import pytest
import shutil
import subprocess
import tempfile
from pathlib import Path
import ctypes
import platform

from brainscore_core.plugin_management.test_plugins import PluginTestRunner


def get_glibc_version():
    """
    Get GLIBC version on Linux systems, return None on other systems.
    Handles different return types from gnu_get_libc_version():
    - Some systems return bytes (needs decode)
    - Some systems return integer pointer (needs c_char_p conversion)
    - Some systems return string directly
    """
    if platform.system() != 'Linux':
        return None
        
    try:
        libc = ctypes.CDLL('libc.so.6')
        version = libc.gnu_get_libc_version()
        
        # Handle different return types
        if isinstance(version, bytes):
            version_str = version.decode('ascii')
        elif isinstance(version, int):
            # Convert integer pointer to string
            version_str = ctypes.c_char_p(version).value.decode('ascii')
        elif isinstance(version, str):
            version_str = version
        else:
            raise TypeError(f"Unexpected return type from gnu_get_libc_version: {type(version)}")
            
        # Split and rejoin to normalize version format
        return '.'.join(str(x) for x in version_str.split('.'))
    except (OSError, AttributeError, TypeError) as e:
        print(f"Error getting GLIBC version: {e}")
        return None


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
        assert plugin_test_runner.returncode == 0

    @pytest.mark.travis_slow
    def test_run_tests_with_r(self):
        # Check if we're on Linux and have a compatible GLIBC version
        glibc_version = get_glibc_version()
        if glibc_version is not None:  # We're on Linux
            if float(glibc_version) < 2.34:  # rpy2 3.6.0 requires GLIBC 2.34 or higher
                pytest.skip(f"Test requires GLIBC 2.34 or higher, but found {glibc_version}")
        else:  # Not on Linux
            pytest.skip("This test is only supported on Linux systems with GLIBC 2.34 or higher")

        r_plugin_path = self.library_path / 'brainscore_dummy' / 'plugintype' / 'r_plugin'
        plugin_test_runner = PluginTestRunner(r_plugin_path, {})
        plugin_test_runner.run_tests()
        assert plugin_test_runner.returncode == 0
