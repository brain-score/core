import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from brainscore_core.plugin_management.import_plugin import import_plugin

dummy_container_dirpath = Path(tempfile.mkdtemp("brainscore-dummy"))
current_dependencies_pref = os.getenv('BS_INSTALL_DEPENDENCIES')


class TestImportPlugin:
    def setup_method(self):
        sys.path.append(str(dummy_container_dirpath))
        local_resource = Path(__file__).parent / 'test_import_plugin__brainscore_dummy'  # dummy-library scripts
        shutil.copytree(local_resource / 'brainscore_dummy', dummy_container_dirpath / 'brainscore_dummy')

    def teardown_method(self):
        model_registry = self._model_registry()
        if 'dummy-model' in model_registry:
            del model_registry['dummy-model']
        subprocess.run('pip uninstall pyaztro', shell=True)
        shutil.rmtree(dummy_container_dirpath)
        if current_dependencies_pref:  # value was set
            os.environ['BS_INSTALL_DEPENDENCIES'] = current_dependencies_pref
        sys.path.remove(str(dummy_container_dirpath))

    def _model_registry(self):
        # this basically runs `from brainscore_dummy import model_registry`
        # but because `brainscore_dummy` is dynamically generated in the setup method,
        # we do a string import so that the linter does not complain
        brainscore_dummy = __import__('brainscore_dummy')
        return brainscore_dummy.model_registry

    def test_yes_dependency_installation(self):
        os.environ['BS_INSTALL_DEPENDENCIES'] = 'yes'
        model_registry = self._model_registry()
        assert 'dummy-model' not in model_registry
        import_plugin('brainscore_dummy', 'models', 'dummy-model')
        assert 'dummy-model' in model_registry

    def test_no_dependency_installation(self):
        os.environ['BS_INSTALL_DEPENDENCIES'] = 'no'
        model_registry = self._model_registry()
        assert 'dummy-model' not in model_registry
        try:
            print("importing plugin")
            import_plugin('brainscore_dummy', 'models', 'dummy-model')
        except Exception as e:
            assert "No module named 'pyaztro'" in str(e)
