import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from brainscore_core.plugin_management.import_plugin import import_plugin, ImportPlugin


class TestImportPlugin:
    dummy_container_dirpath = Path(tempfile.mkdtemp("brainscore-dummy"))
    current_dependencies_pref = os.getenv('BS_INSTALL_DEPENDENCIES')

    def setup_method(self):
        sys.path.append(str(TestImportPlugin.dummy_container_dirpath))
        local_resource = Path(__file__).parent / 'test_import_plugin__brainscore_dummy'  # dummy-library scripts
        shutil.copytree(local_resource / 'brainscore_dummy',
                        TestImportPlugin.dummy_container_dirpath / 'brainscore_dummy')

    def teardown_method(self):
        model_registry = self._model_registry()
        if 'dummy-model' in model_registry:
            del model_registry['dummy-model']
        subprocess.run('pip uninstall pyaztro --yes', shell=True)
        shutil.rmtree(TestImportPlugin.dummy_container_dirpath)
        if TestImportPlugin.current_dependencies_pref:  # value was set
            os.environ['BS_INSTALL_DEPENDENCIES'] = TestImportPlugin.current_dependencies_pref
        sys.path.remove(str(TestImportPlugin.dummy_container_dirpath))

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


class TestRegistryPrefix:
    dummy_container_dirpath = Path(tempfile.mkdtemp("brainscore-dummy-registryprefix"))

    def setup_method(self):
        sys.path.append(str(TestRegistryPrefix.dummy_container_dirpath))
        local_resource = Path(__file__).parent / 'test_import_plugin__brainscore_dummy_registryprefix'
        shutil.copytree(local_resource / 'brainscore_dummy_registryprefix',
                        TestRegistryPrefix.dummy_container_dirpath / 'brainscore_dummy_registryprefix')

    def teardown_method(self):
        shutil.rmtree(TestRegistryPrefix.dummy_container_dirpath)
        sys.path.remove(str(TestRegistryPrefix.dummy_container_dirpath))

    def test_stimulus_set(self):
        importer = ImportPlugin(library_root='brainscore_dummy_registryprefix',
                                plugin_type='data', identifier='dummy-stimulus_set',
                                registry_prefix='stimulus_set')
        assert importer.plugin_dirname == 'dummy_stimulus_set'
