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
        for model_id in list(model_registry.keys()):
            del model_registry[model_id]
        subprocess.run(f'pip uninstall pyaztro pyfiglet --yes', shell=True)
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

    def test_dependency_install_from_setup_py(self):
        os.environ['BS_INSTALL_DEPENDENCIES'] = 'yes'
        model_registry = self._model_registry()
        assert 'dummy-model-setup-py' not in model_registry
        import_plugin('brainscore_dummy', 'models', 'dummy-model-setup-py')
        assert 'dummy-model-setup-py' in model_registry

    def test_dependency_install_from_setup_py_with_requirements_txt(self):
        os.environ['BS_INSTALL_DEPENDENCIES'] = 'yes'
        model_registry = self._model_registry()
        assert 'dummy-model-requirements-and-setup' not in model_registry
        import_plugin('brainscore_dummy', 'models', 'dummy-model-requirements-and-setup')
        assert 'dummy-model-requirements-and-setup' in model_registry


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


class TestInstallSafeguards:
    def test_python_modules_are_skipped(self, tmp_path):
        from brainscore_core.plugin_management.import_plugin import _without_stdlib_modules
        requirements = tmp_path / 'requirements.txt'
        requirements.write_text('torch\nimp\ncollections  # stdlib\ntimm>=1.0\n')
        kept = _without_stdlib_modules(requirements, tmp_path / 'out.txt').read_text().split()
        assert kept == ['torch', 'timm>=1.0']

    def test_installed_pins_are_held(self, tmp_path):
        import numpy
        from brainscore_core.plugin_management.import_plugin import _hold_installed_versions
        assert f'numpy=={numpy.__version__}' in _hold_installed_versions(tmp_path / 'c.txt').read_text()

    def test_conflict_is_explained_with_a_next_step(self):
        from brainscore_core.plugin_management.import_plugin import _explain_install_failure
        stderr = ('ERROR: Cannot install transformers==4.30.2 because these package versions have conflicting dependencies.\n'
                  'The conflict is caused by:\n    The user requested transformers==4.30.2\n'
                  '    The user requested (constraint) transformers==4.57.6\nERROR: ResolutionImpossible\n')
        message = _explain_install_failure('cvt_13', stderr)
        assert 'conflict with this environment' in message
        assert 'the plugin asks for transformers==4.30.2' in message and 'this environment holds transformers==4.57.6' in message
        assert 'conda create --name bs-model --clone' in message

    def test_missing_platform_package_is_explained(self):
        from brainscore_core.plugin_management.import_plugin import _explain_install_failure
        message = _explain_install_failure('openstl', 'ERROR: No matching distribution found for decord\n')
        assert 'no release for this platform' in message and 'decord' in message
        assert 'Linux' in message
