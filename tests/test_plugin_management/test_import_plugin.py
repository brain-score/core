import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from brainscore_core.plugin_management.import_plugin import import_plugin

dummy_container_dirpath = Path(tempfile.mkdtemp("brainscore-dummy"))
dummy_domain_dirpath = dummy_container_dirpath / "brainscore_dummy"
dummy_model_dirpath = dummy_domain_dirpath / "models" / "dummy_model"
current_dependencies_pref = os.getenv('BS_INSTALL_DEPENDENCIES')


class TestImportPlugin:
    def setup_method(self):
        dummy_registry = dummy_domain_dirpath / "__init__.py"
        dummy_model = dummy_model_dirpath / "model.py"
        dummy_testfile = dummy_model_dirpath / "test.py"
        dummy_requirements = dummy_model_dirpath / "test.py"
        dummy_init = dummy_model_dirpath / "__init__.py"

        dummy_model_dirpath.mkdir(parents=True, exist_ok=True)
        sys.path.append(str(dummy_container_dirpath))

        Path(dummy_registry).touch()
        with open(dummy_registry, 'w') as f:
            f.write(textwrap.dedent('''\
            model_registry = {}
            '''))

        Path(dummy_model).touch()
        with open(dummy_model, 'w') as f:
            f.write(textwrap.dedent('''\
            class dummyModel:
                pass       
            '''))
        Path(dummy_testfile).touch()
        with open(dummy_testfile, 'w') as f:
            f.write(textwrap.dedent('''\
            def test_dummy():
                assert True        
            '''))
        Path(dummy_requirements).touch()
        with open(dummy_requirements, 'w') as f:
            f.write(textwrap.dedent('''\
            pyaztro     
            '''))
        Path(dummy_init).touch()
        with open(dummy_init, 'w') as f:
            f.write(textwrap.dedent('''\
            from brainscore_dummy import model_registry
            from .model import dummyModel

            model_registry['dummy-model'] = dummyModel     
            '''))

    def teardown_method(self):
        model_registry = self._model_registry()
        if 'dummy-model' in model_registry:
            del model_registry['dummy-model']
        subprocess.run('pip uninstall pyaztro', shell=True)
        shutil.rmtree(dummy_domain_dirpath)
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
