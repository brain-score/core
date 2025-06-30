import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ImportPlugin:
    """ import plugin and (optionally) install dependencies """

    def __init__(self, library_root: str, plugin_type: str, identifier: str, registry_prefix: str = None):
        """
        :param library_root: the root package of the library we're loading plugins from, e.g. `brainscore_vision`
        :param plugin_type: the directory containing the plugin directories, e.g. `benchmarks`.
            If `registry_prefix` is not explicitly passed, it will be inferred from this parameter.
        :param identifier: the unique identifier for this plugin, e.g. `MajajHong2015` or `alexnet`.
            Note that this is (potentially) different from the plugin directory itself.
        :param registry_prefix: the prefix or the registry in case it is different form the `plugin_type` directory,
            e.g. `stimulus_set`
        """
        self.plugin_type = plugin_type
        library_module = __import__(library_root)
        library_directory = Path(library_module.__file__).parent
        self.plugins_dir = library_directory / plugin_type
        assert self.plugins_dir.is_dir(), f"Plugins directory {self.plugins_dir} is not a directory"
        self.identifier = identifier
        # if registry_prefix not explicitly set, infer from plugin_type:
        # remove plural and determine variable name, e.g. "models" -> "model_registry"
        registry_prefix = registry_prefix or self.plugin_type.strip('s')
        self.registry_name = registry_prefix + '_registry'
        self.plugin_dirname = self.locate_plugin()

    def locate_plugin(self) -> str:
        """ 
        Searches all `plugin_type` __init.py__ files for the plugin denoted with `identifier`.
        If a match is found of format {plugin_type}_registry[{identifier}],
        returns name of directory where __init.py__ is located
        """
        plugins = [d.name for d in self.plugins_dir.iterdir() if d.is_dir()]

        specified_plugin_dirname = None
        plugin_registrations_count = 0
        for plugin_dirname in plugins:
            if plugin_dirname.startswith('.') or plugin_dirname.startswith('_'):  # ignore e.g. __pycache__
                continue
            plugin_dirpath = self.plugins_dir / plugin_dirname
            init_file = plugin_dirpath / "__init__.py"
            if not init_file.is_file():
                logger.warning(f"No __init__.py in {plugin_dirpath}")
                continue
            with open(init_file, encoding='utf-8') as f:
                plugin_registrations = [line for line in f if f"{self.registry_name}['{self.identifier}']"
                                        in line.replace('\"', '\'')]
                if len(plugin_registrations) > 0:
                    specified_plugin_dirname = plugin_dirname
                    plugin_registrations_count += 1

        assert plugin_registrations_count > 0, f"No registrations found for {self.identifier}"
        assert plugin_registrations_count == 1, f"More than one registration found for {self.identifier}"

        return specified_plugin_dirname

    def install_requirements(self):
        """
        Install all the requirements of the given plugin directory.
        - If a `setup.py` file exists, it is run in the current interpreter.
        - Alternatively, if a `requirements.txt` file exists, this is done via
          `pip install` in the current interpreter.
        - If both `setup.py` and `requirements.txt` are present, `setup.py` is installed first.
        """
        setup_file = self.plugins_dir / self.plugin_dirname / 'setup.py'
        requirements_file = self.plugins_dir / self.plugin_dirname / 'requirements.txt'
        
        if not setup_file.is_file() and not requirements_file.is_file():
            logger.debug(
                f"Plugin {self.plugin_dirname} has no requirements file {requirements_file} "
                f"or setup file {setup_file}"
            )
        
        if setup_file.is_file():
            subprocess.run(
                f"pip install {self.plugins_dir / self.plugin_dirname}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

        if requirements_file.is_file():
            subprocess.run(
                f"pip install -r {requirements_file}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )


def installation_preference():
    pref_options = ['yes', 'no', 'newenv']
    pref = os.getenv('BS_INSTALL_DEPENDENCIES', 'yes')
    assert pref in pref_options, f"BS_INSTALL_DEPENDENCIES value {pref} not recognized. Must be one of {pref_options}."
    return pref


def import_plugin(library_root: str, plugin_type: str, identifier: str, registry_prefix: str = None):
    """ 
    Installs the dependencies of the given plugin and imports its base package: 
    Given the identifier `Futrell2018-pearsonr` from library_root `brainscore_language`,
    :meth:`~brainscore_core.plugin_management.ImportPlugin.locate_plugin` sets
    :attr:`~brainscore_core.plugin_management.ImportPlugin.plugin_dirname` directory of plugin
    denoted by the `identifier`, then
    :meth:`~brainscore_core.plugin_management.ImportPlugin.install_requirements` installs all requirements
        in that directory's requirements.txt, and the plugin base package is imported
    """
    importer = ImportPlugin(library_root=library_root, plugin_type=plugin_type, identifier=identifier,
                            registry_prefix=registry_prefix)

    if installation_preference() != 'no':
        importer.install_requirements()

    __import__(f'{library_root}.{plugin_type}.{importer.plugin_dirname}')


def print_plugin_dir(library_root: str, plugin_type: str, identifier: str):
    importer = ImportPlugin(library_root=library_root, plugin_type=plugin_type, identifier=identifier)
    print(importer.locate_plugin())


if __name__ == '__main__':
    import fire

    fire.Fire()
