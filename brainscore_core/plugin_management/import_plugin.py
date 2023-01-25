import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ImportPlugin:
    """ import plugin and (optionally) install dependencies """

    def __init__(self, library_root: str, plugin_type: str, identifier: str):
        self.plugin_type = plugin_type
        library_module = __import__(library_root)
        library_directory = Path(library_module.__file__).parent
        self.plugins_dir = library_directory / plugin_type
        assert self.plugins_dir.is_dir(), f"Plugins directory {self.plugins_dir} is not a directory"
        self.identifier = identifier
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
            with open(init_file) as f:
                registry_name = self.plugin_type.strip(
                    's') + '_registry'  # remove plural and determine variable name, e.g. "models" -> "model_registry"
                plugin_registrations = [line for line in f if f"{registry_name}['{self.identifier}']"
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
        This is done via `pip install` in the current interpreter.
        """
        requirements_file = self.plugins_dir / self.plugin_dirname / 'requirements.txt'
        if requirements_file.is_file():
            subprocess.run(f"pip install -r {requirements_file}", shell=True)
        else:
            logger.debug(f"Plugin {self.plugin_dirname} has no requirements file {requirements_file}")


def installation_preference():
    pref_options = ['yes', 'no', 'newenv']
    pref = os.getenv('BS_INSTALL_DEPENDENCIES', 'yes')
    assert pref in pref_options, f"BS_INSTALL_DEPENDENCIES value {pref} not recognized. Must be one of {pref_options}."
    return pref


def import_plugin(library_root: str, plugin_type: str, identifier: str):
    """ 
    Installs the dependencies of the given plugin and imports its base package: 
    Given the identifier `Futrell2018-pearsonr` from library_root `brainscore_language`,
    :meth:`~brainscore_core.plugin_management.ImportPlugin.locate_plugin` sets
    :attr:`~brainscore_core.plugin_management.ImportPlugin.plugin_dirname` directory of plugin
    denoted by the `identifier`, then
    :meth:`~brainscore_core.plugin_management.ImportPlugin.install_requirements` installs all requirements
        in that directory's requirements.txt, and the plugin base package is imported
    """
    importer = ImportPlugin(library_root, plugin_type, identifier)

    if installation_preference() != 'no':
        importer.install_requirements()

    __import__(f'{library_root}.{plugin_type}.{importer.plugin_dirname}')
