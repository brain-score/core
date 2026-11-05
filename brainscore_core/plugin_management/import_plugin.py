import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class PluginNotFoundError(AssertionError, KeyError):
    """No plugin registration matches; construction errors are separate."""


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

        if plugin_registrations_count == 0:
            raise PluginNotFoundError(f"No registrations found for {self.identifier}")
        assert plugin_registrations_count == 1, f"More than one registration found for {self.identifier}"

        return specified_plugin_dirname

    def install_requirements(self):
        """
        Install the requirements of the given plugin directory into the current interpreter.

        - A `setup.py` installs the directory; a `requirements.txt` is installed with `-r`.
        - Packages the environment is pinned to (see `PROTECTED_PACKAGES`) are held at their
          installed versions. A plugin that needs other versions raises `PluginDependencyError`
          instead of changing them, since that would move every other model's scores.
        - Lines naming Python's own modules (e.g. `imp`, `collections`) are skipped: they are
          not packages, and pip would fail on them.
        """
        plugin_dir = self.plugins_dir / self.plugin_dirname
        setup_file, requirements_file = plugin_dir / 'setup.py', plugin_dir / 'requirements.txt'
        if not setup_file.is_file() and not requirements_file.is_file():
            logger.debug(f"Plugin {self.plugin_dirname} has no requirements.txt or setup.py")
            return
        with tempfile.TemporaryDirectory(prefix='bs-plugin-') as scratch:
            scratch = Path(scratch)
            args = [str(plugin_dir)] if setup_file.is_file() else []
            if requirements_file.is_file():
                args += ['-r', str(_without_stdlib_modules(requirements_file, scratch / 'requirements.txt'))]
            # In a dedicated environment ('newenv'), the plugin may set any versions it needs.
            if installation_preference() == 'yes':
                args = ['-c', str(_hold_installed_versions(scratch / 'constraints.txt')), *args]
            completed = subprocess.run([sys.executable, '-m', 'pip', 'install', *args],
                                       capture_output=True, text=True)
        if completed.returncode != 0:
            raise PluginDependencyError(_explain_install_failure(
                self.identifier, completed.stdout + completed.stderr, plugin_dir))


# Pinned by the shared environment; a plugin may add packages but not move these.
PROTECTED_PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xarray', 'pyarrow', 'netCDF4',
                      'torch', 'torchvision', 'transformers', 'datasets', 'importlib-metadata')


class PluginDependencyError(RuntimeError):
    """A plugin's requirements cannot be installed into this environment."""


def _hold_installed_versions(path: Path) -> Path:
    from importlib.metadata import PackageNotFoundError, version
    lines = []
    for name in PROTECTED_PACKAGES:
        try:
            lines.append(f'{name}=={version(name)}')
        except PackageNotFoundError:
            continue
    path.write_text('\n'.join(lines) + '\n')
    return path


def _without_stdlib_modules(requirements: Path, path: Path) -> Path:
    kept = []
    for line in requirements.read_text().splitlines():
        name = re.match(r'\s*([A-Za-z0-9_.-]+)\s*$', line.split('#')[0])
        if name and name.group(1) in sys.stdlib_module_names:
            logger.warning(f"Skipping '{name.group(1)}' in {requirements}: it is part of Python, not a package.")
            continue
        kept.append(line)
    path.write_text('\n'.join(kept) + '\n')
    return path


def _explain_install_failure(identifier: str, output: str, plugin_dir: Path = None) -> str:
    """A readable reason, the pip lines behind it, and what to try next."""
    stderr = output
    lines = [line for line in output.splitlines() if line.strip()]
    requirements = (plugin_dir / 'requirements.txt') if plugin_dir is not None else Path('<plugin>/requirements.txt')
    runner = ("Next step: give it an environment of its own, so this one keeps its versions:\n"
              "    conda create --name bs-model --clone \"$CONDA_DEFAULT_ENV\" && conda activate bs-model\n"
              f"    python -m pip install -r \"{requirements}\"\n"
              f"then load or score '{identifier}' there.")
    linux = ("Next step: load it on Linux (for example an EC2 instance), where more packages have "
             "ready-made builds. If it fails there too, please tell a maintainer: "
             "https://github.com/brain-score/vision/issues")
    missing = [line for line in lines if 'No matching distribution found for' in line]
    if 'ResolutionImpossible' in stderr or 'conflict' in stderr.lower():
        detail = []
        for line in lines:
            line = line.strip()
            if line.startswith('The user requested (constraint) '):
                detail.append('this environment holds ' + line.split(') ', 1)[1])
            elif line.startswith('The user requested '):
                detail.append('the plugin asks for ' + line[len('The user requested '):])
            elif ' depends on ' in line:
                detail.append(line)
        detail = detail[:6]
        reason = (f"'{identifier}' needs package versions that conflict with this environment. "
                  "Installing them would change the versions scores are verified against, "
                  "so nothing was installed.")
    elif missing:
        detail = [line.strip() for line in missing][:3]
        reason = (f"'{identifier}' needs a package with no release for this platform or Python version. "
                  f"It may install on Linux.")
        runner = linux
    elif 'Failed to build' in stderr:
        detail = [line.strip() for line in lines if 'Failed to build' in line][:3]
        reason = f"'{identifier}' needs a package that failed to build from source here."
        runner = linux
    else:
        detail = lines[-5:]
        reason = f"Installing the requirements of '{identifier}' failed."
    return '\n'.join([reason, *('  ' + line for line in detail), runner])


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
