"""
Imports for domain-instances' generic tests.
By importing this file's `pytest_addoption` and `pytest_generate_tests`,
the respective `generic_plugin_tests.py` can specify tests with an `identifier` argument
such that the tests are automatically parametrized.
"""

from pathlib import Path

from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids


def pytest_addoption(parser):
    """ attaches optional cmd-line args to the pytest machinery """
    parser.addoption("--plugin_directory", type=str, help="the directory of the plugin to test")


def pytest_generate_tests(metafunc):  # function called for every test
    plugin_directory = metafunc.config.option.plugin_directory
    if 'identifier' in metafunc.fixturenames and plugin_directory is not None:
        print(f"Using plugin directory {plugin_directory}")
        plugin_directory = Path(plugin_directory)
        plugin_identifiers = get_plugin_ids(plugin_type=plugin_directory.parent.name,
                                            new_plugin_dirs=[plugin_directory.name],
                                            domain_root=str(plugin_directory.parent.parent))
        print(f"Using identifiers {plugin_identifiers}")
        metafunc.parametrize("identifier", plugin_identifiers)
