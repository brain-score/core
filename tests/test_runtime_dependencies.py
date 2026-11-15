"""Check dependencies that combined domain installs can accidentally supply."""

from importlib.metadata import requires
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

import brainscore_core


def _core_requirements():
    """The requirements of the brainscore_core that is imported.

    From a source checkout (on PYTHONPATH, or an editable install of another copy) that is
    the checkout's pyproject.toml; the installed metadata can belong to another copy. From
    an installed wheel (the GitHub jobs) it is the installed metadata.
    """
    pyproject = Path(brainscore_core.__file__).resolve().parents[1] / 'pyproject.toml'
    if pyproject.is_file():
        project = tomllib.loads(pyproject.read_text())['project']
        if canonicalize_name(project['name']) == canonicalize_name('brainscore_core'):
            return [Requirement(value) for value in project['dependencies']]
    return [Requirement(value) for value in requires('brainscore_core')]


@pytest.mark.parametrize('dependency', ['result-caching', 'psutil'])
def test_scoring_dependency_is_required_without_domain_packages(dependency):
    requirements = _core_requirements()
    assert any(
        canonicalize_name(requirement.name) == dependency
        and (requirement.marker is None or requirement.marker.evaluate({'extra': ''}))
        for requirement in requirements
    ), f'Standalone Core scoring must install {dependency} without Vision or extras'
