"""Check dependencies that combined domain installs can accidentally supply."""

from importlib.metadata import requires

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


@pytest.mark.parametrize('dependency', ['result-caching', 'psutil'])
def test_scoring_dependency_is_required_without_domain_packages(dependency):
    requirements = [Requirement(value) for value in requires('brainscore_core')]
    assert any(
        canonicalize_name(requirement.name) == dependency
        and (requirement.marker is None or requirement.marker.evaluate({'extra': ''}))
        for requirement in requirements
    ), f'Standalone Core scoring must install {dependency} without Vision or extras'
