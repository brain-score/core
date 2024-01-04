# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.test_generic_helpers import pytest_addoption, pytest_generate_tests


def test_model(identifier: str):
    assert identifier is not None
    print(f"Testing model {identifier}")
