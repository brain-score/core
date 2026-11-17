"""The subject contract is one abstract class; core has no typed base class."""
from abc import ABC

import pytest

import brainscore_core
from brainscore_core.contract import Subject


class TestSubjectRename:
    def test_subject_is_an_abc(self):
        assert issubclass(Subject, ABC)
        with pytest.raises(TypeError):
            Subject()  # abstract, cannot instantiate

    def test_subject_exported_from_package(self):
        assert brainscore_core.Subject is Subject
        assert not hasattr(brainscore_core, 'UnifiedModel')      # deleted; inherit Subject
