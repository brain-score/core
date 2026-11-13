"""The minimal subject and the legacy compatibility base remain interoperable."""
from abc import ABC

import pytest

import brainscore_core
from brainscore_core.contract import Subject, UnifiedModel


class TestSubjectRename:
    def test_subject_is_an_abc(self):
        assert issubclass(Subject, ABC)
        with pytest.raises(TypeError):
            Subject()  # abstract, cannot instantiate

    def test_unified_model_is_a_subject(self):
        assert issubclass(UnifiedModel, Subject)
        assert UnifiedModel is not Subject

    def test_subject_exported_from_package(self):
        assert brainscore_core.Subject is Subject
        assert brainscore_core.UnifiedModel is UnifiedModel
