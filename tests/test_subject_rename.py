"""The minimal subject and the legacy compatibility base remain interoperable."""
from abc import ABC

import pytest

import brainscore_core
from brainscore_core.model_interface import Subject, UnifiedModel, BrainScoreModel


class _ConcreteSubject(Subject):
    identifier = "test"
    in_channels = {"vision"}
    out_channels = {"behavior"}

    def interact(self, session):
        pass


class TestSubjectRename:
    def test_subject_is_an_abc(self):
        assert issubclass(Subject, ABC)
        with pytest.raises(TypeError):
            Subject()  # abstract, cannot instantiate

    def test_unifiedmodel_is_compatibility_subclass(self):
        assert issubclass(UnifiedModel, Subject)
        assert UnifiedModel is not Subject

    def test_subject_exported_from_package(self):
        assert brainscore_core.Subject is Subject
        assert brainscore_core.UnifiedModel is UnifiedModel

    def test_brainscoremodel_subclasses_subject(self):
        assert issubclass(BrainScoreModel, Subject)
        assert issubclass(BrainScoreModel, UnifiedModel)  # via the compatibility base

    def test_legacy_subclassing_still_works(self):
        # Code written against the old name keeps working unchanged.
        class LegacyModel(UnifiedModel):
            @property
            def identifier(self):
                return "legacy"

            @property
            def region_layer_map(self):
                return {}

            @property
            def supported_modalities(self):
                return {"text"}

            def process(self, stimuli):
                return "ok"

        m = LegacyModel()
        assert m.identifier == "legacy"
        assert m.process(None) == "ok"
        assert isinstance(m, Subject)

    def test_new_subclassing_via_subject(self):
        m = _ConcreteSubject()
        assert isinstance(m, Subject)
        assert not isinstance(m, UnifiedModel)
        assert m.in_channels == {"vision"}
