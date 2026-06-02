"""Tests for the v1.5 ``Subject`` rename (formerly ``UnifiedModel``).

v1.5 renames the ABC ``UnifiedModel`` to ``Subject`` to make the interface
subject-agnostic, keeping ``UnifiedModel`` as a deprecated alias so existing
imports keep working bit-for-bit.
"""
from abc import ABC

import pytest

import brainscore_core
from brainscore_core.model_interface import Subject, UnifiedModel, BrainScoreModel


class _ConcreteSubject(Subject):
    @property
    def identifier(self):
        return "test"

    @property
    def region_layer_map(self):
        return {}

    @property
    def supported_modalities(self):
        return {"vision"}

    def process(self, stimuli):
        return None


class TestSubjectRename:
    def test_subject_is_an_abc(self):
        assert issubclass(Subject, ABC)
        with pytest.raises(TypeError):
            Subject()  # abstract, cannot instantiate

    def test_unifiedmodel_is_alias_of_subject(self):
        assert UnifiedModel is Subject

    def test_subject_exported_from_package(self):
        assert brainscore_core.Subject is Subject
        assert brainscore_core.UnifiedModel is Subject

    def test_brainscoremodel_subclasses_subject(self):
        assert issubclass(BrainScoreModel, Subject)
        assert issubclass(BrainScoreModel, UnifiedModel)  # via alias

    def test_legacy_subclassing_via_alias_still_works(self):
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
        assert isinstance(m, UnifiedModel)  # alias identity
        assert m.supported_modalities == {"vision"}
