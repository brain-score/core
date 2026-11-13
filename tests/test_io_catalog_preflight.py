"""Tests for the v1.5 Input/Output Catalog wiring into pre-flight.

``check_compatibility`` rejects an input that is not in the catalog, and accepts
a documented input or neural region without a warning.
"""
import warnings

import pytest

from brainscore_core.contract import UnifiedModel
from brainscore_core.compatibility import (
    check_compatibility,
    CompatibilityError,
    CompatibilityWarning,
)


class _Model(UnifiedModel):
    def __init__(self, available, region_map=None):
        self._av = set(available)
        self._rm = region_map or {}

    @property
    def identifier(self):
        return "m"

    @property
    def region_layer_map(self):
        return self._rm

    @property
    def supported_modalities(self):
        return self._av

    def process(self, stimuli):
        return None


class _Bench:
    def __init__(self, required, region=None):
        self.identifier = "b"
        self.required_modalities = required
        self.region = region


class TestIOCatalogPreflight:
    def test_documented_input_no_warning(self):
        m = _Model({"vision"})
        b = _Bench({"vision"})
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompatibilityWarning)
            check_compatibility(m, b)  # vision is documented; must not warn

    def test_undocumented_input_is_rejected(self):
        # Contract v3 rule 3.2: every channel family is registered. An
        # undocumented input is a compatibility error, not a warning.
        m = _Model({"vision", "telepathy"})
        b = _Bench({"telepathy"})
        with pytest.raises(CompatibilityError, match="unknown channel 'telepathy'"):
            check_compatibility(m, b)

    def test_neural_region_is_documented(self):
        m = _Model({"vision"}, region_map={"IT": "layer.10"})
        b = _Bench({"vision"}, region="IT")
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompatibilityWarning)
            check_compatibility(m, b)  # neural:IT resolves to the documented neural family
