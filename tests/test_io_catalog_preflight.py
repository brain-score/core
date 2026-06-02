"""Tests for the v1.5 Input/Output Catalog wiring into pre-flight.

``check_compatibility`` calls ``check_io_catalog`` as a warn-only documentation
conformance check: it surfaces inputs or neural outputs not documented in the
catalog, without changing the pass/fail result.
"""
import warnings

import pytest

from brainscore_core.model_interface import Subject
from brainscore_core.compatibility import (
    check_compatibility,
    CompatibilityWarning,
)


class _Model(Subject):
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

    @property
    def available_modalities(self):
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

    def test_undocumented_input_warns(self):
        m = _Model({"vision", "telepathy"})
        b = _Bench({"telepathy"})
        with pytest.warns(CompatibilityWarning, match="not in the Input/Output Catalog"):
            check_compatibility(m, b)

    def test_catalog_check_is_warn_only(self):
        # An undocumented modality must NOT raise CompatibilityError; it only warns.
        m = _Model({"telepathy"})
        b = _Bench({"telepathy"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_compatibility(m, b)  # no exception

    def test_neural_region_is_documented(self):
        m = _Model({"vision"}, region_map={"IT": "layer.10"})
        b = _Bench({"vision"}, region="IT")
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompatibilityWarning)
            check_compatibility(m, b)  # neural:IT resolves to the documented neural family
