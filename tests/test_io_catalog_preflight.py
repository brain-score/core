"""Tests for the v1.5 Input/Output Catalog wiring into pre-flight.

``check_compatibility`` rejects an input that is not in the catalog, and accepts
a documented input or neural region without a warning.
"""
import warnings

import pytest

from tests.typed_subject import TypedSubject
from tests.unregistered import unregistered_family
from brainscore_core.compatibility import (
    check_compatibility,
    CompatibilityError,
    CompatibilityWarning,
)


class _Model(TypedSubject):
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
    def in_channels(self):
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
        m = _Model({"vision:image"})
        b = _Bench({"vision"})
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompatibilityWarning)
            check_compatibility(m, b)  # vision is documented; must not warn

    @pytest.mark.parametrize('side', ['subject', 'benchmark'])
    def test_an_unknown_family_is_a_malformed_declaration(self, side):
        # Rule 3.2: every channel family is registered. A name outside the catalog
        # is a defect of the declaration (ValueError), never an N/A pair.
        unknown = unregistered_family()
        subject_in, bench_in = {"vision:image"}, {"vision"}
        (subject_in if side == 'subject' else bench_in).add(unknown)
        with pytest.raises(ValueError, match=unknown) as raised:
            check_compatibility(_Model(subject_in), _Bench(bench_in))
        assert not isinstance(raised.value, CompatibilityError)
        assert "'vision'" in str(raised.value)      # the known families are named

    def test_neural_region_is_documented(self):
        m = _Model({"vision:image"}, region_map={"IT": "layer.10"})
        b = _Bench({"vision"}, region="IT")
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompatibilityWarning)
            check_compatibility(m, b)  # neural:IT resolves to the documented neural family
