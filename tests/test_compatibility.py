import pytest
import warnings
from typing import Dict, Set

from brainscore_core.compatibility import (
    CompatibilityError,
    CompatibilityWarning,
    check_compatibility,
)
from brainscore_core.model_interface import UnifiedModel


# ── Helpers ──────────────────────────────────────────────────────────

class FakeModel(UnifiedModel):

    def __init__(self, identifier='test-model', modalities=None, region_layer_map=None):
        self._id = identifier
        self._modalities = modalities or set()
        self._rlm = region_layer_map or {}

    @property
    def identifier(self) -> str:
        return self._id

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return self._rlm

    @property
    def supported_modalities(self) -> Set[str]:
        return self._modalities

    def process(self, stimuli):
        return None


class FakeBenchmark:

    def __init__(self, identifier='test-bench', required_modalities=None,
                 available_modalities=None, region=None):
        self.identifier = identifier
        if required_modalities is not None:
            self.required_modalities = required_modalities
        if available_modalities is not None:
            self.available_modalities = available_modalities
        if region is not None:
            self.region = region


# ── Required modalities (hard gate) ─────────────────────────────────

class TestRequiredModalities:

    def test_matching_modalities_passes(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(required_modalities={'vision'})
        check_compatibility(model, bench)  # should not raise

    def test_superset_modalities_passes(self):
        model = FakeModel(modalities={'vision', 'text'})
        bench = FakeBenchmark(required_modalities={'vision'})
        check_compatibility(model, bench)

    def test_missing_required_raises(self):
        model = FakeModel(modalities={'text'})
        bench = FakeBenchmark(required_modalities={'vision'})
        with pytest.raises(CompatibilityError, match="does not support modalities required"):
            check_compatibility(model, bench)

    def test_missing_one_of_multiple_required_raises(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(required_modalities={'vision', 'text'})
        with pytest.raises(CompatibilityError, match="'text'"):
            check_compatibility(model, bench)

    def test_no_required_modalities_on_benchmark(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark()  # no required_modalities attribute
        check_compatibility(model, bench)  # should not raise


# ── Available modalities (soft warning) ──────────────────────────────

class TestAvailableModalities:

    def test_model_supports_all_available_no_warning(self):
        model = FakeModel(modalities={'vision', 'audio'})
        bench = FakeBenchmark(
            required_modalities={'vision'},
            available_modalities={'audio'},
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            check_compatibility(model, bench)  # no warning

    def test_model_missing_available_warns(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(
            required_modalities={'vision'},
            available_modalities={'audio'},
        )
        with pytest.warns(CompatibilityWarning, match="does not support optional modalities"):
            check_compatibility(model, bench)

    def test_no_available_modalities_on_benchmark(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(required_modalities={'vision'})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            check_compatibility(model, bench)  # no warning


# ── Region mapping ──────────────────────────────────────────────────

class TestRegionMapping:

    def test_region_present_passes(self):
        model = FakeModel(modalities={'vision'}, region_layer_map={'IT': 'layer4'})
        bench = FakeBenchmark(required_modalities={'vision'}, region='IT')
        check_compatibility(model, bench)

    def test_region_missing_raises(self):
        model = FakeModel(modalities={'vision'}, region_layer_map={'V1': 'layer1'})
        bench = FakeBenchmark(required_modalities={'vision'}, region='IT')
        with pytest.raises(CompatibilityError, match="no layer mapping for region 'IT'"):
            check_compatibility(model, bench)

    def test_no_region_on_benchmark(self):
        model = FakeModel(modalities={'vision'}, region_layer_map={})
        bench = FakeBenchmark(required_modalities={'vision'})
        check_compatibility(model, bench)  # no region check needed


# ── Combined scenarios ──────────────────────────────────────────────

class TestCombinedScenarios:

    def test_full_compatible_model(self):
        model = FakeModel(
            modalities={'vision', 'audio'},
            region_layer_map={'IT': 'layer4', 'A1': 'audio.layer3'},
        )
        bench = FakeBenchmark(
            required_modalities={'vision'},
            available_modalities={'audio'},
            region='IT',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            check_compatibility(model, bench)

    def test_degraded_but_valid(self):
        """Model lacks optional audio but has required vision and correct region."""
        model = FakeModel(
            modalities={'vision'},
            region_layer_map={'IT': 'layer4'},
        )
        bench = FakeBenchmark(
            required_modalities={'vision'},
            available_modalities={'audio'},
            region='IT',
        )
        with pytest.warns(CompatibilityWarning):
            check_compatibility(model, bench)

    def test_required_fails_before_region_check(self):
        """Required modality failure should be raised even if region is also bad."""
        model = FakeModel(modalities={'text'}, region_layer_map={})
        bench = FakeBenchmark(
            required_modalities={'vision'},
            region='IT',
        )
        with pytest.raises(CompatibilityError, match="does not support modalities required"):
            check_compatibility(model, bench)

    def test_error_message_includes_identifiers(self):
        model = FakeModel(identifier='gpt-2', modalities={'text'})
        bench = FakeBenchmark(identifier='MajajHong2015', required_modalities={'vision'})
        with pytest.raises(CompatibilityError, match="gpt-2") as exc_info:
            check_compatibility(model, bench)
        assert "MajajHong2015" in str(exc_info.value)
