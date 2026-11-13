import pytest
import warnings
from typing import Dict, Set

from brainscore_core.compatibility import (
    CompatibilityError,
    check_compatibility,
)
from brainscore_core.model_interface import UnifiedModel


# ── Helpers ──────────────────────────────────────────────────────────

class FakeModel(UnifiedModel):

    def __init__(self, identifier='test-model', modalities=None,
                 region_layer_map=None, required_modalities=None):
        self._id = identifier
        self._modalities = modalities or set()
        self._rlm = region_layer_map or {}
        self._required = required_modalities or set()

    @property
    def identifier(self) -> str:
        return self._id

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return self._rlm

    @property
    def supported_modalities(self) -> Set[str]:
        return self._modalities

    @property
    def required_modalities(self) -> Set[str]:
        return set(self._required)

    def process(self, stimuli):
        return None


class FakeBenchmark:

    def __init__(self, identifier='test-bench', required_modalities=None,
                 available_modalities=None, accepted_modalities=None, region=None):
        self.identifier = identifier
        if required_modalities is not None:
            self.required_modalities = required_modalities
        if available_modalities is not None:
            self.available_modalities = available_modalities
        if accepted_modalities is not None:
            self.accepted_modalities = accepted_modalities
        if region is not None:
            self.region = region


# ── Required modalities (hard gate) ─────────────────────────────────

class TestRequiredModalities:

    def test_matching_modalities_passes(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(required_modalities={'vision'})
        match = check_compatibility(model, bench)
        assert match.kind == 'full'
        assert match.consumed_inputs == {'vision:image'}

    def test_missing_required_raises(self):
        model = FakeModel(modalities={'text'})
        bench = FakeBenchmark(required_modalities={'vision'})
        with pytest.raises(CompatibilityError, match="takes in none of the inputs"):
            check_compatibility(model, bench)

    def test_missing_one_of_multiple_required_runs_as_partial(self):
        # Contract v3 rule 5: a subject that takes in some of the inputs runs;
        # the rest never enter it, and the match says so.
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(required_modalities={'vision', 'text'})
        match = check_compatibility(model, bench)
        assert match.kind == 'partial'
        assert match.consumed_inputs == {'vision:image'}
        assert match.dropped_inputs == {'text'}

    def test_full_match_reports_full(self):
        match = check_compatibility(FakeModel(modalities={'vision', 'text'}),
                                    FakeBenchmark(required_modalities={'vision'}))
        assert match.kind == 'full'
        assert match.dropped_inputs == frozenset()
        assert match.as_attrs()['match'] == 'full'

    def test_benchmark_without_inputs_is_rejected_by_name(self):
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(identifier='silent-bench')  # no input declaration
        with pytest.raises(CompatibilityError, match="'silent-bench' declares no input channels"):
            check_compatibility(model, bench)

    def test_empty_declared_inputs_are_rejected(self):
        bench = FakeBenchmark(identifier='empty-bench')
        bench.required_input_channels = frozenset()
        with pytest.raises(CompatibilityError, match="'empty-bench' declares no input channels"):
            check_compatibility(FakeModel(modalities={'vision'}), bench)

    @pytest.mark.parametrize('name, field', [
        ('input_channels', 'required_input_channels'),
        ('output_channels', 'requested_output_channels')])
    def test_protocol_spelling_on_benchmark_raises(self, name, field):
        """Declarations under a subject or protocol name must not pass unchecked."""
        bench = FakeBenchmark()
        setattr(bench, name, {'text'} if name == 'input_channels' else {'neural:IT'})
        with pytest.raises(CompatibilityError, match=field):
            check_compatibility(FakeModel(modalities={'vision'}), bench)

    def test_half_misspelled_declaration_raises(self):
        bench = FakeBenchmark()
        bench.required_input_channels = {'vision'}
        bench.output_channels = {'neural:IT'}
        with pytest.raises(CompatibilityError, match='requested_output_channels'):
            check_compatibility(FakeModel(modalities={'vision'}), bench)

    def test_unrelated_legacy_metadata_is_ignored(self):
        bench = FakeBenchmark(required_modalities={'vision'})
        bench.input_channels = 3  # e.g. image color channels, not channel names
        check_compatibility(FakeModel(modalities={'vision'}), bench)


# ── Accepted modalities (any-of gate) ───────────────────────────────

class TestAcceptedModalities:

    def test_required_plus_accepted_combo(self):
        # multimodal A+V: audio required (all-of) + video|vision accepted (any-of)
        bench = FakeBenchmark(required_modalities={'audio'},
                              accepted_modalities={'video', 'vision'})
        assert check_compatibility(FakeModel(modalities={'audio', 'video'}), bench).consumed_inputs == {
            'audio', 'vision:video'}
        assert check_compatibility(FakeModel(modalities={'audio', 'vision'}), bench).consumed_inputs == {
            'audio', 'vision:image'}
        # Missing audio or missing both visual formats: partial, still runs.
        assert check_compatibility(FakeModel(modalities={'video'}), bench).dropped_inputs == {'audio', 'vision:image'}
        assert check_compatibility(FakeModel(modalities={'audio'}), bench).dropped_inputs == {
            'vision:image', 'vision:video'}
        with pytest.raises(CompatibilityError, match="takes in none of the inputs"):
            check_compatibility(FakeModel(modalities={'text'}), bench)

    def test_model_hard_requires_accepted_format_passes(self):
        # Check-2 fix: a model hard-requiring video isn't rejected when the
        # benchmark ACCEPTS video (required ∪ accepted).
        bench = FakeBenchmark(accepted_modalities={'video', 'vision'})
        model = FakeModel(modalities={'video'}, required_modalities={'video'})
        assert check_compatibility(model, bench).consumed_inputs == {'vision:video'}


class TestVisualFormsAreSeparateChannels:
    """image and video are two channels; neither stands for the other."""

    def test_video_benchmark_does_not_run_an_image_only_model(self):
        with pytest.raises(CompatibilityError):
            check_compatibility(FakeModel(modalities={'vision'}),
                                FakeBenchmark(required_modalities={'video'}))

    def test_image_benchmark_does_not_run_a_video_only_model(self):
        with pytest.raises(CompatibilityError):
            check_compatibility(FakeModel(modalities={'video'}),
                                FakeBenchmark(required_modalities={'vision'}))

    def test_benchmark_with_both_forms_runs_either_model(self):
        bench = FakeBenchmark(accepted_modalities={'vision', 'video'})
        assert check_compatibility(FakeModel(modalities={'vision'}), bench).consumed_inputs == {'vision:image'}
        assert check_compatibility(FakeModel(modalities={'video'}), bench).consumed_inputs == {'vision:video'}

    def test_a_bare_vision_declaration_is_refused(self):
        bench = FakeBenchmark()
        bench.required_input_channels = {'vision'}
        with pytest.raises(CompatibilityError, match="names its form"):
            check_compatibility(FakeModel(modalities={'vision'}), bench)


# ── Deprecated benchmark.available_modalities ────────────────────────

class TestDeprecatedBenchmarkAvailableModalities:
    """Post-April 30, 2026: benchmark-side available_modalities is deprecated.
    Setting it emits a DeprecationWarning and otherwise no longer affects
    compatibility (the benchmark's *required* set is now its full input
    format declaration)."""

    def test_setting_available_emits_deprecation_warning(self):
        model = FakeModel(modalities={'vision', 'audio'})
        bench = FakeBenchmark(
            required_modalities={'vision'},
            available_modalities={'audio'},
        )
        with pytest.warns(DeprecationWarning, match="deprecated as of the April 30"):
            check_compatibility(model, bench)

    def test_redundant_available_no_warning(self):
        """available_modalities matching required is not a multi-input claim;
        no deprecation warning needed."""
        model = FakeModel(modalities={'vision'})
        bench = FakeBenchmark(
            required_modalities={'vision'},
            available_modalities={'vision'},
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            check_compatibility(model, bench)  # no warning

# ── Region mapping ──────────────────────────────────────────────────

class TestRegionMapping:

    def test_region_present_passes(self):
        model = FakeModel(modalities={'vision'}, region_layer_map={'IT': 'layer4'})
        bench = FakeBenchmark(required_modalities={'vision'}, region='IT')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            match = check_compatibility(model, bench)
        assert match.kind == 'full'
        assert match.requested_output_channels == {'neural:IT'}

    def test_region_missing_raises(self):
        model = FakeModel(modalities={'vision'}, region_layer_map={'V1': 'layer1'})
        bench = FakeBenchmark(required_modalities={'vision'}, region='IT')
        with pytest.raises(CompatibilityError, match="no layer mapping for region 'IT'"):
            check_compatibility(model, bench)

    def test_no_region_on_benchmark(self):
        model = FakeModel(modalities={'vision'}, region_layer_map={})
        bench = FakeBenchmark(required_modalities={'vision'})
        assert check_compatibility(model, bench).requested_output_channels == frozenset()


# ── Combined scenarios ──────────────────────────────────────────────

class TestCombinedScenarios:

    def test_missing_output_reported_before_inputs(self):
        """A missing requested output is reported first, even if inputs also fail."""
        model = FakeModel(modalities={'text'}, region_layer_map={})
        bench = FakeBenchmark(
            required_modalities={'vision'},
            region='IT',
        )
        with pytest.raises(CompatibilityError, match="no layer mapping for region 'IT'"):
            check_compatibility(model, bench)

    def test_error_message_includes_identifiers(self):
        model = FakeModel(identifier='gpt-2', modalities={'text'})
        bench = FakeBenchmark(identifier='MajajHong2015', required_modalities={'vision'})
        with pytest.raises(CompatibilityError, match="gpt-2") as exc_info:
            check_compatibility(model, bench)
        assert "MajajHong2015" in str(exc_info.value)


# ── Model-side required modalities (hard gate) ──────────────────────

class TestModelRequiredModalities:
    """A model can declare modalities it hard-requires the benchmark to
    provide (e.g. a locked-fusion model that needs vision+audio+text
    together, or a unimodal backbone that cannot produce a prediction
    without its one modality).

    Post April 30, 2026: each benchmark declares ONE input format, so the
    model's required set must be a subset of the benchmark's required set.
    """

    def test_locked_fusion_model_full_provision(self):
        # Locked fusion that exactly matches the benchmark's input format
        model = FakeModel(
            modalities={'vision', 'audio', 'text'},
            required_modalities={'vision', 'audio', 'text'},
        )
        bench = FakeBenchmark(
            required_modalities={'vision', 'audio', 'text'},
        )
        assert check_compatibility(model, bench).kind == 'full'

    def test_locked_fusion_missing_audio_raises(self):
        # Model needs audio, benchmark only provides vision + text
        model = FakeModel(
            modalities={'vision', 'audio', 'text'},
            required_modalities={'vision', 'audio', 'text'},
        )
        bench = FakeBenchmark(
            identifier='movie-watching-no-sound',
            required_modalities={'vision', 'text'},
        )
        with pytest.raises(CompatibilityError, match="hard-requires") as exc_info:
            check_compatibility(model, bench)
        message = str(exc_info.value)
        assert 'test-model' in message and 'movie-watching-no-sound' in message
        assert 'audio' in message

# ── Available-modalities API on UnifiedModel ────────────────────────

class TestAvailableModalitiesDefault:
    """A model that states only supported_modalities takes its input channels
    from them and requires none."""

    def test_supported_modalities_give_the_inputs_and_no_requirement(self):
        class _OnlySupported(UnifiedModel):
            identifier = 'only-supported'
            region_layer_map = {}
            supported_modalities = {'vision', 'text'}

        model = _OnlySupported()
        assert model.available_modalities == {'vision', 'text'}
        assert model.in_channels == {'vision:image', 'text'}
        assert model.required_modalities == set()
        assert not model.required_channels
