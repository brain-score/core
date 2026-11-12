import pytest

from brainscore_core.compatibility import (
    CompatibilityError,
    benchmark_accepted_input_channels,
    benchmark_required_input_channels,
    check_channel_compatibility,
)
from brainscore_core.model_interface import Subject


class _Subject(Subject):
    def __init__(
        self,
        in_channels,
        out_channels,
        required_channels=None,
        identifier="subject",
    ):
        self._in_channels = set(in_channels)
        self._out_channels = set(out_channels)
        self._required_channels = set(required_channels or set())
        self._identifier = identifier
        self.process_called = False

    @property
    def identifier(self):
        return self._identifier

    @property
    def in_channels(self):
        return set(self._in_channels)

    @property
    def out_channels(self):
        return set(self._out_channels)

    @property
    def required_channels(self):
        return set(self._required_channels)

    def participate(self, session):
        self.process_called = True


class _Benchmark:
    def __init__(
        self,
        required_input_channels=None,
        requested_output_channels=None,
        required_modalities=None,
        accepted_modalities=None,
        region=None,
        identifier="benchmark",
    ):
        self.identifier = identifier
        if required_input_channels is not None:
            self.required_input_channels = required_input_channels
        if requested_output_channels is not None:
            self.requested_output_channels = requested_output_channels
        if required_modalities is not None:
            self.required_modalities = required_modalities
        if accepted_modalities is not None:
            self.accepted_modalities = accepted_modalities
        if region is not None:
            self.region = region


def test_channel_compatible_pair_passes():
    subject = _Subject(
        in_channels={"vision:image", "text"},
        out_channels={"neural:IT", "behavior"},
        required_channels={"vision:image"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:image"},
        requested_output_channels={"neural:IT"},
    )

    check_channel_compatibility(subject, benchmark)


def test_partial_input_overlap_runs_and_no_overlap_raises():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:image", "text"},
        requested_output_channels={"neural:IT"},
    )

    # Contract v3 rule 5.5: partial matches run; the dropped stream is reported.
    match = check_channel_compatibility(subject, benchmark)
    assert match.kind == "partial"
    assert match.dropped_inputs == {"text"}

    no_overlap = _Benchmark(
        required_input_channels={"text"},
        requested_output_channels={"neural:IT"},
    )
    with pytest.raises(CompatibilityError, match="text"):
        check_channel_compatibility(subject, no_overlap)


def test_missing_requested_output_channel_raises_with_channel_name():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:image"},
        requested_output_channels={"neural:IT", "behavior"},
    )

    with pytest.raises(CompatibilityError, match="behavior"):
        check_channel_compatibility(subject, benchmark)


def test_unmet_subject_required_channel_raises_with_channel_name():
    subject = _Subject(
        in_channels={"vision:image", "text"},
        out_channels={"neural:IT"},
        required_channels={"text"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:image"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="text"):
        check_channel_compatibility(subject, benchmark)


def test_incompatible_pair_raises_before_compute_not_zero_score():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:image"},
        requested_output_channels={"behavior"},
    )

    with pytest.raises(CompatibilityError):
        check_channel_compatibility(subject, benchmark)

    assert subject.process_called is False


def test_required_input_channels_derive_from_required_modalities():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_modalities={"vision"},
        region="IT",
    )

    assert benchmark_required_input_channels(benchmark) == {"vision:image"}
    check_channel_compatibility(subject, benchmark)


def test_accepted_input_channels_derive_from_accepted_modalities():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
        required_channels={"vision:image"},
    )
    benchmark = _Benchmark(
        accepted_modalities={"vision", "video"},
        region="IT",
    )

    assert benchmark_required_input_channels(benchmark) == set()
    # each modality names its own form: image or video
    assert benchmark_accepted_input_channels(benchmark) == {"vision:image", "vision:video"}
    check_channel_compatibility(subject, benchmark)


def test_accepted_input_channels_raise_when_subject_has_none():
    subject = _Subject(
        in_channels={"text"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        accepted_modalities={"vision", "video"},
        region="IT",
    )

    with pytest.raises(CompatibilityError, match="vision"):
        check_channel_compatibility(subject, benchmark)


def test_unknown_channel_declaration_raises():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"telepathy"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="telepathy"):
        check_channel_compatibility(subject, benchmark)


def test_wrong_direction_channel_declaration_raises():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"behavior"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="behavior"):
        check_channel_compatibility(subject, benchmark)


def test_invalid_addressed_channel_declaration_raises():
    subject = _Subject(
        in_channels={"text"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"text:left"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="not addressable"):
        check_channel_compatibility(subject, benchmark)


def test_invalid_channel_grammar_raises():
    subject = _Subject(
        in_channels={"vision:image"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:image"},
        requested_output_channels={"neural:"},
    )

    with pytest.raises(CompatibilityError, match="invalid channel"):
        check_channel_compatibility(subject, benchmark)


def test_bare_vision_declaration_raises_for_subject_and_benchmark():
    with pytest.raises(CompatibilityError, match="names its form"):
        check_channel_compatibility(
            _Subject(in_channels={"vision"}, out_channels={"neural:IT"}),
            _Benchmark(required_input_channels={"vision:image"},
                       requested_output_channels={"neural:IT"}))
    with pytest.raises(CompatibilityError, match="names its form"):
        check_channel_compatibility(
            _Subject(in_channels={"vision:image"}, out_channels={"neural:IT"}),
            _Benchmark(required_input_channels={"vision"},
                       requested_output_channels={"neural:IT"}))


def test_image_subject_and_video_benchmark_do_not_match():
    subject = _Subject(in_channels={"vision:image"}, out_channels={"neural:IT"})
    benchmark = _Benchmark(required_input_channels={"vision:video"},
                           requested_output_channels={"neural:IT"})
    with pytest.raises(CompatibilityError):
        check_channel_compatibility(subject, benchmark)


# ── output_inputs: the inputs an output can be answered from ──────────


class _Listing(_Subject):
    """A vision-language subject: its vision region needs a visual input."""

    def __init__(self, output_inputs, **kwargs):
        super().__init__(**kwargs)
        self._output_inputs = output_inputs

    @property
    def output_inputs(self):
        return self._output_inputs


def test_output_inputs_is_empty_by_default_and_changes_nothing():
    subject = _Subject({"text"}, {"neural:IT"})
    assert subject.output_inputs == {}
    match = check_channel_compatibility(subject, _Benchmark(
        required_input_channels={"text"}, requested_output_channels={"neural:IT"}))
    assert match.kind == "full"


def _vision_language(**kwargs):
    return _Listing({"neural:IT": frozenset({"vision:image", "vision:video"}),
                     "neural:language": frozenset({"text"})},
                    in_channels={"vision:image", "vision:video", "text"},
                    out_channels={"neural:IT", "neural:language"}, **kwargs)


def test_a_listed_output_runs_when_the_benchmark_sends_one_of_its_inputs():
    match = check_channel_compatibility(_vision_language(), _Benchmark(
        required_input_channels={"vision:video", "audio"},
        requested_output_channels={"neural:IT"}))
    assert match.consumed_inputs == {"vision:video"}


def test_a_listed_output_is_na_when_the_benchmark_sends_none_of_its_inputs():
    with pytest.raises(CompatibilityError,
                       match=r"answers 'neural:IT' only from the inputs .*'vision:image'.*"
                             r"takes in only \{'text'\}"):
        check_channel_compatibility(_vision_language(), _Benchmark(
            required_input_channels={"text"}, requested_output_channels={"neural:IT"}))
    # The unlisted output, and the listed output whose input is sent, still run.
    check_channel_compatibility(_vision_language(), _Benchmark(
        required_input_channels={"text"}, requested_output_channels={"neural:language"}))


def test_the_existing_rules_still_hold_with_a_listing():
    with pytest.raises(CompatibilityError, match="cannot produce output channels"):
        check_channel_compatibility(_vision_language(), _Benchmark(
            required_input_channels={"text"}, requested_output_channels={"neural:V4"}))
    with pytest.raises(CompatibilityError, match="takes in none of the inputs"):
        check_channel_compatibility(_vision_language(), _Benchmark(
            required_input_channels={"audio"}, requested_output_channels={"neural:language"}))
    with pytest.raises(CompatibilityError, match="hard-requires"):
        check_channel_compatibility(_vision_language(required_channels={"text"}), _Benchmark(
            required_input_channels={"vision:image"}, requested_output_channels={"neural:IT"}))


def test_listed_inputs_are_compared_with_the_inputs_the_subject_takes_in():
    """A listed input the benchmark sends but the subject does not take in does not count."""
    subject = _Listing({"neural:IT": frozenset({"vision:image"})},
                       in_channels={"vision:image", "text"}, out_channels={"neural:IT"})
    match = check_channel_compatibility(subject, _Benchmark(
        required_input_channels={"vision:image", "audio"},
        requested_output_channels={"neural:IT"}))
    assert match.consumed_inputs == {"vision:image"}


@pytest.mark.parametrize("listing, message", [
    ({"neural:V4": frozenset({"vision:image"})}, "not in out_channels"),
    ({"neural:IT": frozenset({"audio"})}, "not in in_channels"),
])
def test_an_inconsistent_listing_is_an_error(listing, message):
    subject = _Listing(listing, in_channels={"vision:image", "text"}, out_channels={"neural:IT"})
    with pytest.raises(CompatibilityError, match=message):
        check_channel_compatibility(subject, _Benchmark(
            required_input_channels={"vision:image"}, requested_output_channels={"neural:IT"}))
