import pytest

from brainscore_core.compatibility import (
    CompatibilityError,
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
    def region_layer_map(self):
        return {}

    @property
    def supported_modalities(self):
        return set()

    @property
    def in_channels(self):
        return set(self._in_channels)

    @property
    def out_channels(self):
        return set(self._out_channels)

    @property
    def required_channels(self):
        return set(self._required_channels)

    def process(self, input_event):
        self.process_called = True
        return 0


class _Benchmark:
    def __init__(
        self,
        required_input_channels=None,
        requested_output_channels=None,
        required_modalities=None,
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
        if region is not None:
            self.region = region


def test_channel_compatible_pair_passes():
    subject = _Subject(
        in_channels={"vision", "text"},
        out_channels={"neural:IT", "behavior"},
        required_channels={"vision"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision"},
        requested_output_channels={"neural:IT"},
    )

    check_channel_compatibility(subject, benchmark)


def test_missing_required_input_channel_raises_with_channel_name():
    subject = _Subject(
        in_channels={"vision"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision", "text"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="text"):
        check_channel_compatibility(subject, benchmark)


def test_missing_requested_output_channel_raises_with_channel_name():
    subject = _Subject(
        in_channels={"vision"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision"},
        requested_output_channels={"neural:IT", "behavior"},
    )

    with pytest.raises(CompatibilityError, match="behavior"):
        check_channel_compatibility(subject, benchmark)


def test_unmet_subject_required_channel_raises_with_channel_name():
    subject = _Subject(
        in_channels={"vision", "text"},
        out_channels={"neural:IT"},
        required_channels={"text"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="text"):
        check_channel_compatibility(subject, benchmark)


def test_incompatible_pair_raises_before_compute_not_zero_score():
    subject = _Subject(
        in_channels={"vision"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision"},
        requested_output_channels={"behavior"},
    )

    with pytest.raises(CompatibilityError):
        check_channel_compatibility(subject, benchmark)

    assert subject.process_called is False


def test_required_input_channels_derive_from_required_modalities():
    subject = _Subject(
        in_channels={"vision"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_modalities={"vision"},
        region="IT",
    )

    assert benchmark_required_input_channels(benchmark) == {"vision"}
    check_channel_compatibility(subject, benchmark)


def test_unknown_channel_declaration_raises():
    subject = _Subject(
        in_channels={"vision"},
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
        in_channels={"vision"},
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
        in_channels={"vision"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision:left"},
        requested_output_channels={"neural:IT"},
    )

    with pytest.raises(CompatibilityError, match="not addressable"):
        check_channel_compatibility(subject, benchmark)


def test_invalid_channel_grammar_raises():
    subject = _Subject(
        in_channels={"vision"},
        out_channels={"neural:IT"},
    )
    benchmark = _Benchmark(
        required_input_channels={"vision"},
        requested_output_channels={"neural:"},
    )

    with pytest.raises(CompatibilityError, match="invalid channel"):
        check_channel_compatibility(subject, benchmark)
