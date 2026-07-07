from brainscore_core.metrics import Score
from brainscore_core.score_metadata import (
    infer_score_protocol,
    requested_output_channels_for_score,
    stamp_score_metadata,
)


class _Subject:
    identifier = "subject"
    in_channels = {"vision", "text"}


class _Benchmark:
    def __init__(self, *, region=None, parent=None, requested_output_channels=None):
        if region is not None:
            self.region = region
        if parent is not None:
            self.parent = parent
        if requested_output_channels is not None:
            self.requested_output_channels = requested_output_channels


class _AbstractParentBenchmark:
    @property
    def parent(self):
        raise NotImplementedError


def test_stamp_score_metadata_adds_provenance_and_preserves_existing_attrs():
    score = Score(1)
    score.attrs["runtime_sec"] = 0.01

    returned = stamp_score_metadata(
        score,
        _Subject(),
        requested_channels={"neural:IT"},
        protocol="neural",
        harness_id="test-harness",
    )

    assert returned is score
    assert score.attrs["runtime_sec"] == 0.01
    assert score.attrs["in_channels"] == ("text", "vision")
    assert score.attrs["out_channels"] == ("neural:IT",)
    assert score.attrs["protocol"] == "neural"
    assert score.attrs["subject_id"] == "subject"
    assert score.attrs["harness_id"] == "test-harness"


def test_requested_output_channels_use_explicit_or_region_fallback():
    assert requested_output_channels_for_score(
        _Benchmark(requested_output_channels={"behavior"})
    ) == {"behavior"}
    assert requested_output_channels_for_score(_Benchmark(region="IT")) == {
        "neural:IT"
    }


def test_requested_output_channels_infer_behavioral_parent():
    assert requested_output_channels_for_score(
        _Benchmark(parent="behavioral")
    ) == {"behavior"}


def test_parent_lookup_is_best_effort_for_legacy_benchmarks():
    benchmark = _AbstractParentBenchmark()

    assert requested_output_channels_for_score(benchmark) == set()
    assert infer_score_protocol(_Subject(), benchmark, set()) is None


def test_infer_score_protocol_from_requested_channels():
    assert infer_score_protocol(_Subject(), _Benchmark(), {"neural:V4"}) == "neural"
    assert infer_score_protocol(_Subject(), _Benchmark(), {"motor"}) == "motor"
    assert infer_score_protocol(
        _Subject(), _Benchmark(), {"perturbation"}
    ) == "perturbation"


def test_infer_score_protocol_distinguishes_behavior_generation():
    subject = _Subject()
    subject._use_generation_for_task = True

    assert infer_score_protocol(subject, _Benchmark(), {"behavior"}) == (
        "behavior:generation"
    )
