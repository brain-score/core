"""Exercise the public session contract without any legacy model methods."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from brainscore_core import Subject, TaskContext
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.memory import _probe_feature_dim
from brainscore_core.streaming import InMemorySession, Session, StreamEvent
from brainscore_core.streaming_helpers import behavioral_response, neural_response


class TextSubject(Subject):
    identifier = "native-text"
    in_channels = {"text", "instruction"}
    out_channels = {"neural:language_network", "behavior"}
    required_channels = {"text"}

    def __init__(self):
        self.seen = []

    def interact(self, session):
        while (event := session.next_input()) is not None:
            self.seen.append(event)
            if event.channel != "text":
                continue
            session.emit(StreamEvent(
                "neural:language_network", np.array([len(event.payload), 1.0]),
                event.t_ms, dict(event.meta),
            ))
            session.emit(StreamEvent(
                "behavior", event.payload.upper(), event.t_ms, dict(event.meta),
            ))

    def reset(self):
        self.seen.clear()


def stimuli():
    return pd.DataFrame({
        "stimulus_id": ["a", "b"], "text": ["cat", "horse"],
        "t_ms": [10.0, 25.0],
    })


def benchmark(**overrides):
    values = dict(identifier="native-benchmark", uses_session=True,
                  required_input_channels={"text"},
                  requested_output_channels={"neural:language_network"})
    return SimpleNamespace(**(values | overrides))


def test_minimal_subject_consumes_and_emits_timestamped_events():
    subject = TextSubject()
    for name in ("process", "start_task", "start_recording", "region_layer_map",
                 "supported_modalities", "check_session_support"):
        assert not hasattr(subject, name)
    session = InMemorySession([StreamEvent("text", "cat", 12.0, {"stimulus_id": "a"})])
    subject.interact(session)
    assert [event.channel for event in session.emitted] == [
        "neural:language_network", "behavior",
    ]
    assert all(event.t_ms == 12.0 for event in session.emitted)
    assert all(event.meta["stimulus_id"] == "a" for event in session.emitted)
    subject.reset()
    assert subject.seen == []


def test_minimal_live_session_needs_no_collection_or_model_probe():
    class FeedbackSession(Session):
        def __init__(self):
            self.responses = []

        def next_input(self):
            if len(self.responses) == 2:
                return None
            text = "cat" if not self.responses else self.responses[-1]
            return StreamEvent("text", text, float(len(self.responses)))

        def emit(self, event):
            if event.channel == "behavior":
                self.responses.append(event.payload)

    session = FeedbackSession()
    TextSubject().interact(session)
    assert session.responses == ["CAT", "CAT"]


@pytest.mark.parametrize("missing", ["identifier", "in_channels", "out_channels", "interact"])
def test_incomplete_native_subject_is_rejected(missing):
    declarations = dict(identifier="incomplete", in_channels={"text"},
                        out_channels={"behavior"}, interact=lambda self, session: None)
    del declarations[missing]
    incomplete = type("Incomplete", (Subject,), declarations)
    with pytest.raises(TypeError, match=missing):
        incomplete()


def test_optional_requirements_and_stateless_reset_have_safe_defaults():
    class Stateless(Subject):
        identifier = "stateless"
        in_channels = {"text"}
        out_channels = {"behavior"}

        def interact(self, session):
            pass

    first, second = Stateless(), Stateless()
    first.required_channels.add("text")
    assert second.required_channels == set()
    assert first.reset() is None


def test_native_subject_runs_through_neural_and_behavioral_helpers():
    subject = TextSubject()
    neural = neural_response(subject, stimuli(), record="language_network")
    np.testing.assert_array_equal(neural.values, [[3, 1], [5, 1]])
    assert list(neural["stimulus_id"].values) == ["a", "b"]
    behavior = behavioral_response(subject, TaskContext(
        task_type="generation", instruction="Uppercase the text",
        metadata={"stimulus_set": stimuli()},
    ))
    np.testing.assert_array_equal(behavior.values, [[1, 0], [0, 1]])
    assert list(behavior["choice"].values) == ["CAT", "HORSE"]
    assert list(behavior["stimulus_id"].values) == ["a", "b"]


def test_scoring_preflight_accepts_native_contract_without_legacy_properties():
    subject = TextSubject()
    check_compatibility(subject, benchmark())
    assert subject.seen == []


@pytest.mark.parametrize("overrides, missing", [
    ({"required_input_channels": {"vision"}}, "vision"),
    ({"requested_output_channels": {"motor"}}, "motor"),
    ({"required_input_channels": {"instruction"}}, "text"),
])
def test_native_preflight_rejects_incompatible_pair_before_interaction(overrides, missing):
    subject = TextSubject()
    with pytest.raises(CompatibilityError, match=missing):
        check_compatibility(subject, benchmark(**overrides))
    assert subject.seen == []


def test_optional_model_specific_probe_is_still_honored():
    class Restricted(TextSubject):
        def check_session_support(self, channels):
            raise ValueError("unsupported combination")

    with pytest.raises(CompatibilityError, match="unsupported combination"):
        check_compatibility(Restricted(), benchmark())


def test_memory_feature_probe_uses_native_session_without_layer_map():
    subject = TextSubject()
    assert _probe_feature_dim(subject, benchmark(), stimuli()) == (2, "language_network")
    assert len(subject.seen) == 1
