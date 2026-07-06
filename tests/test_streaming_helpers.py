import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore_core.model_interface import BrainScoreModel, Subject, TaskContext
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    behavior_session,
    score_behavior,
    score_stimuli,
    stimulus_session,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from tests.test_behavioral_readout import (
    _fake_preprocessor_returning,
    _make_image_stimulus_set,
)


def _stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        "stimulus_id": ["s0", "s1"],
        "image_path": ["s0.png", "s1.png"],
        "sentence": ["cat", "dog"],
        "condition": ["a", "b"],
    }))
    stimuli.identifier = "synthetic"
    return stimuli


def _assembly():
    return NeuroidAssembly(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={
            "stimulus_id": ("presentation", ["s0", "s1"]),
            "condition": ("presentation", ["a", "b"]),
            "neuroid_id": ("neuroid", ["synthetic.layer4.0", "synthetic.layer4.1"]),
            "neuroid_num": ("neuroid", [0, 1]),
            "model": ("neuroid", ["synthetic"] * 2),
            "layer": ("neuroid", ["layer4"] * 2),
        },
        dims=["presentation", "neuroid"],
    )


class _SyntheticSubject(Subject):
    def __init__(self, output):
        self.output = output
        self.recording_target = None
        self.process_input = None

    @property
    def identifier(self):
        return "synthetic"

    @property
    def region_layer_map(self):
        return {"IT": "layer4"}

    @property
    def supported_modalities(self):
        return {"vision"}

    def start_recording(self, recording_target, **kwargs):
        self.recording_target = recording_target

    def process(self, input_event):
        self.process_input = input_event
        return self.output

    def interact(self, session):
        raise AssertionError("Phase 2 score_stimuli must use _drive_via_process")


def test_stimulus_session_emits_input_events_per_stimulus_column():
    session = stimulus_session(_stimulus_set(), record="IT")

    events = []
    while True:
        event = session.next_input()
        if event is None:
            break
        events.append(event)

    assert [event.channel for event in events] == [
        "vision", "text", "vision", "text",
    ]
    assert [event.payload for event in events] == [
        "s0.png", "cat", "s1.png", "dog",
    ]
    assert [event.meta["stimulus_id"] for event in events] == [
        "s0", "s0", "s1", "s1",
    ]
    assert [event.meta["column"] for event in events] == [
        "image_path", "sentence", "image_path", "sentence",
    ]


def test_collect_packages_raw_neural_events_as_neuroid_assembly():
    session = stimulus_session(_stimulus_set(), record="IT")
    session.emit(StreamEvent(
        channel="neural:IT",
        payload=np.array([1.0, 2.0]),
        t_ms=0.0,
        meta={"stimulus_id": "s0"},
    ))
    session.emit(StreamEvent(
        channel="neural:IT",
        payload=np.array([3.0, 4.0]),
        t_ms=1.0,
        meta={"stimulus_id": "s1"},
    ))

    collected = session.collect("neural:IT")

    assert isinstance(collected, NeuroidAssembly)
    assert collected.dims == ("presentation", "neuroid")
    np.testing.assert_array_equal(collected.values, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert list(collected["stimulus_id"].values) == ["s0", "s1"]
    assert list(collected["region"].values) == ["IT", "IT"]


def test_score_stimuli_matches_process_assembly_exactly():
    expected = _assembly()
    subject = _SyntheticSubject(expected)
    stimuli = _stimulus_set()

    scored = score_stimuli(subject, stimuli, record="IT")

    assert subject.recording_target == "IT"
    assert subject.process_input is stimuli
    xr.testing.assert_identical(scored, expected)


def test_collect_returns_emitted_assembly_without_repackaging():
    expected = _assembly()
    session = stimulus_session(_stimulus_set(), record="IT")
    session.emit(StreamEvent(
        channel="neural:IT",
        payload=expected,
        t_ms=0.0,
        meta={"stimulus_id": "synthetic"},
    ))

    assert session.collect("neural:IT") is expected


def test_collect_rejects_missing_channel():
    session = stimulus_session(_stimulus_set(), record="IT")

    with pytest.raises(ValueError, match="No emitted events"):
        session.collect("neural:IT")


def _behavior_model(features):
    return BrainScoreModel(
        identifier="test-model",
        model=None,
        region_layer_map={},
        preprocessors={"vision": _fake_preprocessor_returning(features)},
        behavioral_readout_layer="some_layer",
    )


def _behavior_context(fitting_stimuli, scoring_stimuli):
    return TaskContext(
        task_type="probabilities",
        fitting_stimuli=fitting_stimuli,
        label_set=["cat", "dog"],
        instruction="Choose the animal",
        metadata={"stimulus_set": scoring_stimuli},
    )


def test_behavior_session_emits_instruction_fitting_and_scoring_events():
    fitting = _make_image_stimulus_set(["cat", "dog"], identifier="fit")
    scoring = _make_image_stimulus_set(["cat", "dog"], identifier="score")
    session = behavior_session(_behavior_context(fitting, scoring))

    events = []
    while True:
        event = session.next_input()
        if event is None:
            break
        events.append(event)

    assert events[0].channel == "instruction"
    assert events[0].payload == "Choose the animal"
    assert events[0].meta["role"] == "instruction"

    stimulus_events = events[1:]
    assert [event.channel for event in stimulus_events] == [
        "vision", "vision", "vision", "vision",
    ]
    assert [event.meta["role"] for event in stimulus_events] == [
        "fitting", "fitting", "scoring", "scoring",
    ]
    assert [event.meta["stimulus_id"] for event in stimulus_events] == [
        "s0", "s1", "s0", "s1",
    ]


def test_score_behavior_matches_legacy_start_task_process_exactly():
    fitting = _make_image_stimulus_set(["cat"] * 6 + ["dog"] * 6)
    scoring = _make_image_stimulus_set(
        ["cat"] * 3 + ["dog"] * 3, identifier="behavior_score"
    )
    features = np.random.default_rng(42).normal(size=(12, 8))
    features[:6, 0] += 3.0
    features[6:, 0] -= 3.0

    legacy = _behavior_model(features)
    legacy_context = _behavior_context(fitting, scoring)
    legacy.start_task(legacy_context)
    expected = legacy.process(scoring)

    helper_subject = _behavior_model(features)
    helper_context = _behavior_context(fitting, scoring)
    scored = score_behavior(helper_subject, helper_context)

    assert isinstance(scored, BehavioralAssembly)
    xr.testing.assert_identical(scored, expected)


def _generation_model(calls):
    def fake_generate(stimulus_row, instruction, label_set):
        calls.append((stimulus_row["stimulus_id"], instruction, tuple(label_set)))
        return "cat" if stimulus_row["stimulus_id"] in {"s0", "s1", "s2"} else "dog"

    return BrainScoreModel(
        identifier="test-vlm",
        model=None,
        region_layer_map={},
        preprocessors={"vision": lambda *args, **kwargs: None},
        generation_fn=fake_generate,
    )


def _generation_context(scoring_stimuli):
    return TaskContext(
        task_type="probabilities",
        label_set=["cat", "dog"],
        instruction="What animal is this?",
        metadata={"stimulus_set": scoring_stimuli},
    )


def test_score_behavior_matches_legacy_generation_path_exactly():
    scoring = _make_image_stimulus_set(
        ["cat"] * 3 + ["dog"] * 3, identifier="gen_test"
    )

    legacy_calls = []
    legacy = _generation_model(legacy_calls)
    legacy_context = _generation_context(scoring)
    legacy.start_task(legacy_context)
    expected = legacy.process(scoring)

    helper_calls = []
    helper_subject = _generation_model(helper_calls)
    helper_context = _generation_context(scoring)
    scored = score_behavior(helper_subject, helper_context)

    assert isinstance(scored, BehavioralAssembly)
    xr.testing.assert_identical(scored, expected)
    assert helper_calls == legacy_calls


def test_behavior_collect_packages_raw_label_events():
    session = behavior_session(TaskContext(task_type="probabilities"))
    session.emit(StreamEvent(
        channel="behavior",
        payload="cat",
        t_ms=0.0,
        meta={"stimulus_id": "s0", "label_set": ["cat", "dog"]},
    ))
    session.emit(StreamEvent(
        channel="behavior",
        payload="dog",
        t_ms=1.0,
        meta={"stimulus_id": "s1", "label_set": ["cat", "dog"]},
    ))

    collected = session.collect("behavior")

    assert isinstance(collected, BehavioralAssembly)
    assert collected.dims == ("presentation", "choice")
    np.testing.assert_array_equal(collected.values, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert list(collected["choice"].values) == ["cat", "dog"]
    assert list(collected["stimulus_id"].values) == ["s0", "s1"]


def test_behavior_collect_packages_raw_probability_events():
    session = behavior_session(TaskContext(task_type="probabilities"))
    session.emit(StreamEvent(
        channel="behavior",
        payload=np.array([0.25, 0.75]),
        t_ms=0.0,
        meta={"stimulus_id": "s0", "label_set": ["cat", "dog"]},
    ))
    session.emit(StreamEvent(
        channel="behavior",
        payload=np.array([0.8, 0.2]),
        t_ms=1.0,
        meta={"stimulus_id": "s1", "label_set": ["cat", "dog"]},
    ))

    collected = session.collect("behavior")

    assert isinstance(collected, BehavioralAssembly)
    np.testing.assert_array_equal(collected.values, np.array([[0.25, 0.75], [0.8, 0.2]]))
    assert list(collected["choice"].values) == ["cat", "dog"]


def test_behavior_collect_rejects_missing_channel():
    session = behavior_session(TaskContext(task_type="probabilities"))

    with pytest.raises(ValueError, match="No emitted events"):
        session.collect("behavior")
