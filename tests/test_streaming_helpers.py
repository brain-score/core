import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore_core.model_interface import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import score_stimuli, stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


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
