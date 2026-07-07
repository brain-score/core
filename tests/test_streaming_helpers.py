import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore_core import io_catalog
from brainscore_core.model_interface import (
    BrainScoreModel,
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
    Subject,
    TaskContext,
)
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    apply_state_change,
    behavior_session,
    environment_session,
    score_behavior,
    score_stimuli,
    state_change_session,
    stimulus_session,
    run_environment,
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
from tests.test_environment_step import (
    _droid_action_fn,
    _droid_step,
    _make_model as _make_environment_model,
)
from tests.test_model_interface import (
    _XArrayActivationsModel,
    _XArrayPreprocessor,
    make_stub_preprocessor,
)
from tests.test_state_change import (
    _counting_state_change_fn,
    _make_model as _make_state_change_model,
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


def _metadata_stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        "stimulus_id": ["s0", "s1"],
        "image_path": ["s0.png", "s1.png"],
        "object_name": ["cat", "dog"],
        "difficulty": [1, 2],
    }))
    stimuli.identifier = "metadata-synthetic"
    return stimuli


def _multimodal_stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        "stimulus_id": ["s0", "s1"],
        "image_file_name": ["s0.png", "s1.png"],
        "sentence": ["hello", "world"],
    }))
    stimuli.identifier = "multimodal-synthetic"
    return stimuli


class _MetadataPreservingExtractor:
    identifier = "metadata-preserving-extractor"

    def __init__(self):
        self.last_stimuli = None

    def __call__(self, stimuli, layers=None, **kwargs):
        del kwargs
        self.last_stimuli = stimuli
        layer = (layers or ["layer4"])[0]
        data = np.array([[1.0, 2.0], [3.0, 4.0]])[:len(stimuli)]
        return NeuroidAssembly(
            data,
            coords={
                "stimulus_id": (
                    "presentation", list(stimuli["stimulus_id"].values)
                ),
                "object_name": (
                    "presentation", list(stimuli["object_name"].values)
                ),
                "difficulty": (
                    "presentation", list(stimuli["difficulty"].values)
                ),
                "neuroid_id": (
                    "neuroid", [f"{layer}.{index}" for index in range(data.shape[1])]
                ),
                "layer": ("neuroid", [layer] * data.shape[1]),
            },
            dims=["presentation", "neuroid"],
        )


class _InteractTrackingBrainScoreModel(BrainScoreModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interact_called = False
        self.interact_requested_output_channels = None

    def interact(self, session):
        self.interact_called = True
        self.interact_requested_output_channels = tuple(
            session.requested_output_channels
        )
        return super().interact(session)


def _native_neural_model(extractor, model_cls=BrainScoreModel):
    return model_cls(
        identifier="native-neural",
        model=None,
        region_layer_map={"IT": "layer4"},
        preprocessors={"vision": extractor},
    )


def _native_state_change_model(state_change_fn, model_cls=BrainScoreModel):
    return model_cls(
        identifier="native-state-change",
        model=None,
        region_layer_map={},
        preprocessors={"vision": lambda stimuli: stimuli},
        state_change_fn=state_change_fn,
    )


def _native_environment_model(action_fn, model_cls=BrainScoreModel):
    return model_cls(
        identifier="native-environment",
        model=None,
        region_layer_map={},
        preprocessors={"vision": lambda stimuli: stimuli},
        action_fn=action_fn,
    )


def _native_multimodal_model(model_cls=BrainScoreModel):
    activations = _XArrayActivationsModel()
    text = _XArrayPreprocessor(label="text")
    model = model_cls(
        identifier="native-multimodal",
        model=None,
        region_layer_map={"IT": "layer.10"},
        preprocessors={
            "vision": make_stub_preprocessor(),
            "text": text,
        },
        activations_model=activations,
    )
    return model, activations, text


def _native_cross_tower_model(model_cls=BrainScoreModel):
    activations = _XArrayActivationsModel()
    text = _XArrayPreprocessor(label="text")
    model = model_cls(
        identifier="native-cross-tower",
        model=None,
        region_layer_map={
            "IT": "vision_model.layer.10",
            "language_network": "text_model.layer.20",
        },
        preprocessors={
            "vision": make_stub_preprocessor(),
            "text": text,
        },
        activations_model=activations,
        region_modality_map={
            "IT": "vision",
            "language_network": "text",
        },
    )
    return model, activations, text


class _TemporalBrainScoreModel(BrainScoreModel):
    def __init__(self):
        super().__init__(
            identifier="native-temporal",
            model=None,
            region_layer_map={"IT": "temporal_layer"},
            preprocessors={"vision": lambda stimuli: stimuli},
        )

    def process(self, input_event, multi_modality=False):
        del multi_modality
        stimulus_ids = list(input_event["stimulus_id"].values)
        neuroid_ids = ["temporal_layer.0", "temporal_layer.1"]
        if self._time_bins is None:
            return NeuroidAssembly(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                coords={
                    "stimulus_id": ("presentation", stimulus_ids),
                    "neuroid_id": ("neuroid", neuroid_ids),
                    "layer": ("neuroid", ["temporal_layer"] * 2),
                },
                dims=["presentation", "neuroid"],
            )

        data = np.arange(
            len(stimulus_ids) * len(self._time_bins) * 2, dtype=float
        ).reshape(len(stimulus_ids), len(self._time_bins), 2)
        return NeuroidAssembly(
            data,
            coords={
                "stimulus_id": ("presentation", stimulus_ids),
                "time_bin_start_ms": (
                    "time_bin", [start for start, _ in self._time_bins]
                ),
                "time_bin_end_ms": (
                    "time_bin", [end for _, end in self._time_bins]
                ),
                "neuroid_id": ("neuroid", neuroid_ids),
                "layer": ("neuroid", ["temporal_layer"] * 2),
            },
            dims=["presentation", "time_bin", "neuroid"],
        )


class _TrackingTemporalBrainScoreModel(_TemporalBrainScoreModel):
    def __init__(self):
        super().__init__()
        self.interact_called = False
        self.interact_requested_output_channels = None

    def interact(self, session):
        self.interact_called = True
        self.interact_requested_output_channels = tuple(
            session.requested_output_channels
        )
        return super().interact(session)


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


class _NonNativeEnvironmentSubject(Subject):
    def __init__(self, action_fn):
        self.action_fn = action_fn
        self.processed_steps = []

    @property
    def identifier(self):
        return "non-native-environment"

    @property
    def region_layer_map(self):
        return {}

    @property
    def supported_modalities(self):
        return {"vision", "proprioception"}

    def process(self, input_event):
        self.processed_steps.append(input_event.step_num)
        return self.action_fn(input_event)


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


def test_stimulus_session_exposes_time_bins():
    time_bins = [(70, 170), (170, 270)]
    session = stimulus_session(_stimulus_set(), record="IT",
                               time_bins=time_bins)

    assert session.time_bins == time_bins


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

    assert type(subject).interact is Subject.interact
    scored = score_stimuli(subject, stimuli, record="IT")

    assert subject.recording_target == "IT"
    assert subject.process_input is stimuli
    xr.testing.assert_identical(scored, expected)


def test_score_stimuli_uses_native_interact_and_preserves_metadata_exactly():
    stimuli = _metadata_stimulus_set()

    legacy_extractor = _MetadataPreservingExtractor()
    legacy = _native_neural_model(legacy_extractor)
    legacy.start_recording("IT")
    expected = legacy.process(stimuli)

    native_extractor = _MetadataPreservingExtractor()
    subject = _native_neural_model(
        native_extractor, model_cls=_InteractTrackingBrainScoreModel
    )
    scored = score_stimuli(subject, stimuli, record="IT")

    assert subject.interact_called is True
    assert subject.interact_requested_output_channels == ("neural:IT",)
    assert native_extractor.last_stimuli is stimuli
    assert "modality" not in scored.coords
    xr.testing.assert_identical(scored, expected)


def test_score_stimuli_native_interact_matches_multimodal_process_exactly():
    stimuli = _multimodal_stimulus_set()

    legacy, _, _ = _native_multimodal_model()
    legacy.start_recording("IT")
    expected = legacy.process(stimuli, multi_modality=True)

    subject, activations, text = _native_multimodal_model(
        model_cls=_InteractTrackingBrainScoreModel
    )
    scored = score_stimuli(subject, stimuli, record="IT")

    assert subject.interact_called is True
    assert subject.interact_requested_output_channels == ("neural:IT",)
    assert activations.call_args is not None
    assert text.call_args is not None
    assert set(scored["modality"].values.tolist()) == {"vision", "text"}
    xr.testing.assert_identical(scored, expected)


def test_score_stimuli_native_interact_matches_cross_tower_process_exactly():
    stimuli = _multimodal_stimulus_set()

    legacy, _, _ = _native_cross_tower_model()
    legacy.start_recording("all")
    expected = legacy.process(stimuli, multi_modality=True)

    subject, activations, text = _native_cross_tower_model(
        model_cls=_InteractTrackingBrainScoreModel
    )
    scored = score_stimuli(subject, stimuli, record="all")

    assert subject.interact_called is True
    assert subject.interact_requested_output_channels == ("neural:all",)
    assert activations.call_args["layers"] == ["vision_model.layer.10"]
    assert text.call_args["layers"] == ["text_model.layer.20"]
    assert set(scored["modality"].values.tolist()) == {"vision", "text"}
    xr.testing.assert_identical(scored, expected)


def test_score_stimuli_native_interact_threads_time_bins_exactly():
    stimuli = _metadata_stimulus_set()
    time_bins = [(70, 170), (170, 270)]

    legacy = _TemporalBrainScoreModel()
    legacy.start_recording("IT", time_bins=time_bins)
    expected = legacy.process(stimuli)

    subject = _TrackingTemporalBrainScoreModel()
    scored = score_stimuli(subject, stimuli, record="IT", time_bins=time_bins)

    assert subject.interact_called is True
    assert subject.interact_requested_output_channels == ("neural:IT",)
    assert scored.dims == ("presentation", "time_bin", "neuroid")
    assert list(scored["time_bin_start_ms"].values) == [70, 170]
    assert list(scored["time_bin_end_ms"].values) == [170, 270]
    xr.testing.assert_identical(scored, expected)


def test_score_stimuli_without_time_bins_keeps_2d_output():
    stimuli = _metadata_stimulus_set()

    legacy = _TemporalBrainScoreModel()
    legacy.start_recording("IT")
    expected = legacy.process(stimuli)

    subject = _TrackingTemporalBrainScoreModel()
    scored = score_stimuli(subject, stimuli, record="IT")

    assert subject.interact_called is True
    assert subject.interact_requested_output_channels == ("neural:IT",)
    assert scored.dims == ("presentation", "neuroid")
    assert "time_bin" not in scored.dims
    xr.testing.assert_identical(scored, expected)


def test_event_only_reconstruction_is_input_columns_only():
    events = [
        StreamEvent(
            channel="vision",
            payload="s0.png",
            t_ms=0.0,
            meta={
                "stimulus_id": "s0",
                "stimulus_index": 0,
                "column": "image_path",
                "object_name": "cat",
            },
        ),
        StreamEvent(
            channel="text",
            payload="cat sentence",
            t_ms=0.0,
            meta={
                "stimulus_id": "s0",
                "stimulus_index": 0,
                "column": "sentence",
                "object_name": "cat",
            },
        ),
    ]

    reconstructed = BrainScoreModel._reconstruct_stimulus_set_from_events(events)

    assert list(reconstructed["stimulus_id"].values) == ["s0"]
    assert list(reconstructed["image_path"].values) == ["s0.png"]
    assert list(reconstructed["sentence"].values) == ["cat sentence"]
    assert "object_name" not in reconstructed.columns


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


def _behavior_model(features, model_cls=BrainScoreModel):
    return model_cls(
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

    assert session.requested_output_channels == ("behavior",)

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

    helper_subject = _behavior_model(
        features, model_cls=_InteractTrackingBrainScoreModel
    )
    helper_context = _behavior_context(fitting, scoring)
    scored = score_behavior(helper_subject, helper_context)

    assert helper_subject.interact_called is True
    assert helper_subject.interact_requested_output_channels == ("behavior",)
    assert isinstance(scored, BehavioralAssembly)
    xr.testing.assert_identical(scored, expected)


def _generation_model(calls, model_cls=BrainScoreModel):
    def fake_generate(stimulus_row, instruction, label_set):
        calls.append((stimulus_row["stimulus_id"], instruction, tuple(label_set)))
        return "cat" if stimulus_row["stimulus_id"] in {"s0", "s1", "s2"} else "dog"

    return model_cls(
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
    helper_subject = _generation_model(
        helper_calls, model_cls=_InteractTrackingBrainScoreModel
    )
    helper_context = _generation_context(scoring)
    scored = score_behavior(helper_subject, helper_context)

    assert helper_subject.interact_called is True
    assert helper_subject.interact_requested_output_channels == ("behavior",)
    assert isinstance(scored, BehavioralAssembly)
    xr.testing.assert_identical(scored, expected)
    assert helper_calls == legacy_calls


def test_score_behavior_falls_back_for_non_native_subject():
    scoring = _make_image_stimulus_set(
        ["cat", "dog"], identifier="behavior_fallback"
    )
    expected = BehavioralAssembly(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        coords={
            "stimulus_id": ("presentation", ["s0", "s1"]),
            "choice": ("choice", ["cat", "dog"]),
        },
        dims=["presentation", "choice"],
    )
    subject = _SyntheticSubject(expected)
    context = TaskContext(
        task_type="probabilities",
        label_set=["cat", "dog"],
        metadata={"stimulus_set": scoring},
    )

    scored = score_behavior(subject, context)

    assert type(subject).interact is Subject.interact
    assert subject.process_input is scoring
    xr.testing.assert_identical(scored, expected)


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


class _SyntheticEnvironment:
    def __init__(self, steps):
        self.steps = list(steps)
        self.actions = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        if not self.steps:
            return None
        return self.steps[0]

    def step(self, action):
        self.actions.append(action)
        next_index = len(self.actions)
        if next_index >= len(self.steps):
            return None
        return self.steps[next_index]


def _environment_steps(count):
    steps = []
    for step_num in range(count):
        step = _droid_step(step_num=step_num, instruction="reach the goal")
        step.is_last = step_num == count - 1
        step.is_terminal = step.is_last
        steps.append(step)
    return steps


def _legacy_environment_rollout(subject, environment):
    trajectory = []
    step = environment.reset()
    while step is not None:
        response = subject.process(step)
        trajectory.append(response)
        if step.is_last or step.is_terminal:
            break
        step = environment.step(response.action)
    return trajectory


def test_environment_session_advances_after_motor_emit():
    session = environment_session(_SyntheticEnvironment(_environment_steps(2)))

    assert session.requested_output_channels == ("motor",)

    first_step = session.next_input()
    assert first_step.step_num == 0
    assert session.input_events[0].channel == "observation"
    assert session.next_input() is None

    session.emit(StreamEvent(
        channel="motor",
        payload=np.zeros(7, dtype=np.float64),
        t_ms=0.0,
    ))
    second_step = session.next_input()

    assert second_step.step_num == 1


def test_native_environment_interact_matches_legacy_rollout_exactly():
    session = environment_session(_SyntheticEnvironment(_environment_steps(4)))
    native_model = _native_environment_model(
        _droid_action_fn, model_cls=_InteractTrackingBrainScoreModel
    )

    native_model.interact(session)
    native_trajectory = session.collect("motor")

    legacy_model = _make_environment_model(action_fn=_droid_action_fn)
    legacy_env = _SyntheticEnvironment(_environment_steps(4))
    expected_trajectory = _legacy_environment_rollout(legacy_model, legacy_env)

    assert native_model.interact_called is True
    assert native_model.interact_requested_output_channels == ("motor",)
    assert [event.channel for event in session.emitted] == [
        "motor", "motor", "motor", "motor",
    ]
    assert [event.meta["driver"] for event in session.emitted] == [
        "interact", "interact", "interact", "interact",
    ]
    assert [event.meta["step_num"] for event in session.emitted] == [0, 1, 2, 3]
    assert len(native_trajectory) == len(expected_trajectory)
    for native_response, expected_response in zip(
        native_trajectory, expected_trajectory
    ):
        np.testing.assert_array_equal(
            native_response.action, expected_response.action
        )
        assert native_response.metadata == expected_response.metadata


def test_environment_session_emits_motor_events_for_multistep_rollout():
    session = environment_session(_SyntheticEnvironment(_environment_steps(3)))
    model = _make_environment_model(action_fn=_droid_action_fn)

    from brainscore_core.streaming_helpers import _drive_environment_via_process
    _drive_environment_via_process(model, session)

    assert [event.channel for event in session.emitted] == [
        "motor", "motor", "motor",
    ]
    assert [
        event.payload.metadata["step_num_seen"]
        for event in session.emitted
    ] == [0, 1, 2]
    assert [event.meta["step_num"] for event in session.emitted] == [0, 1, 2]


def test_run_environment_matches_legacy_process_rollout_exactly():
    helper_model = _native_environment_model(
        _droid_action_fn, model_cls=_InteractTrackingBrainScoreModel
    )
    helper_env = _SyntheticEnvironment(_environment_steps(4))
    helper_trajectory = run_environment(helper_model, helper_env)

    legacy_model = _make_environment_model(action_fn=_droid_action_fn)
    legacy_env = _SyntheticEnvironment(_environment_steps(4))
    expected_trajectory = _legacy_environment_rollout(legacy_model, legacy_env)

    assert len(helper_trajectory) == len(expected_trajectory)
    for helper_response, expected_response in zip(
        helper_trajectory, expected_trajectory
    ):
        np.testing.assert_array_equal(
            helper_response.action, expected_response.action
        )
        assert helper_response.metadata == expected_response.metadata
    assert len(helper_env.actions) == len(legacy_env.actions)
    for helper_action, expected_action in zip(helper_env.actions, legacy_env.actions):
        np.testing.assert_array_equal(helper_action, expected_action)
    assert helper_model.interact_called is True
    assert helper_model.interact_requested_output_channels == ("motor",)


def test_run_environment_falls_back_for_non_native_subjects():
    subject = _NonNativeEnvironmentSubject(_droid_action_fn)
    env = _SyntheticEnvironment(_environment_steps(3))

    trajectory = run_environment(subject, env)

    assert type(subject).interact is Subject.interact
    assert subject.processed_steps == [0, 1, 2]
    assert [response.metadata["step_num_seen"] for response in trajectory] == [
        0, 1, 2,
    ]


def test_run_environment_terminates_cleanly_for_zero_and_one_step_envs():
    model = _native_environment_model(
        _droid_action_fn, model_cls=_InteractTrackingBrainScoreModel
    )
    zero_env = _SyntheticEnvironment([])

    assert run_environment(model, zero_env) == []
    assert zero_env.actions == []
    assert model.interact_called is True
    assert model.interact_requested_output_channels == ("motor",)

    model = _native_environment_model(
        _droid_action_fn, model_cls=_InteractTrackingBrainScoreModel
    )
    one_env = _SyntheticEnvironment(_environment_steps(1))
    trajectory = run_environment(model, one_env)

    assert len(trajectory) == 1
    assert trajectory[0].metadata["step_num_seen"] == 0
    assert one_env.actions == []
    assert model.interact_called is True
    assert model.interact_requested_output_channels == ("motor",)


def _ablation_state_change():
    return StateChange(
        kind="ablation",
        target=Selection(layer="blocks.10", indices=[1, 2, 3]),
        perturbation=Perturbation(kind="zero"),
    )


def test_state_change_session_emits_lesion_event_with_address():
    session = state_change_session(_ablation_state_change())
    event = session.next_input()

    assert session.requested_output_channels == ("perturbation",)
    assert event.channel == "lesion:blocks.10[1:4]"
    assert isinstance(event.payload, StateChange)
    assert event.meta == {"kind": "ablation"}
    assert session.next_input() is None


def test_state_change_session_routes_pharmacological_event_to_registry_channel():
    state_change = StateChange(
        kind="pharmacological",
        metadata={"address": "ketamine"},
        perturbation=Perturbation(kind="dose", scale=0.1),
    )
    session = state_change_session(state_change)
    event = session.next_input()

    assert event.channel == "pharmacological:ketamine"
    assert io_catalog.validate(
        event.channel, event.payload, direction=io_catalog.INPUT
    ) == []


def test_state_change_session_routes_lesion_kind_to_lesion_channel():
    state_change = StateChange(
        kind="lesion",
        target=Selection(layer="blocks.10"),
        perturbation=Perturbation(kind="zero"),
    )
    session = state_change_session(state_change)
    event = session.next_input()

    assert event.channel == "lesion:blocks.10"


def test_state_change_reset_uses_metadata_family_and_address_when_present():
    reset = StateChange(
        kind="reset",
        handle_id="stimulation-handle",
        metadata={
            "family": "stimulation",
            "address": "blocks.10",
        },
    )
    session = state_change_session(reset)
    event = session.next_input()

    assert event.channel == "stimulation:blocks.10"
    assert event.meta["reset"] == "stimulation-handle"


def test_apply_state_change_produces_handle_and_ack_event():
    fn, state = _counting_state_change_fn()
    model = _make_state_change_model(state_change_fn=fn)
    state_change = _ablation_state_change()

    session = state_change_session(state_change)
    _ = session.next_input()
    from brainscore_core.streaming_helpers import _drive_state_change_via_process
    _drive_state_change_via_process(model, session, state_change)

    applied = session.collect("perturbation")

    assert isinstance(applied, PerturbationApplied)
    assert applied.handle_id in state["installed"]
    assert session.emitted[0].meta["handle_id"] == applied.handle_id
    assert session.emitted[0].payload is applied


def test_apply_state_change_matches_legacy_process_result():
    fn_legacy, _ = _counting_state_change_fn()
    legacy = _make_state_change_model(state_change_fn=fn_legacy)
    legacy_result = legacy.process(_ablation_state_change())

    fn_helper, _ = _counting_state_change_fn()
    helper_model = _native_state_change_model(
        fn_helper, model_cls=_InteractTrackingBrainScoreModel
    )
    helper_result = apply_state_change(helper_model, _ablation_state_change())

    assert helper_model.interact_called is True
    assert helper_model.interact_requested_output_channels == ("perturbation",)
    assert isinstance(helper_result, PerturbationApplied)
    assert helper_result.handle_id == legacy_result.handle_id
    assert helper_result.target == legacy_result.target
    assert helper_result.perturbation == legacy_result.perturbation
    assert helper_result.applied_at == legacy_result.applied_at


def test_native_state_change_reset_restores_baseline():
    model_state = {"output": 1.0}

    def state_change_fn(state_change):
        saved = model_state["output"]
        model_state["output"] = 0.0
        applied = PerturbationApplied(
            handle_id="ablation-handle",
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup():
            model_state["output"] = saved

        return applied, cleanup

    model = _native_state_change_model(
        state_change_fn, model_cls=_InteractTrackingBrainScoreModel
    )
    applied = apply_state_change(model, _ablation_state_change())
    assert model_state["output"] == 0.0
    assert model.interact_requested_output_channels == ("perturbation",)

    reset = StateChange(kind="reset", handle_id=applied.handle_id)
    reset_session = state_change_session(reset)
    reset_event = reset_session.next_input()
    model.interact_called = False
    reset_result = apply_state_change(model, reset)

    assert model.interact_called is True
    assert model.interact_requested_output_channels == ("perturbation",)
    assert reset_event.channel == "lesion:ablation-handle"
    assert reset_event.meta["reset"] == applied.handle_id
    assert reset_result == applied.handle_id
    assert model_state["output"] == 1.0


def test_apply_state_change_falls_back_for_non_native_subject():
    applied = PerturbationApplied(
        handle_id="fallback-handle",
        target=Selection(layer="blocks.10"),
        perturbation=Perturbation(kind="zero"),
    )
    subject = _SyntheticSubject(applied)
    state_change = _ablation_state_change()

    result = apply_state_change(subject, state_change)

    assert type(subject).interact is Subject.interact
    assert subject.process_input is state_change
    assert result is applied
