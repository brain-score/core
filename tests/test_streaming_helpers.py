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
    helper_model = _make_environment_model(action_fn=_droid_action_fn)
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


def test_run_environment_terminates_cleanly_for_zero_and_one_step_envs():
    model = _make_environment_model(action_fn=_droid_action_fn)
    zero_env = _SyntheticEnvironment([])

    assert run_environment(model, zero_env) == []
    assert zero_env.actions == []

    one_env = _SyntheticEnvironment(_environment_steps(1))
    trajectory = run_environment(model, one_env)

    assert len(trajectory) == 1
    assert trajectory[0].metadata["step_num_seen"] == 0
    assert one_env.actions == []


def _ablation_state_change():
    return StateChange(
        kind="ablation",
        target=Selection(layer="blocks.10", indices=[1, 2, 3]),
        perturbation=Perturbation(kind="zero"),
    )


def test_state_change_session_emits_lesion_event_with_address():
    session = state_change_session(_ablation_state_change())
    event = session.next_input()

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
    helper_model = _make_state_change_model(state_change_fn=fn_helper)
    helper_result = apply_state_change(helper_model, _ablation_state_change())

    assert isinstance(helper_result, PerturbationApplied)
    assert helper_result.handle_id == legacy_result.handle_id
    assert helper_result.target == legacy_result.target
    assert helper_result.perturbation == legacy_result.perturbation
    assert helper_result.applied_at == legacy_result.applied_at


def test_state_change_reset_event_restores_baseline():
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

    model = _make_state_change_model(state_change_fn=state_change_fn)
    applied = apply_state_change(model, _ablation_state_change())
    assert model_state["output"] == 0.0

    reset = StateChange(kind="reset", handle_id=applied.handle_id)
    reset_session = state_change_session(reset)
    reset_event = reset_session.next_input()
    reset_result = apply_state_change(model, reset)

    assert reset_event.channel == "lesion:ablation-handle"
    assert reset_event.meta["reset"] == applied.handle_id
    assert reset_result == applied.handle_id
    assert model_state["output"] == 1.0
