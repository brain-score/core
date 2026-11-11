from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore_core import io_catalog
from brainscore_core.model_interface import (
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
    UnifiedModel,
    TaskContext,
)
from brainscore_core.streaming import NamedStreams
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    apply_state_change,
    behavior_session,
    environment_session,
    behavioral_response,
    neural_response,
    state_change_session,
    stimulus_session,
    run_environment,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _make_image_stimulus_set(labels, identifier='test_stim'):
    n = len(labels)
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_file_name': [f'/tmp/s{i}.png' for i in range(n)],
        'image_label': labels,
    }))
    stimuli.identifier = identifier
    return stimuli


def _droid_row(step_num: int = 0, instruction: str = "pick up the red block"):
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    return {
        'vision:image': NamedStreams({'exterior_1': frame, 'exterior_2': frame, 'wrist': frame}),
        'proprioception': {'joint_position': np.zeros(7), 'gripper_position': np.zeros(1)},
        'instruction': instruction,
    }


def _droid_policy(row):
    return {'motor': np.zeros(7, dtype=np.float64)}



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


class _SyntheticSubject(UnifiedModel):
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


class _NonNativeEnvironmentSubject(UnifiedModel):
    def __init__(self, action_fn):
        self.action_fn = action_fn
        self.processed_rows = []

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
        self.processed_rows.append(input_event)
        return self.action_fn(input_event)


def test_stimulus_session_emits_input_events_per_stimulus_column():
    session = stimulus_session(_stimulus_set(), record="IT").declare_inputs(
        {'control', 'vision', 'text'})

    events = []
    while True:
        event = session.next_input()
        if event is None:
            break
        events.append(event)

    # Each stimulus row is one independent trial: begin, its columns, end.
    assert [event.channel for event in events] == [
        "control", "vision", "text", "control", "control", "vision", "text", "control",
    ]
    assert [event.payload for event in events if event.channel == "control"] == [
        "begin", "end", "begin", "end",
    ]
    events = [event for event in events if event.channel != "control"]
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
    session.respond(StreamEvent(
        channel="neural:IT",
        payload=np.array([1.0, 2.0]),
        t_ms=0.0,
        meta={"stimulus_id": "s0"},
    ))
    session.respond(StreamEvent(
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


@pytest.mark.parametrize('regions', ['IT', ['V4', 'IT']])
@pytest.mark.parametrize('kind', ['batch', 'streaming', 'windowed'])
def test_collect_defaults_to_all_requested_neural_channels(regions, kind):
    from brainscore_core.streaming import Session
    from brainscore_core.streaming_helpers import (
        stimulus_session, streaming_stimulus_session, windowed_stream,
    )
    if kind == 'windowed':
        channels = ([f'neural:{regions}'] if isinstance(regions, str)
                    else [f'neural:{region}' for region in regions])
        session = Session(windowed_stream(iter([]), fps=30, window_ms=1000), channels,
                          streaming=True)
    else:
        builder = stimulus_session if kind == 'batch' else streaming_stimulus_session
        session = builder(_stimulus_set(), record=regions)
    expected = {}
    for channel in session.requested_output_channels:
        expected[channel] = _assembly()
        session.respond(StreamEvent(channel=channel, payload=expected[channel], t_ms=0))
    collected = session.collect()
    if isinstance(regions, str):
        xr.testing.assert_identical(collected, expected['neural:IT'])
    else:
        assert collected.keys() == expected.keys()
        for channel, assembly in collected.items():
            xr.testing.assert_identical(assembly, session.collect(channel=channel))


def test_neural_response_matches_process_assembly_exactly():
    expected = _assembly()
    subject = _SyntheticSubject(expected)
    stimuli = _stimulus_set()

    assert type(subject).participate is UnifiedModel.participate
    scored = neural_response(subject, stimuli, record="IT")

    assert subject.recording_target == "IT"
    assert subject.process_input is stimuli
    xr.testing.assert_identical(scored, expected)


def test_collect_returns_emitted_assembly_without_repackaging():
    expected = _assembly()
    session = stimulus_session(_stimulus_set(), record="IT")
    session.respond(StreamEvent(
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
    session = behavior_session(_behavior_context(fitting, scoring)).declare_inputs(
        {'control', 'instruction', 'vision'})

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

    # Fitting rows are not trials; each scoring row is one begin ... end.
    assert [event.payload for event in events if event.channel == "control"] == [
        "begin", "end", "begin", "end",
    ]
    stimulus_events = [event for event in events[1:] if event.channel != "control"]
    assert [event.channel for event in stimulus_events] == [
        "vision", "vision", "vision", "vision",
    ]
    assert [event.meta["role"] for event in stimulus_events] == [
        "fitting", "fitting", "scoring", "scoring",
    ]
    assert [event.meta["stimulus_id"] for event in stimulus_events] == [
        "s0", "s1", "s0", "s1",
    ]


def test_behavioral_response_falls_back_for_non_native_subject():
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

    scored = behavioral_response(subject, context)

    assert type(subject).participate is UnifiedModel.participate
    assert subject.process_input is scoring
    xr.testing.assert_identical(scored, expected)


def test_behavior_collect_packages_raw_label_events():
    session = behavior_session(TaskContext(task_type="probabilities"))
    session.respond(StreamEvent(
        channel="behavior",
        payload="cat",
        t_ms=0.0,
        meta={"stimulus_id": "s0", "label_set": ["cat", "dog"]},
    ))
    session.respond(StreamEvent(
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
    session.respond(StreamEvent(
        channel="behavior",
        payload=np.array([0.25, 0.75]),
        t_ms=0.0,
        meta={"stimulus_id": "s0", "label_set": ["cat", "dog"]},
    ))
    session.respond(StreamEvent(
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
    def __init__(self, count):
        self.count = count
        self.actions = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return _droid_row(0, "reach the goal")

    def step(self, outputs):
        self.actions.append(outputs['motor'])
        done = len(self.actions) >= self.count
        return _droid_row(len(self.actions), "reach the goal"), 'terminal' if done else None


def test_environment_session_advances_after_motor_emit():
    session = environment_session(_SyntheticEnvironment(2)).declare_inputs(
        {'control', 'vision:image', 'proprioception', 'instruction', 'reward', 'observation'})

    assert session.requested_output_channels == ("motor",)

    events = []
    while (event := session.next_input()).channel != 'control' or event.payload != 'respond':
        events.append(event)
    assert [e.channel for e in events[:2]] == ['control', 'vision:image']
    assert {e.meta['camera'] for e in events if e.channel == 'vision:image'} == {
        'exterior_1', 'exterior_2', 'wrist'}
    with pytest.raises(RuntimeError, match="Emit an action"):
        session.next_input()

    session.respond(StreamEvent(
        channel="motor",
        payload=np.zeros(7, dtype=np.float64),
        t_ms=0.0,
    ))
    assert session.next_input().channel == 'vision:image'
    assert session.source.input_events[-1].meta['step_num'] == 1


def test_run_environment_falls_back_for_non_native_subjects():
    subject = _NonNativeEnvironmentSubject(_droid_policy)
    env = _SyntheticEnvironment(3)

    trajectory = run_environment(subject, env)

    assert type(subject).participate is UnifiedModel.participate
    assert len(subject.processed_rows) == 3 and len(trajectory) == 3
    assert set(subject.processed_rows[0]['vision:image']) == {'exterior_1', 'exterior_2', 'wrist'}
    assert subject.processed_rows[1]['instruction'] == "reach the goal"


def test_run_environment_rejects_objects_without_participate():
    subject = SimpleNamespace(process=lambda step: pytest.fail("process() reached"))
    with pytest.raises(TypeError, match="participate"):
        run_environment(subject, _SyntheticEnvironment(2))


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


def test_apply_state_change_falls_back_for_non_native_subject():
    applied = PerturbationApplied(
        handle_id="fallback-handle",
        target=Selection(layer="blocks.10"),
        perturbation=Perturbation(kind="zero"),
    )
    subject = _SyntheticSubject(applied)
    state_change = _ablation_state_change()

    result = apply_state_change(subject, state_change)

    assert type(subject).participate is UnifiedModel.participate
    assert subject.process_input is state_change
    assert result is applied


class TestRecordAcceptsSeveralRegions:
    """`record` mirrors start_recording: a region name OR a sequence of them.

    Regression for a silent failure: the session constructors formatted `record`
    straight into the channel string, so a list produced ONE channel literally
    named "neural:['V4', 'IT']". Nothing raised; the channel simply matched no
    region, and callers got an empty or wrong result.
    """

    @staticmethod
    def _stimuli():
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        return StimulusSet(pd.DataFrame([
            {'stimulus_id': 'a', 'image_file_name': 'a.png'}]))

    def test_list_record_yields_one_channel_per_region(self):
        from brainscore_core.streaming_helpers import (
            stimulus_session, requested_output_channels)
        session = stimulus_session(self._stimuli(), record=['V4', 'IT'])
        assert requested_output_channels(session) == ['neural:V4', 'neural:IT']

    def test_single_region_string_unchanged(self):
        from brainscore_core.streaming_helpers import (
            stimulus_session, requested_output_channels)
        session = stimulus_session(self._stimuli(), record='IT')
        assert requested_output_channels(session) == ['neural:IT']

    def test_streaming_session_also_splits(self):
        from brainscore_core.streaming_helpers import streaming_stimulus_session
        session = streaming_stimulus_session(self._stimuli(), record=['V4', 'IT'])
        assert list(session.requested_output_channels) == ['neural:V4', 'neural:IT']

    def test_empty_record_raises(self):
        import pytest
        from brainscore_core.streaming_helpers import stimulus_session
        with pytest.raises(ValueError, match='at least one region'):
            stimulus_session(self._stimuli(), record=[])

    def test_non_string_region_raises(self):
        import pytest
        from brainscore_core.streaming_helpers import stimulus_session
        with pytest.raises(TypeError, match='region name'):
            stimulus_session(self._stimuli(), record=[1, 2])

    def test_neural_response_rejects_several_regions(self):
        """It returns one assembly, so it must not silently pick a region."""
        import pytest
        from brainscore_core.streaming_helpers import neural_response
        with pytest.raises(TypeError, match='single region'):
            neural_response(object(), self._stimuli(), record=['V4', 'IT'])


class _TypedSubject:
    """A plain object with the typed methods the public drivers call.

    No base class: the drivers rely on these calls only, never on discovery.
    """

    def __init__(self, output=None):
        self.calls = []
        self.output = output

    def start_recording(self, target, time_bins=None):
        self.calls.append(('start_recording', target, time_bins))

    def start_task(self, task_context):
        self.calls.append(('start_task', task_context))

    def process(self, stimuli, **kwargs):
        self.calls.append(('process', stimuli, kwargs))
        return self.output


class TestPublicSessionHelpers:
    """The promoted session helpers keep their behavior under the public names."""

    def test_private_names_are_the_same_functions(self):
        from brainscore_core import streaming_helpers as helpers
        assert helpers._drain_stream_events is helpers.drain_session_inputs
        assert helpers._stimuli_from_stream_session is helpers.session_stimuli
        assert helpers._drive_neural_session_via_process is helpers.drive_neural_session
        assert helpers._drive_behavior_session_via_process is helpers.drive_behavior_session

    def test_drain_returns_every_input_in_order_then_none(self):
        from brainscore_core.streaming import Session
        from brainscore_core.streaming_helpers import drain_session_inputs
        events = [StreamEvent('vision', f'p{index}', float(index), {'i': index})
                  for index in range(3)]
        session = Session(events, ('neural:IT',))
        assert drain_session_inputs(session) == events
        assert session.next_input() is None
        assert drain_session_inputs(session) == []

    def test_drain_rejects_non_events(self):
        from brainscore_core.streaming_helpers import drain_session_inputs

        class _Session:
            def next_input(self):
                return 'not an event'
        with pytest.raises(TypeError, match='StreamEvent'):
            drain_session_inputs(_Session())

    def test_session_stimuli_returns_the_session_table_itself(self):
        from brainscore_core.streaming_helpers import (
            drain_session_inputs, session_stimuli)
        stimuli = _stimulus_set()
        session = stimulus_session(stimuli, record='IT')
        assert session_stimuli(session, drain_session_inputs(session)) is stimuli

    def test_session_stimuli_rebuilds_inputs_without_a_table(self):
        from brainscore_core.streaming_helpers import session_stimuli
        events = [StreamEvent('vision', 'a.png', 0.0,
                              {'stimulus_id': 's0', 'column': 'image_path'})]

        class _Session:
            pass
        rebuilt = session_stimuli(_Session(), events)
        assert list(rebuilt['stimulus_id']) == ['s0']
        assert list(rebuilt['image_path']) == ['a.png']

    def test_drive_neural_session_one_process_call_with_time_bins(self):
        from brainscore_core.streaming_helpers import drive_neural_session
        stimuli = _stimulus_set()
        output = _assembly()
        subject = _TypedSubject(output)
        session = stimulus_session(stimuli, record='IT', time_bins=[(70, 170)])
        drive_neural_session(subject, session)
        assert subject.calls == [('start_recording', 'IT', [(70, 170)]),
                                 ('process', stimuli, {})]
        event, = session.emitted
        assert event.channel == 'neural:IT'
        assert event.payload is output
        assert event.meta == {'driver': 'interact', 'record': 'IT'}

    def test_drive_neural_session_refuses_variance_before_compute(self):
        from brainscore_core.streaming_helpers import drive_neural_session
        subject = _TypedSubject()
        session = stimulus_session(_stimulus_set(), record='IT')
        session.settings = {'require_variance': True, 'number_of_trials': 3}
        with pytest.raises(NotImplementedError, match='number_of_trials=3'):
            drive_neural_session(subject, session)
        assert subject.calls == []
        assert session.emitted == []

    def test_drive_behavior_session_starts_task_once_and_scores_the_task_table(self):
        from brainscore_core.streaming_helpers import drive_behavior_session
        scored = _stimulus_set()
        context = TaskContext(task_type='probabilities', label_set=['a', 'b'],
                              fitting_stimuli=_stimulus_set(),
                              metadata={'stimulus_set': scored})
        output = BehavioralAssembly(
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            coords={'stimulus_id': ('presentation', ['s0', 's1']),
                    'choice': ('choice', ['a', 'b'])},
            dims=['presentation', 'choice'])
        subject = _TypedSubject(output)
        session = behavior_session(context)
        drive_behavior_session(subject, session)
        assert subject.calls == [('start_task', context), ('process', scored, {})]
        event, = session.emitted
        assert event.channel == 'behavior'
        assert event.payload is output
        assert event.meta == {'driver': 'interact', 'task_type': 'probabilities'}

    def test_drive_behavior_session_refuses_variance_before_compute(self):
        from brainscore_core.streaming_helpers import drive_behavior_session
        context = TaskContext(task_type='probabilities',
                              metadata={'stimulus_set': _stimulus_set()})
        subject = _TypedSubject()
        session = behavior_session(context)
        session.settings = {'require_variance': True}
        with pytest.raises(NotImplementedError, match='trial-to-trial variance'):
            drive_behavior_session(subject, session)
        assert subject.calls == []
