"""Environment preflight, episode boundaries, clocks and execution receipts."""
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore_core.environment import ActionSpec, ArraySpec, DiscreteSpec, MappingSpec, TextSpec
from brainscore_core.events import EnvironmentResponse, EnvironmentStep
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import EnvironmentSession, run_environment


class Environment:
    def __init__(self, *, error=None, terminal_on_reset=False):
        self.error = error
        self.terminal_on_reset = terminal_on_reset
        self.actions = []
        self.closed = 0
        self.reset_args = None
        self.t_ms = None

    def observation_spec(self):
        return MappingSpec({'image': ArraySpec((2, 3, 3), 'uint8'), 'task': TextSpec()})

    def action_spec(self):
        return DiscreteSpec(3)

    def observation(self, *, last=False):
        return EnvironmentStep(
            observation={'image': np.zeros((2, 3, 3), dtype=np.uint8), 'task': 'move'},
            step_num=len(self.actions), is_last=last, is_terminal=last,
            context={'t_ms': self.t_ms, 'time_source': 'simulation', 'time_inferred': False},
        )

    def reset(self, *, seed=None, options=None):
        self.reset_args = (seed, options)
        return self.observation(last=self.terminal_on_reset)

    def step(self, action):
        self.actions.append(action)
        if self.error:
            raise self.error
        return self.observation(last=True)

    def close(self):
        self.closed += 1


def test_terminal_observation_is_retained_without_a_policy_call():
    env = Environment()
    calls = []
    subject = SimpleNamespace(process=lambda step: calls.append(step) or EnvironmentResponse(1))
    session = EnvironmentSession(env, seed=11, options={'task': 2}, require_specs=True)
    from brainscore_core.streaming_helpers import _drive_environment_via_process
    with session:
        _drive_environment_via_process(subject, session)
    assert len(calls) == len(env.actions) == len(session.emitted) == 1
    assert len(session.input_events) == 2
    assert session.input_events[-1].payload.is_terminal
    assert env.reset_args == (11, {'task': 2}) and env.closed == 1
    assert session.input_events[0].payload.context['environment_session']['reset']['seed'] == 11
    assert [e.meta['action_status'] for e in session.action_events] == ['proposed', 'applied']
    with pytest.raises(RuntimeError, match='No observation'):
        session.emit(StreamEvent('motor', 1, None))


def test_terminal_reset_does_not_call_policy():
    env = Environment(terminal_on_reset=True)
    subject = SimpleNamespace(process=lambda step: pytest.fail('Terminal input reached policy'))
    assert run_environment(subject, env) == []
    assert env.actions == [] and env.closed == 1


@pytest.mark.parametrize('action', [-1, 3, 1.5, True, [0, 1], [[0]]])
def test_invalid_actions_are_rejected_before_the_environment(action):
    env = Environment()
    session = EnvironmentSession(env)
    session.next_input()
    with pytest.raises(ValueError):
        session.emit(StreamEvent('motor', action, None))
    assert env.actions == session.emitted == []
    assert session.action_events[-1].meta['action_status'] == 'rejected'
    assert session.next_input() is None


@pytest.mark.parametrize('error', [RuntimeError('lost connection'), KeyboardInterrupt()])
def test_failed_step_is_unknown_execution_never_success(error):
    env = Environment(error=error)
    with EnvironmentSession(env) as session:
        session.next_input()
        with pytest.raises(type(error)):
            session.emit(StreamEvent('motor', 1, None))
        assert session.emitted == []
        assert session.action_events[-1].meta['action_status'] == 'execution_unknown'
        assert session.next_input() is None
    assert env.closed == 1


def test_bad_observation_fails_before_inference_and_closes():
    env = Environment()
    env.observation = lambda **kwargs: EnvironmentStep(observation={'image': [1], 'task': 'move'})
    subject = SimpleNamespace(process=lambda step: pytest.fail('Bad observation reached policy'))
    with pytest.raises(ValueError, match='image'):
        run_environment(subject, env)
    assert env.actions == [] and env.closed == 1


def test_missing_specs_or_unsupported_seed_fail_before_reset():
    calls = []
    env = SimpleNamespace(reset=lambda: calls.append('reset'), step=lambda action: None)
    with pytest.raises(TypeError, match='observation_spec'):
        EnvironmentSession(env, require_specs=True)
    with pytest.raises(TypeError, match='seed'):
        EnvironmentSession(env, seed=1)
    assert calls == []


def test_unknown_time_stays_unknown_and_simulated_time_is_preserved():
    env = Environment()
    session = EnvironmentSession(env)
    session.next_input()
    assert session.input_events[0].t_ms is None
    assert session.input_events[0].meta['time_source'] == 'step_index'
    env = Environment()
    env.t_ms = 250.
    session = EnvironmentSession(env)
    session.next_input()
    session.emit(StreamEvent('motor', 1, 999))
    assert session.emitted[0].t_ms == 250.
    assert session.emitted[0].meta['time_source'] == 'simulation'
    env.t_ms = 0.
    session._pending_step = env.observation(last=True)
    with pytest.raises(ValueError, match='backwards'):
        session.next_input()


@pytest.mark.parametrize('t_ms', [float('nan'), float('inf'), True])
def test_invalid_time_is_rejected(t_ms):
    env = Environment()
    env.t_ms = t_ms
    with pytest.raises(ValueError, match='finite milliseconds'):
        EnvironmentSession(env).next_input()


def test_environment_mutation_cannot_rewrite_submitted_command():
    class Mutating(Environment):
        def action_spec(self):
            return ArraySpec((2,), 'float64')

        def step(self, action):
            action[:] = 99
            return None
    original = np.array([1., 2.])
    with EnvironmentSession(Mutating()) as session:
        session.next_input()
        session.emit(StreamEvent('motor', EnvironmentResponse(original), None))
    np.testing.assert_array_equal(original, [1, 2])
    np.testing.assert_array_equal(session.emitted[0].payload.action, [1, 2])


def test_specs_describe_nested_values_and_do_not_cast_or_clip():
    spec = MappingSpec({'camera': ArraySpec((None, None, 3), 'uint8'), 'task': TextSpec()})
    value = {'camera': np.zeros((4, 5, 3), dtype=np.uint8), 'task': 'go'}
    result = spec.validate(value)
    value['camera'][:] = 255
    assert not result['camera'].any()
    assert spec.describe()['fields']['camera']['shape'] == [None, None, 3]
    with pytest.raises(ValueError, match='dtype'):
        spec.validate({'camera': np.zeros((4, 5, 3)), 'task': 'go'})
    with pytest.raises(ValueError, match='extra'):
        spec.validate({**value, 'answer': 1})
    with pytest.raises(ValueError, match='bounds'):
        ArraySpec((2,), lower=0, upper=1).validate([0., 2.])
    spec = ActionSpec(('joint',), ('rad/s',), 'joint', 50, (-1,), (1,))
    assert spec.describe()['units'] == ('rad/s',)


def test_recording_failure_before_execution_prevents_an_action():
    env = Environment()
    with EnvironmentSession(env) as session:
        session.next_input()
        def unavailable(event):
            raise OSError('record storage unavailable')
        with session.observe_actions(unavailable):
            with pytest.raises(OSError, match='storage unavailable'):
                session.emit(StreamEvent('motor', 1, None))
        assert not session.complete
        assert env.actions == []
        assert session.next_input() is None
    assert env.closed == 1


def test_proposal_is_observed_before_environment_step():
    env = Environment()
    ordering = []
    original_step = env.step
    def step(action):
        ordering.append('environment_step')
        return original_step(action)
    env.step = step
    with EnvironmentSession(env) as session:
        session.next_input()
        with session.observe_actions(lambda event: ordering.append(event.meta['action_status'])):
            session.emit(StreamEvent('motor', 1, None))
    assert ordering == ['proposed', 'environment_step', 'applied']


def test_custom_action_validator_cannot_bypass_declared_bounds():
    env = Environment()
    session = EnvironmentSession(env, action_validator=lambda value: 99)
    session.next_input()
    with pytest.raises(ValueError, match='bounds'):
        session.emit(StreamEvent('motor', 1, None))
    assert env.actions == []


def test_helper_closes_environment_if_preflight_fails():
    closed = []
    env = SimpleNamespace(
        reset=lambda: None, step=lambda action: None,
        close=lambda: closed.append(True),
    )
    with pytest.raises(TypeError, match='seed'):
        run_environment(SimpleNamespace(), env, seed=2)
    assert closed == [True]


def test_native_session_rejects_an_unscheduled_chunk():
    env = Environment()
    session = EnvironmentSession(env)
    session.next_input()
    with pytest.raises(ValueError, match='Schedule policy chunks'):
        session.emit(StreamEvent('motor', EnvironmentResponse(
            np.ones((2, 3)), metadata={'action_kind': 'chunk'},
        ), None))
    assert env.actions == []
