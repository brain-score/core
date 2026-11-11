"""Environment preflight, episode boundaries, clocks and execution receipts."""
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore_core.environment import ActionSpec, ArraySpec, DiscreteSpec, MappingSpec, TextSpec
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import (
    EnvironmentSource, _drive_environment_via_process, run_environment,
)


ENV_INPUTS = ('control', 'observation', 'proprioception', 'reward', 'instruction')


def make(env, **kwargs):
    """A session over ``env``; returns (session, source)."""
    source = EnvironmentSource(env, **kwargs)
    return Session(source, ('motor',)).declare_inputs(ENV_INPUTS), source


def advance(session):
    """Read input events up to the next respond or end; returns them."""
    events = []
    while (event := session.next_input()) is not None:
        events.append(event)
        if event.channel == 'control' and event.payload in ('respond', 'end'):
            break
    return events


def motor(payload, t_ms=None, **meta):
    return StreamEvent('motor', payload, t_ms, meta)


class Environment:
    def __init__(self, *, error=None, reset_reason=None):
        self.error = error
        self.reset_reason = reset_reason
        self.actions = []
        self.closed = 0
        self.reset_args = None
        self.t_ms = None

    def observation_spec(self):
        return MappingSpec({'image': ArraySpec((2, 3, 3), 'uint8'), 'task': TextSpec()})

    def action_spec(self):
        return DiscreteSpec(3)

    def observation(self):
        return {'observation': {'image': np.zeros((2, 3, 3), dtype=np.uint8), 'task': 'move'}}

    def clock(self):
        return {'t_ms': self.t_ms, 'time_source': 'simulation', 'time_inferred': False}

    def reset(self, *, seed=None, options=None):
        self.reset_args = (seed, options)
        if self.reset_reason:
            return self.observation(), self.reset_reason
        return self.observation()

    def step(self, outputs):
        self.actions.append(outputs['motor'])
        if self.error:
            raise self.error
        return self.observation(), 'terminal'

    def close(self):
        self.closed += 1


def test_terminal_observation_is_retained_without_a_policy_call():
    env = Environment()
    calls = []
    subject = SimpleNamespace(process=lambda row: calls.append(row) or {'motor': 1})
    session, source = make(env, seed=11, options={'task': 2}, require_specs=True)
    with session:
        _drive_environment_via_process(subject, session)
    assert len(calls) == len(env.actions) == len(session.emitted) == 1
    assert len(source.input_events) == 2
    assert env.reset_args == (11, {'task': 2}) and env.closed == 1
    assert session.complete
    assert [e.meta['action_status'] for e in source.action_events] == ['proposed', 'applied']
    with pytest.raises(RuntimeError, match='No observation'):
        session.respond(motor(1))


@pytest.mark.parametrize('reason, complete', [('terminal', True), ('truncated', True), ('failure', False)])
def test_reset_with_reason_ends_without_a_policy_call(reason, complete):
    env = Environment(reset_reason=reason)
    calls = []
    subject = SimpleNamespace(process=lambda row: calls.append(row) or {'motor': 1})
    session, source = make(env)
    with session:
        _drive_environment_via_process(subject, session)
    assert calls == [] and env.actions == [] and env.closed == 1
    assert session.complete is complete and session.emitted == []
    assert [(e.channel, e.payload) for e in source.input_events] == [
        ('observation', source.input_events[0].payload)]


def test_reset_with_reason_emits_begin_row_end_and_no_respond():
    class Empty(Environment):
        def reset(self, **kwargs):
            return {}, 'terminal'
    session, source = make(Environment(reset_reason='terminal'))
    events = advance(session)
    assert [e.payload for e in events if e.channel == 'control'] == ['begin', 'end']
    assert events[-1].meta['reason'] == 'terminal' and session.next_input() is None
    session, source = make(Empty())
    events = advance(session)
    assert [e.payload for e in events] == ['begin', 'end'] and session.complete


def test_reset_with_reason_validates_the_reason():
    class Bad(Environment):
        def reset(self, **kwargs):
            return self.observation(), 'done'
    session, _ = make(Bad())
    with pytest.raises(ValueError, match='end reason'):
        session.next_input()


def test_zero_history_reset_with_reason_collects_empty():
    session, source = make(Environment(reset_reason='terminal'), history_limit=0)
    assert advance(session)[-1].payload == 'end' and session.next_input() is None
    assert session.complete and session.emitted == [] and source.input_events == []


def test_source_emits_begin_rows_respond_and_end_in_order():
    env = Environment()
    env.t_ms = 5.
    session, source = make(env, seed=3)
    events = advance(session)
    assert [(e.channel, e.payload) for e in events] == [
        ('control', 'begin'), ('observation', events[1].payload), ('control', 'respond')]
    assert events[0].meta['environment_session']['reset'] == {'seed': 3}
    assert events[1].t_ms == 5. and events[1].meta['step_num'] == 0
    assert not session.complete
    session.respond(motor(1))
    events = advance(session)
    assert [e.channel for e in events] == ['observation', 'control']
    assert events[-1].payload == 'end' and events[-1].meta['reason'] == 'terminal'
    assert session.complete and session.next_input() is None


def test_truncated_ends_complete_and_failure_does_not():
    class Truncating(Environment):
        def step(self, outputs):
            return self.observation(), 'truncated'
    session, _ = make(Truncating())
    advance(session)
    session.respond(motor(1))
    assert advance(session)[-1].meta['reason'] == 'truncated' and session.complete
    session, _ = make(Environment())
    advance(session)
    with pytest.raises(ValueError):
        session.respond(motor(9))
    assert session.next_input().meta['reason'] == 'failure'
    assert session.next_input() is None and not session.complete


def test_unrequested_outputs_are_refused_and_do_not_advance_the_environment():
    env = Environment()
    session, _ = make(env)
    advance(session)
    with pytest.raises(ValueError, match='not requested'):
        session.respond(StreamEvent('neural:V4', 'activity', None))
    assert env.actions == [] and session.emitted == []


def test_a_policy_output_that_was_not_requested_is_an_error():
    env = Environment()
    subject = _typed_subject(lambda row: {'motor': 1, 'neural:V4': 'activity'})
    with pytest.raises(ValueError, match='not requested'):
        run_environment(subject, env)


def test_a_source_over_another_output_channel_needs_no_motor():
    class Talking:
        def __init__(self):
            self.actions = []

        def reset(self):
            return {'observation': 0}

        def step(self, outputs):
            self.actions.append(outputs['behavior:text'])
            return {'observation': 1}, 'terminal'
    env = Talking()
    source = EnvironmentSource(env, action_channel='behavior:text')
    session = Session(source, ('behavior:text',)).declare_inputs(ENV_INPUTS)
    advance(session)
    session.respond(StreamEvent('behavior:text', 'hello', None))
    assert env.actions == ['hello']
    assert [e.payload for e in session.emitted] == ['hello']


def _typed_subject(process):
    """A minimal typed subject: process() plus the shared typed session driver."""
    from brainscore_core.streaming_helpers import drive_typed_session
    subject = SimpleNamespace(process=process)
    subject.participate = lambda session: drive_typed_session(subject, session)
    return subject


@pytest.mark.parametrize('action', [-1, 3, 1.5, True, [0, 1], [[0]]])
def test_invalid_actions_are_rejected_before_the_environment(action):
    env = Environment()
    session, source = make(env)
    advance(session)
    with pytest.raises(ValueError):
        session.respond(motor(action))
    assert env.actions == session.emitted == []
    assert source.action_events[-1].meta['action_status'] == 'rejected'
    assert session.next_input().meta['reason'] == 'failure'
    assert session.next_input() is None


@pytest.mark.parametrize('error', [RuntimeError('lost connection'), KeyboardInterrupt()])
def test_failed_step_is_unknown_execution_never_success(error):
    env = Environment(error=error)
    session, source = make(env)
    with session:
        advance(session)
        with pytest.raises(type(error)):
            session.respond(motor(1))
        assert session.emitted == []
        assert source.action_events[-1].meta['action_status'] == 'execution_unknown'
        assert session.next_input().meta['reason'] == 'failure'
        assert session.next_input() is None
    assert env.closed == 1


def test_bad_observation_fails_before_inference_and_closes():
    env = Environment()
    env.observation = lambda **kwargs: {'observation': {'image': [1], 'task': 'move'}}
    subject = _typed_subject(lambda row: pytest.fail('Bad observation reached policy'))
    with pytest.raises(ValueError, match='image'):
        run_environment(subject, env, require_specs=True)
    assert env.actions == [] and env.closed == 1


def test_missing_specs_or_unsupported_seed_fail_before_reset():
    calls = []
    env = SimpleNamespace(reset=lambda: calls.append('reset'), step=lambda outputs: None)
    with pytest.raises(TypeError, match='observation_spec'):
        EnvironmentSource(env, require_specs=True)
    with pytest.raises(TypeError, match='seed'):
        EnvironmentSource(env, seed=1)
    assert calls == []


def test_unknown_time_stays_unknown_and_simulated_time_is_preserved():
    env = Environment()
    session, source = make(env)
    advance(session)
    assert source.input_events[0].t_ms is None
    assert source.input_events[0].meta['time_source'] == 'step_index'
    env = Environment()
    env.t_ms = 250.
    session, source = make(env)
    advance(session)
    session.respond(motor(1, 999))
    assert session.emitted[0].t_ms == 250.
    assert session.emitted[0].meta['time_source'] == 'simulation'
    env.t_ms = 0.
    with pytest.raises(ValueError, match='backwards'):
        session.next_input()


@pytest.mark.parametrize('t_ms', [float('nan'), float('inf'), True])
def test_invalid_time_is_rejected(t_ms):
    env = Environment()
    env.t_ms = t_ms
    session, _ = make(env)
    with pytest.raises(ValueError, match='finite milliseconds'):
        session.next_input()


def test_environment_mutation_cannot_rewrite_submitted_command():
    class Mutating(Environment):
        def action_spec(self):
            return ArraySpec((2,), 'float64')

        def step(self, outputs):
            outputs['motor'][:] = 99
            return {'observation': {}}, 'terminal'
    original = np.array([1., 2.])
    session, _ = make(Mutating())
    with session:
        advance(session)
        session.respond(motor(original))
    np.testing.assert_array_equal(original, [1, 2])
    np.testing.assert_array_equal(session.emitted[0].payload, [1, 2])


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
    session, source = make(env)
    with session:
        advance(session)
        def unavailable(event):
            raise OSError('record storage unavailable')
        with source.observe_actions(unavailable):
            with pytest.raises(OSError, match='storage unavailable'):
                session.respond(motor(1))
        assert not session.complete
        assert env.actions == []
        assert session.next_input().meta['reason'] == 'failure'
    assert env.closed == 1


def test_proposal_is_observed_before_environment_step():
    env = Environment()
    ordering = []
    original_step = env.step
    def step(outputs):
        ordering.append('environment_step')
        return original_step(outputs)
    env.step = step
    session, source = make(env)
    with session:
        advance(session)
        with source.observe_actions(lambda event: ordering.append(event.meta['action_status'])):
            session.respond(motor(1))
    assert ordering == ['proposed', 'environment_step', 'applied']


def test_custom_action_validator_cannot_bypass_declared_bounds():
    env = Environment()
    session, _ = make(env, action_validator=lambda value: 99)
    advance(session)
    with pytest.raises(ValueError, match='bounds'):
        session.respond(motor(1))
    assert env.actions == []


def test_helper_closes_environment_if_preflight_fails():
    closed = []
    env = SimpleNamespace(
        reset=lambda: None, step=lambda outputs: None,
        close=lambda: closed.append(True),
    )
    with pytest.raises(TypeError, match='seed'):
        run_environment(SimpleNamespace(), env, seed=2)
    assert closed == [True]


def test_native_session_rejects_an_unscheduled_chunk():
    env = Environment()
    session, _ = make(env)
    advance(session)
    with pytest.raises(ValueError, match='Schedule policy chunks'):
        session.respond(motor(np.ones((2, 3)), action_kind='chunk'))
    assert env.actions == []


class LongEnvironment:
    """Model-free episode of `length` actions, then a terminal observation."""

    def __init__(self, length, error=None):
        self.length, self.error, self.actions = length, error, []

    def clock(self):
        return {'t_ms': 10. * len(self.actions)}

    def reset(self, *, seed=None, options=None):
        return {'observation': {'n': 0}}

    def step(self, outputs):
        self.actions.append(outputs['motor'])
        if self.error and len(self.actions) == self.length:
            raise self.error
        last = len(self.actions) >= self.length
        return {'observation': {'n': len(self.actions)}}, 'terminal' if last else None


def _run_long(history_limit, length=50, error=None):
    env = LongEnvironment(length, error)
    seen, observed = [], []
    subject = SimpleNamespace(process=lambda row: seen.append(row) or {'motor': np.array([row['observation']['n']])})
    session, source = make(env, seed=3, history_limit=history_limit)
    with session, source.observe_actions(observed.append):
        _drive_environment_via_process(subject, session)
    return env, session, source, seen, observed


def test_history_limit_default_is_unchanged():
    _, default, source, _, _ = _run_long(None)
    assert len(source.input_events) == 51 and len(default.emitted) == 50
    assert len(source.action_events) == 100
    assert 'history_limit' not in source.describe()
    assert [r.payload[0] for r in default.collect('motor')] == list(range(50))


@pytest.mark.parametrize('limit', [0, 1, 7])
def test_history_limit_bounds_lists_without_hiding_events(limit):
    env, session, source, seen, observed = _run_long(limit)
    assert session.complete and len(env.actions) == 50
    assert len(source.input_events) == len(session.emitted) == len(source.action_events) == limit
    assert type(source.input_events) is type(session.emitted) is type(source.action_events) is list
    if limit:
        assert session.emitted[-1].payload[0] == 49
        assert source.input_events[-1].t_ms == 500.
    # The subject and observers saw every event, in order.
    assert [r['observation']['n'] for r in seen] == list(range(50))
    assert [e.meta['action_status'] for e in observed] == ['proposed', 'applied'] * 50
    assert [e.meta['step_num'] for e in observed[1::2]] == list(range(50))
    assert source.describe()['history_limit'] == limit
    with pytest.raises(RuntimeError, match='dropped'):
        session.collect('motor')


def test_history_limit_large_enough_still_collects():
    _, session, _, _, _ = _run_long(200)
    assert [r.payload[0] for r in session.collect('motor')] == list(range(50))


@pytest.mark.parametrize('limit', [0, 1])
def test_history_limit_keeps_failure_receipts_observable(limit):
    observed = []
    env = Environment()
    session, source = make(env, history_limit=limit)
    with source.observe_actions(observed.append):
        advance(session)
        with pytest.raises(ValueError):
            session.respond(motor(5))
    assert [e.meta['action_status'] for e in observed] == ['proposed', 'rejected']
    assert [e.meta['action_status'] for e in source.action_events] == ['rejected'][:limit]
    assert session.next_input().meta['reason'] == 'failure' and not session.complete

    observed = []
    env = LongEnvironment(3, error=RuntimeError('lost'))
    env.closed = 0
    env.close = lambda: setattr(env, 'closed', env.closed + 1)
    subject = SimpleNamespace(process=lambda row: {'motor': np.array([0])})
    session, source = make(env, history_limit=limit)
    with session, source.observe_actions(observed.append):
        with pytest.raises(RuntimeError, match='lost'):
            _drive_environment_via_process(subject, session)
    assert [e.meta['action_status'] for e in observed] == ['proposed', 'applied'] * 2 + [
        'proposed', 'execution_unknown']
    assert session.next_input().meta['reason'] == 'failure' and not session.complete and env.closed == 1


@pytest.mark.parametrize('limit, error', [(-1, ValueError), (True, TypeError),
                                          (1.0, TypeError), ('3', TypeError)])
def test_invalid_history_limit_fails_before_reset(limit, error):
    calls = []
    env = SimpleNamespace(reset=lambda: calls.append('reset'), step=lambda outputs: None)
    with pytest.raises(error, match='history_limit'):
        EnvironmentSource(env, history_limit=limit)
    assert calls == []


def test_completion_is_declared_by_the_source():
    assert Session(()).complete is True
    with Session(EnvironmentSource(Environment()), ('motor',)).declare_inputs(ENV_INPUTS) as session:
        assert session.complete is False


def test_a_subject_that_does_not_declare_proprioception_never_receives_it():
    class Robot(Environment):
        def observation(self):
            return {'observation': {'task': 'move', 'image': np.zeros((2, 3, 3), dtype=np.uint8)},
                    'proprioception': {'joint_position': np.zeros(7)}}
    received = []

    def participate(session):
        session.declare_inputs({'observation', 'control'})
        while (event := session.next_input()) is not None:
            if event.channel == 'control' and event.payload == 'respond':
                session.respond(motor(1))
            received.append(event.channel)
    subject = SimpleNamespace(participate=participate)
    assert run_environment(subject, Robot()) == [1]
    assert 'proprioception' not in received and 'observation' in received


def test_no_session_subclass_exists_in_core():
    import importlib
    import pkgutil
    import brainscore_core
    for module in pkgutil.walk_packages(brainscore_core.__path__, 'brainscore_core.'):
        if '.plugin_management' in module.name or '.submission' in module.name or '.travis' in module.name:
            continue
        try:
            importlib.import_module(module.name)
        except Exception:
            continue
    assert Session.__subclasses__() == []


@pytest.mark.parametrize('reason', ['terminal', 'truncated'])
def test_run_environment_with_reset_reason_returns_no_actions(reason):
    env = Environment(reset_reason=reason)
    calls = []
    inner = SimpleNamespace(process=lambda row: calls.append(row) or {'motor': 1})
    subject = SimpleNamespace(
        participate=lambda session: _drive_environment_via_process(inner, session))
    assert run_environment(subject, env) == []
    assert calls == [] and env.actions == [] and env.closed == 1
