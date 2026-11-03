"""Turn-based environment sessions, action receipts and lifecycle."""

from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import time
from typing import Optional
import numpy as np
from ..streaming import Session, StreamEvent
from ..events import EnvironmentResponse, EnvironmentStep


class EnvironmentSession(Session):
    """One turn-based episode, with declared spaces and execution receipts.

    Wrappers provide observation_spec()/action_spec() when available. Existing
    untyped wrappers remain usable unless require_specs=True. Use as a context
    manager to close the environment; subject reset belongs to the trial owner.
    Terminal observations are delivered once, without requesting another action.

    history_limit bounds memory for long episodes: input_events, action_events
    and emitted each keep only their latest N entries (0 keeps none; None, the
    default, keeps everything). The model and observe_actions() callbacks still
    see every event. collect() refuses once an applied action has been dropped,
    so a truncated history is never returned as a whole episode.
    """
    records_action_execution = True

    def __init__(self, environment, *, action_validator=None, seed=None,
                 options=None, require_specs=False, history_limit: Optional[int] = None):
        if history_limit is not None and (
                isinstance(history_limit, bool) or not isinstance(history_limit, int)):
            raise TypeError("history_limit must be an int or None")
        if history_limit is not None and history_limit < 0:
            raise ValueError("history_limit must be >= 0")
        self.history_limit = history_limit
        self._dropped_actions = 0
        self._described = False
        self.environment = environment
        self.action_validator = action_validator
        self.reset_kwargs = {}
        if seed is not None:
            self.reset_kwargs['seed'] = seed
        if options is not None:
            self.reset_kwargs['options'] = deepcopy(options)
        for method in ('reset', 'step'):
            if not callable(getattr(environment, method, None)):
                raise TypeError(f"Environment requires {method}()")
        if self.reset_kwargs:
            # Fail before starting an episode if reset cannot accept its settings.
            inspect.signature(environment.reset).bind(**self.reset_kwargs)
        self.observation_spec = self._specification('observation_spec', require_specs)
        self.action_spec = self._specification('action_spec', require_specs)
        self.requested_output_channels = ("motor",)
        self.input_events: list[StreamEvent] = []
        self.emitted: list[StreamEvent] = []
        self.action_events: list[StreamEvent] = []
        self._action_observers = []
        self.complete = False
        self._started = False
        self._done = False
        self._closed = False
        self._awaiting_emit = False
        self._pending_step = None
        self._last_step = None
        self._previous_t_ms = None

    def _specification(self, name, required):
        method = getattr(self.environment, name, None)
        if method is None:
            if required:
                raise TypeError(f"Environment must declare {name}()")
            return None
        spec = method()
        if not callable(getattr(spec, 'validate', None)) or not callable(getattr(spec, 'describe', None)):
            raise TypeError(f"{name} needs validate(value) and describe()")
        return spec

    def describe(self):
        return {
            'observation_spec': self.observation_spec.describe() if self.observation_spec else None,
            'action_spec': self.action_spec.describe() if self.action_spec else None,
            'reset': deepcopy(self.reset_kwargs),
            'execution': 'turn_based',
            'scheduler': 'one_action_per_step',
            **({} if self.history_limit is None else {'history_limit': self.history_limit}),
        }

    @contextmanager
    def observe_actions(self, callback):
        """Observe proposals before execution and receipts afterward.

        Observers must not mutate events. A recording error prevents further
        actions; the session cannot then be counted as complete.
        """
        self._action_observers.append(callback)
        try:
            yield
        finally:
            self._action_observers.remove(callback)

    def _retain(self, history, event) -> bool:
        """Append within history_limit; return True if an older entry was dropped."""
        history.append(event)
        if self.history_limit is None or len(history) <= self.history_limit:
            return False
        # Keep the public history attributes as lists; trimming costs O(history_limit).
        del history[0]
        return True

    def _record_action(self, event):
        self._retain(self.action_events, event)
        try:
            for callback in tuple(self._action_observers):
                callback(event)
        except BaseException:
            self._done = True
            self.complete = False
            raise

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Session is closed")
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if not self._closed:
            self._closed = self._done = True
            close = getattr(self.environment, 'close', None)
            if callable(close):
                close()

    def next_input(self) -> Optional[EnvironmentStep]:
        if self._done:
            return None
        if self._awaiting_emit:
            raise RuntimeError("Emit an action before requesting the next observation")
        try:
            if not self._started:
                self._started = True
                self._pending_step = _coerce_environment_step(
                    self.environment.reset(**self.reset_kwargs)
                )
            step = self._pending_step
            self._pending_step = None
            if step is None:
                self._done = self.complete = True
                return None
            if self.observation_spec is not None:
                step = replace(step, observation=self.observation_spec.validate(step.observation))
            t_ms, timing = _environment_time(step)
            if (t_ms is not None and self._previous_t_ms is not None
                    and t_ms < self._previous_t_ms):
                raise ValueError("Environment timestamps must not go backwards")
            if t_ms is not None:
                self._previous_t_ms = t_ms
            context = {**step.context, **timing, 't_ms': t_ms}
            if not self._described:
                self._described = True
                context['environment_session'] = self.describe()
            step = replace(step, context=context)
            self._last_step = step
            self._done = self.complete = _environment_step_terminates(step)
            self._awaiting_emit = not self._done
            if self.history_limit != 0:
                self._retain(self.input_events, deepcopy(_environment_step_event(step)))
            return step
        except BaseException:
            self._done = True
            self.complete = False
            raise

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        if not self._awaiting_emit or self._done:
            raise RuntimeError("No observation is awaiting an action")
        if event.channel != "motor":
            raise ValueError(f"EnvironmentSession.emit only supports 'motor'; got {event.channel!r}.")
        t_ms, timing = _environment_time(self._last_step)
        meta = {**event.meta, **timing, 'step_num': self._last_step.step_num,
                'execution': 'turn_based', 'action_status': 'proposed'}
        proposal = deepcopy(replace(event, t_ms=t_ms, meta=meta))
        self._record_action(proposal)
        action = _motor_action_payload(event.payload)
        attempted = False
        try:
            if (isinstance(event.payload, EnvironmentResponse)
                    and event.payload.metadata.get('action_kind') == 'chunk'):
                raise ValueError("Schedule policy chunks before submitting a single environment action")
            if self.action_spec is not None:
                action = self.action_spec.validate(action)
            if self.action_validator is not None:
                action = self.action_validator(action)
                if self.action_spec is not None:
                    action = self.action_spec.validate(action)
            # Snapshot the actual submitted command before an environment can mutate it.
            payload = replace(deepcopy(event.payload), action=deepcopy(action)) if isinstance(
                event.payload, EnvironmentResponse
            ) else deepcopy(action)
            attempted = True
            result = self.environment.step(action)
        except BaseException as error:
            status = 'execution_unknown' if attempted else 'rejected'
            self._done = True
            self._awaiting_emit = False
            self._record_action(replace(proposal, meta={
                **meta, 'action_status': status, 'error_type': type(error).__name__,
                'error': str(error),
            }))
            raise
        receipt = replace(proposal, payload=payload, meta={**meta, 'action_status': 'applied'})
        self._dropped_actions += self._retain(self.emitted, receipt)
        self._record_action(receipt)
        self._awaiting_emit = False
        try:
            self._pending_step = _coerce_environment_step(result)
        except BaseException:
            self._done = True
            raise
        if self._pending_step is None:
            self._done = self.complete = True

    def collect(self, channel: str = "motor") -> list:
        if channel != "motor":
            raise ValueError(f"EnvironmentSession.collect only supports 'motor'; got {channel!r}.")
        if self._dropped_actions:
            raise RuntimeError(
                f"history_limit={self.history_limit} dropped {self._dropped_actions} applied "
                "action(s); stream them with observe_actions() instead of collect()"
            )
        return [event.payload for event in self.emitted]


def environment_session(environment, *, action_validator=None, seed=None,
                        options=None, require_specs=False,
                        history_limit: Optional[int] = None) -> EnvironmentSession:
    return EnvironmentSession(
        environment, action_validator=action_validator, seed=seed,
        options=options, require_specs=require_specs, history_limit=history_limit,
    )


def run_environment(subject, environment, *, seed=None, options=None,
                    action_validator=None, require_specs=False) -> list:
    """Run one episode and close its environment, returning applied actions.

    The caller owns subject reset. For isolated trials and tools, use
    Experiment with SessionProtocol, which resets before and after each trial.
    """
    try:
        session = environment_session(
            environment, seed=seed, options=options,
            action_validator=action_validator, require_specs=require_specs,
        )
    except BaseException:
        close = getattr(environment, 'close', None)
        if callable(close):
            close()
        raise
    with session:
        subject.participate(session)
        if not session.complete:
            raise RuntimeError("Environment session ended before episode completion")
        return session.collect("motor")


def _drive_environment_via_process(subject, session: EnvironmentSession) -> None:
    """Drive an environment for a subject that exposes process()."""
    _drive_environment_session_via_process(subject, session, driver="process")


def _drive_environment_session_via_process(
    subject, session: EnvironmentSession, driver: str = "interact"
) -> None:
    """Drive a live environment session through existing EnvironmentStep dispatch."""
    step = session.next_input()
    while step is not None and not _environment_step_terminates(step):
        start = time.perf_counter()
        response = subject.process(step)
        latency_ms = (time.perf_counter() - start) * 1000
        t_ms, timing = _environment_time(step)
        session.emit(StreamEvent(
            channel="motor",
            payload=response,
            t_ms=t_ms,
            meta={"driver": driver, **timing, "step_num": step.step_num,
                  "latency_ms": latency_ms, "action_status": "proposed"},
        ))
        step = session.next_input()


def _coerce_environment_step(result) -> Optional[EnvironmentStep]:
    if result is None:
        return None
    if isinstance(result, EnvironmentStep):
        return result
    raise TypeError(
        "environment reset/step must return an EnvironmentStep or None; "
        f"got {type(result).__name__}."
    )


def _environment_time(step):
    t_ms = step.context.get('t_ms')
    if t_ms is None:
        return None, {'time_source': 'step_index', 'time_inferred': False}
    if isinstance(t_ms, (bool, np.bool_)) or not np.isfinite(t_ms):
        raise ValueError("Environment time must be finite milliseconds or None")
    return float(t_ms), {
        'time_source': step.context.get('time_source', 'environment'),
        'time_inferred': step.context.get('time_inferred', False),
    }


def _environment_step_event(step: EnvironmentStep) -> StreamEvent:
    t_ms, timing = _environment_time(step)
    return StreamEvent(
        channel="observation",
        payload=step,
        t_ms=t_ms,
        meta={**timing, "step_num": step.step_num,
              "is_first": step.is_first, "is_last": step.is_last,
              "is_terminal": step.is_terminal,
              "is_truncated": step.is_last and not step.is_terminal},
    )


def _environment_step_terminates(step: Optional[EnvironmentStep]) -> bool:
    return step is None or bool(step.is_last or step.is_terminal)


def _motor_action_payload(payload):
    if isinstance(payload, EnvironmentResponse):
        return payload.action
    return payload
