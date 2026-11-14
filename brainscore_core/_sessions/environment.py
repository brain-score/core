"""Closed-loop input: an environment driven as a session source.

An environment is plain benchmark code with two methods: ``reset(**kwargs) -> row``
(or ``(row, reason)``, the form ``step`` uses, for an episode that is already
complete at reset: the row is delivered if non-empty, then ``end(reason)``, and
the subject is never asked for an action; a row is a mapping, so a tuple is
unambiguous) and ``step(outputs) -> (row, reason_or_None)``. A row is a mapping of channel ->
payload; ``outputs`` is a mapping of output channel -> payload; ``reason`` is
``None`` while the episode runs and ``'terminal'``, ``'truncated'`` or
``'failure'`` when it ends. Optional: ``observation_spec()`` / ``action_spec()``
(``validate`` and ``describe``), ``clock() -> {'t_ms', 'time_source',
'time_inferred'}`` and ``close()``.
"""

from __future__ import annotations
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import time
from typing import Any, Mapping, Optional
import numpy as np
from ..streaming import (
    BEGIN, CONTROL_CHANNEL, END_REASONS, RESPOND, Session, Source, StreamEvent,
    end_event, events_row, respond_event, row_events,
)


class EnvironmentSource(Source):
    """One turn-based episode, with declared spaces and execution receipts.

    Per episode it emits ``begin``; per step one event per row entry, then
    ``respond``; after the subject answers on ``action_channel`` it advances the
    environment; at the end it emits ``end(reason)``. Specs are used when the
    environment provides them, and required with ``require_specs=True``.
    Use as a context manager to close the environment; subject reset belongs to
    the trial owner. Terminal rows are delivered once, without a ``respond``.

    ``history_limit`` bounds memory for long episodes: ``input_events`` and
    ``action_events`` here, and the session's ``emitted``, keep only their latest
    N entries (0 keeps none; None keeps everything). The subject and
    ``observe_actions()`` callbacks still see every event. ``Session.collect``
    refuses once an applied action has been dropped.

    ``complete`` is False until an episode ends as ``terminal`` or ``truncated``.
    """

    def __init__(self, environment, *, action_channel: str = 'motor', action_validator=None,
                 seed=None, options=None, require_specs=False,
                 history_limit: Optional[int] = None):
        if history_limit is not None and (
                isinstance(history_limit, bool) or not isinstance(history_limit, int)):
            raise TypeError("history_limit must be an int or None")
        if history_limit is not None and history_limit < 0:
            raise ValueError("history_limit must be >= 0")
        self.history_limit = history_limit
        self.action_channel = action_channel
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
        self.input_events: list[StreamEvent] = []
        self.action_events: list[StreamEvent] = []
        self._action_observers = []
        self._queue: deque = deque()
        self._started = self._done = self._closed = self._awaiting = self._failed = False
        self._complete = False
        self._pending = None
        self._step_num = 0
        self._timing: dict = {}
        self._t_ms = None
        self._previous_t_ms = None

    @property
    def complete(self) -> bool:
        return self._complete

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
        actions; the source cannot then be counted as complete.
        """
        self._action_observers.append(callback)
        try:
            yield
        finally:
            self._action_observers.remove(callback)

    def _retain(self, history, event) -> None:
        """Append within history_limit, dropping the oldest entry."""
        history.append(event)
        if self.history_limit is not None and len(history) > self.history_limit:
            del history[0]  # keep the public history attributes as lists

    def _record_action(self, event):
        self._retain(self.action_events, event)
        try:
            for callback in tuple(self._action_observers):
                callback(event)
        except BaseException:
            self._fail()
            raise

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Environment source is closed")
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

    def _fail(self) -> None:
        """Stop the episode as a failure; the next input is its ``end('failure')``."""
        if self._failed:
            return
        self._failed = True
        self._queue.append(end_event('failure', self._t_ms))
        self._done = True
        self._awaiting = False
        self._complete = False

    def next_input(self) -> Optional[StreamEvent]:
        event = self._next_event()
        if event is not None and event.channel != CONTROL_CHANNEL and self.history_limit != 0:
            self._retain(self.input_events, deepcopy(event))
        return event

    def _next_event(self) -> Optional[StreamEvent]:
        if self._queue:
            return self._queue.popleft()
        if self._done:
            return None
        if self._awaiting:
            raise RuntimeError("Emit an action before requesting the next observation. A closed-loop "
                               "subject must declare the 'control' input and answer on the respond signal.")
        try:
            if not self._started:
                self._started = True
                result = self.environment.reset(**self.reset_kwargs)
                row, reason = result if isinstance(result, tuple) else (result, None)
                self._enqueue(row, reason, first=True)
            else:
                row, reason = self._pending
                self._pending = None
                self._enqueue(row, reason)
        except BaseException:
            self._fail()
            raise
        return self._queue.popleft()

    def _clock(self):
        clock = getattr(self.environment, 'clock', None)
        info = clock() if callable(clock) else {}
        t_ms = info.get('t_ms')
        if t_ms is None:
            return None, {'time_source': 'step_index', 'time_inferred': False}
        if isinstance(t_ms, (bool, np.bool_)) or not np.isfinite(t_ms):
            raise ValueError("Environment time must be finite milliseconds or None")
        return float(t_ms), {'time_source': info.get('time_source', 'environment'),
                             'time_inferred': info.get('time_inferred', False)}

    def _enqueue(self, row: Mapping[str, Any], reason: Optional[str], first: bool = False) -> None:
        if not isinstance(row, Mapping):
            raise TypeError(f"environment reset/step must return a row (a mapping); got {type(row).__name__}.")
        if reason is not None and reason not in END_REASONS:
            raise ValueError(f"end reason must be one of {END_REASONS}; got {reason!r}")
        row = dict(row)
        if self.observation_spec is not None and 'observation' in row:
            row['observation'] = self.observation_spec.validate(row['observation'])
        t_ms, timing = self._clock()
        if (t_ms is not None and self._previous_t_ms is not None
                and t_ms < self._previous_t_ms):
            raise ValueError("Environment timestamps must not go backwards")
        if t_ms is not None:
            self._previous_t_ms = t_ms
        self._t_ms, self._timing = t_ms, timing
        if first:
            self._queue.append(StreamEvent(CONTROL_CHANNEL, BEGIN, t_ms, {
                **timing, 'environment_session': self.describe()}))
        self._queue.extend(row_events(row, t_ms, step_num=self._step_num, **timing))
        if reason is None:
            self._queue.append(respond_event(t_ms))
            self._awaiting = True
        else:
            self._queue.append(end_event(reason, t_ms))
            self._done = True
            self._awaiting = False
            self._complete = reason != 'failure'

    def respond(self, event: StreamEvent) -> Optional[StreamEvent]:
        if event.channel != self.action_channel:
            return None  # another output (a recording): the session keeps it
        if not self._awaiting or self._done:
            raise RuntimeError("No observation is awaiting an action. A closed-loop subject must "
                               "declare the 'control' input and answer once, on the 'respond' signal.")
        meta = {**event.meta, **self._timing, 'step_num': self._step_num,
                'execution': 'turn_based', 'action_status': 'proposed'}
        proposal = deepcopy(replace(event, t_ms=self._t_ms, meta=meta))
        self._record_action(proposal)
        action = event.payload
        attempted = False
        try:
            if event.meta.get('action_kind') == 'chunk':
                raise ValueError("Schedule policy chunks before submitting a single environment action")
            if self.action_spec is not None:
                action = self.action_spec.validate(action)
            if self.action_validator is not None:
                action = self.action_validator(action)
                if self.action_spec is not None:
                    action = self.action_spec.validate(action)
            # Snapshot the actual submitted command before an environment can mutate it.
            payload = deepcopy(action)
            attempted = True
            row, reason = self.environment.step({self.action_channel: action})
        except BaseException as error:
            status = 'execution_unknown' if attempted else 'rejected'
            self._fail()
            self._record_action(replace(proposal, meta={
                **meta, 'action_status': status, 'error_type': type(error).__name__,
                'error': str(error),
            }))
            raise
        receipt = replace(proposal, payload=payload, meta={**meta, 'action_status': 'applied'})
        self._record_action(receipt)
        self._awaiting = False
        self._step_num += 1
        self._pending = (row, reason)
        return receipt


def environment_session(environment, outputs=None, task=None, *,
                        action_channel: str = 'motor', action_validator=None, seed=None,
                        options=None, require_specs=False,
                        history_limit: Optional[int] = None) -> Session:
    """A session driving ``environment``.

    ``outputs`` defaults to the action channel alone.
    """
    source = EnvironmentSource(
        environment, action_channel=action_channel, action_validator=action_validator,
        seed=seed, options=options, require_specs=require_specs, history_limit=history_limit,
    )
    requested = ((action_channel,) if outputs is None
                 else outputs)
    return Session(source, requested, task)


def run_environment(subject, environment, *, outputs=None, action_channel: str = 'motor',
                    task=None, seed=None, options=None, action_validator=None,
                    require_specs=False) -> list:
    """Run one closed-loop episode and close its environment, returning the applied actions.

    The subject must implement ``participate(session)``; there is no ``process()``
    fallback for closed-loop sessions. The returned list holds the payloads sent on
    ``action_channel``.

    Reset rule: the caller owns ``subject.reset()`` between episodes (or uses
    Experiment with SessionProtocol, which resets before and after each trial).
    A policy that keeps episode state is cleared when the episode begins.
    """
    try:
        session = environment_session(
            environment, outputs, task, action_channel=action_channel,
            seed=seed, options=options,
            action_validator=action_validator, require_specs=require_specs,
        )
    except BaseException:
        close = getattr(environment, 'close', None)
        if callable(close):
            close()
        raise
    with session:
        participate = getattr(subject, "participate", None)
        if not callable(participate):
            raise TypeError(
                f"{type(subject).__name__} has no participate(session); closed-loop "
                "sessions require a Subject. Wrap a bare policy function as "
                "BrainScoreModel(..., policy=fn)."
            )
        participate(session)
        if not session.complete:
            raise RuntimeError("Environment session ended before episode completion")
        if not session.emitted:  # episode ended at reset, no action was sent
            return []
        return [event.payload for event in session.collect(action_channel)]


def drive_environment_session(subject, session: Session, driver: str = "interact",
                              on_begin=None) -> None:
    """Drive a live session through ``subject.process(row) -> outputs``.

    Events between ``respond`` controls form one row; ``outputs`` is a mapping of
    output channel -> payload, answered on every channel it holds.
    ``on_begin`` runs when an episode begins (a stateful policy clears itself).
    """
    session._request_control('A closed-loop session')
    pending = []
    while (event := session.next_input()) is not None:
        if event.channel != CONTROL_CHANNEL:
            pending.append(event)
        elif event.payload == BEGIN:
            pending = []
            if on_begin is not None:
                on_begin()
        elif event.payload == RESPOND:
            row, pending = events_row(pending), []
            start = time.perf_counter()
            outputs = subject.process(row)
            latency_ms = (time.perf_counter() - start) * 1000
            send_outputs(session, outputs, event.t_ms, {
                "driver": driver, "latency_ms": latency_ms, "action_status": "proposed"})


def send_outputs(session: Session, outputs: Mapping[str, Any], t_ms: Optional[float],
                 meta: Mapping[str, Any]) -> None:
    """Answer a step: one response per output channel a policy returned.

    The session refuses a channel that was not requested.

    A value that is already a ``StreamEvent`` keeps its own meta (telemetry).
    """
    if not isinstance(outputs, Mapping):
        raise TypeError("A policy returns a mapping of output channel -> payload; "
                        f"got {type(outputs).__name__}.")
    for channel, value in outputs.items():
        if isinstance(value, StreamEvent):
            session.respond(replace(value, channel=channel, t_ms=t_ms, meta={**meta, **value.meta}))
        else:
            session.respond(StreamEvent(channel, value, t_ms, dict(meta)))
