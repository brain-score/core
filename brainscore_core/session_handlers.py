"""Session handlers for the parts assembled by BrainScoreModel."""

from dataclasses import dataclass
from typing import Callable

from .capabilities.base import Capability
from .capabilities.registry import register_capability

from .streaming_helpers import (
    _drive_behavior_session_streaming,
    _drive_behavior_session_via_process,
    _drive_environment_session_via_process,
    _drive_neural_session_streaming,
    _drive_neural_session_via_process,
    _drive_state_change_session_via_process,
)


@dataclass(frozen=True)
class SessionHandler(Capability):
    """One registered way to fulfill a complete set of requested outputs."""

    identifier: str
    accepts: Callable
    run: Callable
    enabled: Callable

    def enabled_for(self, model) -> bool:
        return self.enabled(model)

    def handles(self, model, event, **kwargs) -> bool:
        return False

    def process(self, model, event, **kwargs):
        raise TypeError('Session handlers are invoked through interact')

    def supports_session(self, model, channels) -> bool:
        return self.accepts(channels)

    def interact(self, model, session) -> None:
        self.run(model, session)


def _neural(model, session) -> None:
    driver = (_drive_neural_session_streaming if session.streaming
              else _drive_neural_session_via_process)
    driver(model, session)


def _behavior(model, session) -> None:
    driver = (_drive_behavior_session_streaming if session.streaming
              else _drive_behavior_session_via_process)
    driver(model, session)


# Stock and external session handlers use the same registration and selection.
register_capability(SessionHandler(
    'neural-session',
    lambda channels: bool(channels) and all(channel.startswith('neural:') for channel in channels),
    _neural,
    lambda model: bool(model._extractors),
))
register_capability(SessionHandler(
    'behavior-session', lambda channels: channels == {'behavior'}, _behavior,
    lambda model: model._generation_fn is not None or (
        model._behavioral_readout_layer is not None and bool(model._extractors)),
))
register_capability(SessionHandler(
    'motor-session', lambda channels: channels == {'motor'},
    _drive_environment_session_via_process, lambda model: model._action_fn is not None,
))
register_capability(SessionHandler(
    'perturbation-session', lambda channels: channels == {'perturbation'},
    _drive_state_change_session_via_process, lambda model: model._state_change_fn is not None,
))
