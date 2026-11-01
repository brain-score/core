"""Explicit adapters for session records created before channel envelopes."""

from collections import deque

from .events import EnvironmentStep
from .io_catalog import INPUT, validate
from .streaming import BufferedSession


class EnvironmentReplaySession(BufferedSession):
    """Replay old typed observations without changing their recorded values.

    This compatibility source is only for existing EnvironmentStep records.
    New buffered sessions use StreamEvent('observation', step, t_ms).
    """

    requested_output_channels = ('motor',)

    def __init__(self, observations) -> None:
        observations = tuple(observations)
        for observation in observations:
            if not isinstance(observation, EnvironmentStep):
                raise TypeError('Environment replay requires EnvironmentStep observations')
            failures = validate('observation', observation, direction=INPUT)
            if failures:
                raise ValueError('Invalid saved observation: ' + '; '.join(failures))
        super().__init__()
        self._inputs = deque(observations)


def interact_through_process(subject, session) -> None:
    """Adapt the explicitly inherited UnifiedModel typed interface."""
    from .streaming_helpers import (
        _drive_behavior_session_via_process,
        _drive_environment_session_via_process,
        _drive_neural_session_via_process,
        _drive_state_change_session_via_process,
        _requested_output_channels,
    )
    channels = set(_requested_output_channels(session))
    drivers = {
        'behavior': _drive_behavior_session_via_process,
        'motor': _drive_environment_session_via_process,
        'perturbation': _drive_state_change_session_via_process,
    }
    if channels and all(channel.startswith('neural:') for channel in channels):
        _drive_neural_session_via_process(subject, session, driver='process')
    elif len(channels) == 1 and next(iter(channels)) in drivers:
        drivers[next(iter(channels))](subject, session, driver='process')
    else:
        raise ValueError(f'Typed compatibility driver cannot fulfill {sorted(channels)}')
