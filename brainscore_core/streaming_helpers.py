"""Public session helpers: build a session, run a subject on it, read its outputs.

The implementations are grouped under ``_sessions``; import them from here.
"""

from brainscore_core._sessions.buffered import (
    apply_state_change,
    behavioral_response,
    neural_response,
    stimulus_session,
)

from brainscore_core._sessions.environment import (
    EnvironmentSource,
    drive_environment_session,
    environment_session,
    run_environment,
    send_outputs,
)

from brainscore_core._sessions.drivers import (
    demultiplex_neural_output,
    drive_behavior_session_streaming,
    drive_neural_session_streaming,
    drive_state_change_session,
    refuse_variance,
    stream_trials,
    drive_behavior_session,
    drive_neural_session,
    drive_typed_session,
    measurement_settings,
)

from brainscore_core._sessions.payloads import (
    behavior_stimuli_from_session,
    session_outputs,
    drain_session_inputs,
    session_stimuli,
)

from brainscore_core._sessions.view import declared_only, task_input_channels
