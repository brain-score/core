"""Public session helpers; implementations are grouped under _sessions.

Existing imports resolve to the same functions and classes.
"""

from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import time
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from brainscore_core.contract import Subject, TaskContext, UnifiedModel  # noqa: F401  (re-exported)
from brainscore_core.io_catalog import modalities_to_input_channels
from brainscore_core.streaming import BufferedSession, Session, StreamEvent, parse_channel
from brainscore_core.events import (
    EnvironmentResponse,
    EnvironmentStep,
    PerturbationApplied,
    StateChange,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)

from brainscore_core._sessions.buffered import (
    BehavioralSession,
    StateChangeSession,
    StimulusSetSession,
    apply_state_change,
    behavior_session,
    behavioral_response,
    neural_response,
    state_change_session,
    stimulus_session,
)

from brainscore_core._sessions.environment import (
    EnvironmentSession,
    _coerce_environment_step,
    drive_environment_session,
    _drive_environment_via_process,
    _environment_step_event,
    _environment_step_terminates,
    _environment_time,
    _motor_action_payload,
    environment_session,
    run_environment,
)

from brainscore_core._sessions.perception import (
    RealTimeStreamSession,
    StreamingStimulusSetSession,
    WindowedStreamSession,
    _default_window_to_stimuli,
)

from brainscore_core._sessions.drivers import (
    _combined_recording_target,
    _demultiplex_neural_output,
    drive_behavior_session_streaming,
    _drive_behavior_session_via_process,
    _drive_behavior_via_process,
    drive_neural_session_streaming,
    _drive_neural_session_via_process,
    drive_state_change_session,
    _drive_state_change_via_process,
    _drive_via_process,
    refuse_variance,
    _start_recording,
    drive_behavior_and_neural_session,
    drive_behavior_session,
    drive_neural_session,
    drive_typed_session,
    measurement_settings,
)

from brainscore_core._sessions.payloads import (
    STIMULUS_COLUMN_TO_MODALITY,
    _STATE_CHANGE_KIND_TO_CHANNEL_FAMILY,
    _STIMULUS_INPUT_CHANNEL_FAMILIES,
    _behavior_events,
    _behavior_label_set,
    behavior_stimuli_from_session,
    _behavior_stimuli_to_score,
    _channel_for_stimulus_column,
    _collect_behavior_events,
    _collect_neural_events,
    _collect_requested_channels,
    _drain_stream_events,
    _is_label_payload,
    _is_neural_assembly_payload,
    _neural_channels,
    _one_hot,
    _one_row_stimuli,
    preserve_stimulus_set_type,
    reconstruct_stimulus_set_from_events,
    requested_output_channels,
    _row_value,
    _single_row_stimuli,
    _spans_several_inputs,
    _state_change_address,
    _state_change_channel,
    _state_change_event,
    _state_change_family,
    _state_change_handle_id,
    _stimuli_from_stream_session,
    _stimulus_events,
    _tag_events,
    drain_session_inputs,
    session_stimuli,
)
