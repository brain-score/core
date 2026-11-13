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
from brainscore_core.streaming import Session, StreamEvent, parse_channel
from brainscore_core.events import (
    PerturbationApplied,
    StateChange,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)

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

from brainscore_core._sessions.perception import (
    _default_window_to_stimuli,
    windowed_stream,
)

from brainscore_core._sessions.drivers import (
    _combined_recording_target,
    _demultiplex_neural_output,
    drive_behavior_session_streaming,
    drive_neural_session_streaming,
    drive_state_change_session,
    refuse_variance,
    _start_recording,
    stream_trials,
    drive_behavior_session,
    drive_neural_session,
    drive_typed_session,
    measurement_settings,
)

from brainscore_core._sessions.payloads import (
    _STATE_CHANGE_KIND_TO_CHANNEL_FAMILY,
    _behavior_label_set,
    behavior_stimuli_from_session,
    _behavior_stimuli_to_score,
    _collect_behavior_events,
    _collect_neural_events,
    _is_label_payload,
    _is_neural_assembly_payload,
    _one_hot,
    _one_row_stimuli,
    preserve_stimulus_set_type,
    reconstruct_stimulus_set_from_events,
    session_outputs,
    _row_value,
    _single_row_stimuli,
    _spans_several_inputs,
    _state_change_channel,
    _state_change_event,
    _state_change_family,
    _state_change_handle_id,
    _stimulus_events,
    drain_session_inputs,
    session_stimuli,
)
