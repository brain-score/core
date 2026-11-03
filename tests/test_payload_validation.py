"""Core is device-agnostic: it passes closed-loop observations to the policy
without inspecting them. Robot camera checks live in the robotics harness."""
import warnings

import numpy as np

from brainscore_core import BrainScoreModel
from brainscore_core.events import CameraFrame, EnvironmentResponse, EnvironmentStep


def _agent(seen):
    return BrainScoreModel('agent', None, {}, {}, None,
                           policy=lambda step: seen.append(step) or EnvironmentResponse(np.zeros(7)))


def test_core_passes_device_payloads_to_the_policy_unchecked():
    seen = []
    # A malformed frame core used to reject now reaches the policy untouched.
    step = EnvironmentStep(observation={'cameras': {'wrist': CameraFrame(rgb=np.zeros((2, 2)))}})
    assert isinstance(_agent(seen).process(step), EnvironmentResponse)
    assert seen == [step]


def test_deprecated_droid_fields_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        EnvironmentStep(cameras={'wrist': CameraFrame(rgb=np.zeros((8, 8, 3), 'uint8'))})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
