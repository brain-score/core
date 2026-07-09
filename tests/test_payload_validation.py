"""A4: EnvironmentStep camera payloads are validated at the boundary, so a
malformed frame is rejected with a clear error instead of failing in the policy."""
import numpy as np
import pytest

from brainscore_core import BrainScoreModel
from brainscore_core.events import EnvironmentStep, EnvironmentResponse, CameraFrame


def _agent():
    return BrainScoreModel(
        'agent', None, {}, {}, None,
        action_fn=lambda step: EnvironmentResponse(action=np.zeros(7)),
    )


def test_malformed_camera_payload_rejected():
    step = EnvironmentStep(
        cameras={'wrist': CameraFrame(rgb=np.zeros((2, 2), dtype='float64'))})
    with pytest.raises(ValueError, match="failed validation"):
        _agent().process(step)


def test_valid_camera_payload_passes():
    step = EnvironmentStep(
        cameras={'wrist': CameraFrame(rgb=np.zeros((8, 8, 3), dtype='uint8'))})
    resp = _agent().process(step)
    assert isinstance(resp, EnvironmentResponse)


def test_no_cameras_is_fine():
    resp = _agent().process(EnvironmentStep(observation={'anything': 1}))
    assert isinstance(resp, EnvironmentResponse)
