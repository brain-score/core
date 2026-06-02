"""Tests for the v1.5 device-agnostic ``EnvironmentStep`` envelope.

v1.5 generalizes ``EnvironmentStep`` so the observation payload is
harness-defined. The deprecated DROID fields (``cameras``, ``proprioception``)
stay optional for one release; constructing with them still works and bridges
onto ``observation``.
"""
import numpy as np

from brainscore_core.model_interface import (
    BrainScoreModel,
    CameraFrame,
    EnvironmentResponse,
    EnvironmentStep,
    Proprioception,
)


def _droid_proprio():
    return Proprioception(
        joint_position=np.zeros(7),
        cartesian_position=np.zeros(6),
        gripper_position=np.zeros(1),
    )


class TestDeviceAgnosticEnvelope:
    def test_generic_observation_construction(self):
        # New code: a harness-defined observation payload (here, an arbitrary dict).
        step = EnvironmentStep(
            observation={"frame": np.zeros((84, 84, 3)), "score": 0},
            instruction="play breakout",
        )
        assert step.observation["score"] == 0
        assert step.instruction == "play breakout"
        assert step.cameras is None and step.proprioception is None

    def test_droid_construction_still_works(self):
        # Zero-regression: the deprecated DROID-shaped construction is unchanged.
        step = EnvironmentStep(
            cameras={"wrist": CameraFrame(rgb=np.zeros((180, 320, 3), dtype=np.uint8))},
            proprioception=_droid_proprio(),
            instruction="pick up the block",
            step_num=3,
            is_first=False,
        )
        assert step.step_num == 3
        assert "wrist" in step.cameras

    def test_droid_fields_bridge_onto_observation(self):
        step = EnvironmentStep(
            cameras={"wrist": CameraFrame(rgb=np.zeros((180, 320, 3), dtype=np.uint8))},
            proprioception=_droid_proprio(),
        )
        assert step.observation is not None
        assert "cameras" in step.observation and "proprioception" in step.observation

    def test_explicit_observation_not_overwritten_by_bridge(self):
        step = EnvironmentStep(
            observation="already set",
            cameras={"wrist": CameraFrame(rgb=np.zeros((1, 1, 3), dtype=np.uint8))},
        )
        assert step.observation == "already set"

    def test_response_action_is_harness_defined(self):
        # A non-DROID action payload (e.g. a discrete game action) is fine.
        assert EnvironmentResponse(action="LEFT").action == "LEFT"
        assert EnvironmentResponse(action=np.zeros(7)).action.shape == (7,)

    def test_action_fn_dispatch_reads_observation(self):
        def policy(env_step):
            return EnvironmentResponse(action=env_step.observation["n"] * 2)

        model = BrainScoreModel(
            identifier="generic-policy",
            model=None,
            region_layer_map={},
            preprocessors={"vision": lambda x: x},
            action_fn=policy,
        )
        resp = model.process(EnvironmentStep(observation={"n": 21}))
        assert resp.action == 42
