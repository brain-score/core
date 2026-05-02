"""Tests for the embodied input event types and `process(EnvironmentStep)`
dispatch on `BrainScoreModel`.

The schema follows DROID (https://droid-dataset.github.io/) for arm
manipulation. These tests cover the dataclass shape, action_fn dispatch,
error handling when no action_fn is registered, and the mobile-manipulation
extension fields on Proprioception.
"""

import numpy as np
import pytest

from brainscore_core.model_interface import (
    BrainScoreModel,
    CameraFrame,
    EnvironmentResponse,
    EnvironmentStep,
    Proprioception,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _droid_step(step_num: int = 0, instruction: str = "pick up the red block"):
    """Construct a DROID-shaped EnvironmentStep with the canonical 3-camera
    setup, 7-DOF Franka proprioception, and a language instruction."""
    cameras = {
        'exterior_1': CameraFrame(rgb=np.zeros((180, 320, 3), dtype=np.uint8)),
        'exterior_2': CameraFrame(rgb=np.zeros((180, 320, 3), dtype=np.uint8)),
        'wrist': CameraFrame(rgb=np.zeros((180, 320, 3), dtype=np.uint8)),
    }
    proprio = Proprioception(
        joint_position=np.zeros(7, dtype=np.float64),
        cartesian_position=np.zeros(6, dtype=np.float64),
        gripper_position=np.zeros(1, dtype=np.float64),
    )
    return EnvironmentStep(
        cameras=cameras,
        proprioception=proprio,
        instruction=instruction,
        step_num=step_num,
        is_first=(step_num == 0),
    )


def _droid_action_fn(env_step):
    """Trivial policy: zero action, echo the step_num in metadata."""
    return EnvironmentResponse(
        action=np.zeros(7, dtype=np.float64),
        metadata={'step_num_seen': env_step.step_num,
                  'instruction_seen': env_step.instruction},
    )


def _make_model(action_fn=None):
    """Construct a minimal BrainScoreModel for embodied dispatch testing.

    `model=None` is fine because the embodied path doesn't touch
    activations_model / preprocessors when action_fn handles the step.
    """
    return BrainScoreModel(
        identifier='test-embodied-policy',
        model=None,
        region_layer_map={},
        preprocessors={'vision': lambda x: x},  # token preprocessor; unused
        action_fn=action_fn,
    )


# ── Dataclass shape ─────────────────────────────────────────────────

class TestEnvironmentStepShape:

    def test_droid_shaped_step_constructs(self):
        step = _droid_step()
        assert step.step_num == 0
        assert step.is_first is True
        assert step.is_last is False
        assert step.instruction == "pick up the red block"
        assert set(step.cameras.keys()) == {'exterior_1', 'exterior_2', 'wrist'}
        for cam in step.cameras.values():
            assert cam.rgb.shape == (180, 320, 3)
            assert cam.depth is None  # depth optional

    def test_proprioception_droid_fields(self):
        proprio = Proprioception(
            joint_position=np.zeros(7),
            cartesian_position=np.zeros(6),
            gripper_position=np.zeros(1),
        )
        assert proprio.joint_position.shape == (7,)
        assert proprio.cartesian_position.shape == (6,)
        assert proprio.gripper_position.shape == (1,)
        # Optional velocity fields default to None
        assert proprio.joint_velocity is None
        assert proprio.cartesian_velocity is None
        # Optional mobile-manip fields default to None
        assert proprio.base_position is None
        assert proprio.base_velocity is None

    def test_proprioception_with_velocities(self):
        proprio = Proprioception(
            joint_position=np.zeros(7),
            cartesian_position=np.zeros(6),
            gripper_position=np.zeros(1),
            joint_velocity=np.ones(7),
            cartesian_velocity=np.ones(6),
            gripper_velocity=np.array([0.5]),
        )
        assert proprio.joint_velocity.shape == (7,)
        assert proprio.gripper_velocity.shape == (1,)

    def test_mobile_manipulation_extension(self):
        """Mobile robots populate base_position / base_velocity. Existing
        DROID-format consumers see them as optional and ignore."""
        proprio = Proprioception(
            joint_position=np.zeros(7),
            cartesian_position=np.zeros(6),
            gripper_position=np.zeros(1),
            base_position=np.array([1.0, 2.0, 0.5]),  # x, y, theta
            base_velocity=np.array([0.1, 0.0, 0.05]),  # vx, vy, omega
        )
        assert proprio.base_position.shape == (3,)
        assert proprio.base_velocity.shape == (3,)

    def test_camera_frame_with_depth(self):
        cam = CameraFrame(
            rgb=np.zeros((180, 320, 3), dtype=np.uint8),
            depth=np.ones((180, 320), dtype=np.float32) * 1.5,  # 1.5 m
        )
        assert cam.depth.shape == (180, 320)
        assert cam.depth.dtype == np.float32

    def test_episode_control_signals(self):
        first = _droid_step(step_num=0)
        assert first.is_first is True
        assert first.is_last is False
        assert first.is_terminal is False

        mid = _droid_step(step_num=10)
        assert mid.is_first is False  # _droid_step sets is_first iff step_num==0

        # Terminal step with reward
        terminal = EnvironmentStep(
            cameras=mid.cameras,
            proprioception=mid.proprioception,
            step_num=99,
            is_last=True,
            is_terminal=True,
            reward=1.0,
            discount=0.0,
        )
        assert terminal.is_last is True
        assert terminal.reward == 1.0

    def test_environment_response_compact_action(self):
        """DROID's compact action is (7,): 6 joint vel + 1 gripper pos."""
        resp = EnvironmentResponse(action=np.zeros(7, dtype=np.float64))
        assert resp.action.shape == (7,)
        assert resp.action_dict is None
        assert resp.metadata == {}

    def test_environment_response_structured_action(self):
        """Optional action_dict mirrors DROID's structured action."""
        resp = EnvironmentResponse(
            action=np.zeros(7),
            action_dict={
                'cartesian_velocity': np.zeros(6),
                'gripper_position': np.zeros(1),
            },
        )
        assert 'cartesian_velocity' in resp.action_dict
        assert resp.action_dict['cartesian_velocity'].shape == (6,)


# ── Dispatch ────────────────────────────────────────────────────────

class TestProcessEnvironmentStep:

    def test_process_routes_to_action_fn(self):
        model = _make_model(action_fn=_droid_action_fn)
        step = _droid_step(step_num=5, instruction="pour the water")
        response = model.process(step)
        assert isinstance(response, EnvironmentResponse)
        assert response.action.shape == (7,)
        assert response.metadata['step_num_seen'] == 5
        assert response.metadata['instruction_seen'] == "pour the water"

    def test_process_no_action_fn_raises(self):
        model = _make_model(action_fn=None)
        step = _droid_step()
        with pytest.raises(NotImplementedError, match="no action_fn registered"):
            model.process(step)

    def test_process_action_fn_must_return_environment_response(self):
        """An action_fn that returns a bare ndarray (instead of wrapping in
        EnvironmentResponse) raises a clear TypeError."""
        def bad_action_fn(env_step):
            return np.zeros(7)  # wrong — should be EnvironmentResponse(...)

        model = _make_model(action_fn=bad_action_fn)
        with pytest.raises(TypeError, match="expected EnvironmentResponse"):
            model.process(_droid_step())

    def test_action_fn_receives_full_step(self):
        """The action_fn sees the unmodified EnvironmentStep — cameras,
        proprio, instruction, step_num all reach the model."""
        captured = {}

        def capturing_action_fn(env_step):
            captured['cameras'] = list(env_step.cameras.keys())
            captured['joint_pos'] = env_step.proprioception.joint_position.copy()
            captured['instruction'] = env_step.instruction
            captured['step_num'] = env_step.step_num
            return EnvironmentResponse(action=np.zeros(7))

        model = _make_model(action_fn=capturing_action_fn)
        step = _droid_step(step_num=42, instruction="hello world")
        # Customize joint_position to verify it propagates verbatim
        step.proprioception.joint_position = np.arange(7, dtype=np.float64)
        model.process(step)

        assert captured['cameras'] == ['exterior_1', 'exterior_2', 'wrist']
        assert np.array_equal(captured['joint_pos'], np.arange(7))
        assert captured['instruction'] == "hello world"
        assert captured['step_num'] == 42

    def test_step_num_propagates_to_metadata(self):
        """Round-trip the step_num through a no-op policy — confirms the
        full event flows from benchmark → process() → action_fn → response."""
        model = _make_model(action_fn=_droid_action_fn)
        for step_num in (0, 1, 5, 100):
            response = model.process(_droid_step(step_num=step_num))
            assert response.metadata['step_num_seen'] == step_num


# ── Multi-step rollout (proxy for benchmark loop) ───────────────────

class TestMultiStepRollout:
    """Proxy for what an embodied benchmark's scoring loop will look like:
    the benchmark calls model.process(env_step) repeatedly, threading the
    environment's response back into the next step."""

    def test_rollout_terminates_on_is_terminal(self):
        model = _make_model(action_fn=_droid_action_fn)
        responses = []
        max_steps = 5
        for step_num in range(max_steps):
            is_last = (step_num == max_steps - 1)
            step = EnvironmentStep(
                cameras={'wrist': CameraFrame(
                    rgb=np.zeros((180, 320, 3), dtype=np.uint8))},
                proprioception=Proprioception(
                    joint_position=np.zeros(7),
                    cartesian_position=np.zeros(6),
                    gripper_position=np.zeros(1),
                ),
                instruction="reach the goal",
                step_num=step_num,
                is_first=(step_num == 0),
                is_last=is_last,
                is_terminal=is_last,
                reward=1.0 if is_last else 0.0,
            )
            responses.append(model.process(step))
            if step.is_terminal:
                break
        assert len(responses) == max_steps
        assert all(isinstance(r, EnvironmentResponse) for r in responses)
