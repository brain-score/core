"""Closed-loop capability: the subject sees an observation and returns an action.

Closed-loop means each response can change what the subject observes next
(games, dialogue, robots). "Embodied" is reserved for the robotics subset; core
knows nothing about cameras or joints. Device payloads are validated by their
harness (for example ``brainscore.harnesses.robotics``), not here.
"""

from .base import Capability
from .registry import register_capability
from ..events import EnvironmentResponse, EnvironmentStep, Message


class ClosedLoopCapability(Capability):
    """Route EnvironmentStep and Message events to the model's ``policy``.

    ``policy`` has one signature: it always receives an ``EnvironmentStep``.
    A ``Message`` from another agent arrives as that step's ``observation``, so a
    responder handles games, dialogue and robots the same way.
    """

    identifier = 'closed-loop'
    order = 300

    def handles(self, model, event, **kwargs) -> bool:
        del model, kwargs
        return isinstance(event, (EnvironmentStep, Message))

    def process(self, model, event, **kwargs):
        del kwargs
        if isinstance(event, Message):
            return self._process_message(model, event)
        return self._process_environment_step(model, event)

    def _process_environment_step(self, model, input_event: EnvironmentStep):
        if model._action_fn is None:
            raise NotImplementedError(
                f"Model '{model.identifier}' has no policy (no action_fn registered "
                f"under the compatibility name). Closed-loop evaluation requires "
                f"BrainScoreModel(policy=...), a callable "
                f"policy(env_step) -> EnvironmentResponse. Received: "
                f"EnvironmentStep(step_num={input_event.step_num})."
            )
        response = model._action_fn(input_event)
        if not isinstance(response, EnvironmentResponse):
            raise TypeError(
                f"Model '{model.identifier}' action_fn returned "
                f"{type(response).__name__}; expected EnvironmentResponse. "
                f"Wrap the action in EnvironmentResponse(action=...) so "
                f"benchmarks see a consistent shape across models."
            )
        return response

    def _process_message(self, model, input_event: Message):
        if model._action_fn is None:
            raise NotImplementedError(
                f"Model '{model.identifier}' has no policy (no action_fn registered "
                f"under the compatibility name); a Message routes to the agent's "
                f"policy. Pass BrainScoreModel(policy=...), a callable "
                f"policy(env_step) -> EnvironmentResponse|Message."
            )
        response = model._action_fn(EnvironmentStep(observation=input_event))
        if not isinstance(response, (Message, EnvironmentResponse)):
            raise TypeError(
                f"Model '{model.identifier}' action_fn returned "
                f"{type(response).__name__}; a Message responder must return "
                f"a Message or EnvironmentResponse."
            )
        return response


register_capability(ClosedLoopCapability())
