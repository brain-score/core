"""Embodied action capability."""

from .base import Capability
from .registry import register_capability
from ..events import EnvironmentResponse, EnvironmentStep, Message


class EmbodiedActionCapability(Capability):
    """Dispatch EnvironmentStep and Message events through ``action_fn``."""

    identifier = 'embodied-action'
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
                f"Model '{model.identifier}' has no action_fn registered. "
                f"Embodied evaluation requires the model to declare an "
                f"action_fn(env_step) -> EnvironmentResponse callable at "
                f"BrainScoreModel construction time. Received: "
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
                f"Model '{model.identifier}' has no action_fn registered; "
                f"process(Message) routes to the agent's responder. Declare "
                f"action_fn(env_step) -> EnvironmentResponse|Message at "
                f"BrainScoreModel construction time."
            )
        response = model._action_fn(EnvironmentStep(observation=input_event))
        if not isinstance(response, (Message, EnvironmentResponse)):
            raise TypeError(
                f"Model '{model.identifier}' action_fn returned "
                f"{type(response).__name__}; a Message responder must return "
                f"a Message or EnvironmentResponse."
            )
        return response


register_capability(EmbodiedActionCapability())
