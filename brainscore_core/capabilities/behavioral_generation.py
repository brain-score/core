"""Behavioral generation capability."""

from .base import Capability
from .registry import register_capability


class BehavioralGenerationCapability(Capability):
    """Dispatch instruction-following behavioral tasks."""

    identifier = 'behavioral-generation'
    order = 100

    def enabled_for(self, model) -> bool:
        return (
            model._use_generation_for_task
            and model._task_context is not None
            and model._requires_behavioral_readout(model._task_context.task_type)
        )

    def handles(self, model, event, **kwargs) -> bool:
        del model, kwargs
        try:
            event.columns
        except AttributeError:
            return False
        return True

    def process(self, model, stimuli, **kwargs):
        del kwargs
        return model._generate_predictions(stimuli)


register_capability(BehavioralGenerationCapability())
