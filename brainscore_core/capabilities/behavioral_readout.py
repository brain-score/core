"""Behavioral readout capability."""

from .base import Capability
from .registry import register_capability


class BehavioralReadoutCapability(Capability):
    """Dispatch fitted behavioral readout tasks."""

    identifier = 'behavioral-readout'
    order = 200

    def enabled_for(self, model) -> bool:
        return (
            model._readout_classifier is not None
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
        return model._predict_probabilities(stimuli)


register_capability(BehavioralReadoutCapability())
