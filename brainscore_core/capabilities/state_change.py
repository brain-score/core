"""State-change capability for BrainScoreModel perturbation events."""

from .base import Capability
from .registry import register_capability
from ..events import StateChange


class StateChangeCapability(Capability):
    """Dispatch perturbation StateChange events through the capability seam."""

    identifier = 'state-change'
    order = 50

    def handles(self, model, event, **kwargs) -> bool:
        del model, kwargs
        return isinstance(event, StateChange)

    def process(self, model, event, **kwargs):
        del kwargs
        return model._perturbations.dispatch_state_change(event)


register_capability(StateChangeCapability())
