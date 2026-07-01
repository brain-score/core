"""Capability registry for BrainScoreModel."""

from typing import Dict, Iterable

from .base import Capability


capability_registry: Dict[str, Capability] = {}


def register_capability(capability: Capability) -> Capability:
    """Register ``capability`` by identifier and return it."""
    if not getattr(capability, 'identifier', None):
        raise ValueError("Capability must define a non-empty identifier.")
    capability_registry[capability.identifier] = capability
    return capability


def enabled_capabilities(model) -> Iterable[Capability]:
    """Yield capabilities enabled for ``model`` in stable dispatch order."""
    for capability in sorted(capability_registry.values(),
                             key=lambda c: (c.order, c.identifier)):
        if not capability.enabled_for(model):
            continue
        _setup_once(model, capability)
        yield capability


def _setup_once(model, capability: Capability) -> None:
    if not hasattr(model, '_capability_state'):
        model._capability_state = {}
    if not hasattr(model, '_capability_setup_done'):
        model._capability_setup_done = set()
    if capability.identifier in model._capability_setup_done:
        return
    state = capability.setup(model)
    model._capability_state[capability.identifier] = (
        state if state is not None else {}
    )
    model._capability_setup_done.add(capability.identifier)
