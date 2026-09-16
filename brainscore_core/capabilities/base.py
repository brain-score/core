"""Capability extension contract for BrainScoreModel."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Capability(ABC):
    """Self-contained BrainScoreModel behavior.

    A capability owns one dispatch path plus any per-model setup/state it
    needs. New capabilities subclass this contract and register themselves;
    the BrainScoreModel host and Subject ABC stay unchanged.
    """

    identifier: str
    order: int = 1000
    input_channels = frozenset()
    output_channels = frozenset()

    def reset(self, model, state) -> None:
        """Release experiment state. Retain configuration needed by later calls."""
        state.clear()

    def supports_session(self, model, channels) -> bool:
        """Opt into a complete requested output combination before execution."""
        return False

    def interact(self, model, session) -> None:
        """Execute a session accepted by supports_session."""
        raise NotImplementedError(self.identifier)


    def setup(self, model) -> Dict[str, Any]:
        """Return per-model state stored under ``model._capability_state``."""
        del model
        return {}

    def enabled_for(self, model) -> bool:
        """Whether this capability should be considered for ``model``."""
        del model
        return True

    @abstractmethod
    def handles(self, model, event, **kwargs) -> bool:
        """Return True when this capability should process ``event``."""
        ...

    @abstractmethod
    def process(self, model, event, **kwargs):
        """Process ``event`` for ``model``."""
        ...
