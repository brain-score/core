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
