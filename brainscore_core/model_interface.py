"""The subject contract and its events under an older name.

A legacy vision plugin test imports from here, so the name stays. New code
imports from ``brainscore_core.contract`` and ``brainscore_core.events``.
"""

from .events import (
    InputEvent,
    OutputEvent,
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
)
from .contract import Subject, TaskContext, UnifiedModel
