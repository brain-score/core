"""Public names of the Brain-Score subject contract and its events.

The concrete ``BrainScoreModel``, its unit selectors and its capability
registry live in the unified ``brainscore`` package.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

from .events import (
    InputEvent,
    OutputEvent,
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
    dispatch_metric,
    output_event_kind,
)
from .contract import Subject, TaskContext, UnifiedModel
