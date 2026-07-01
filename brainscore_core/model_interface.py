"""Compatibility facade for the Brain-Score model interface.

The implementation lives in smaller modules, but all names historically
importable from ``brainscore_core.model_interface`` remain importable here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

from .events import (
    CameraFrame,
    EnvironmentResponse,
    EnvironmentStep,
    InputEvent,
    Message,
    OutputEvent,
    Perturbation,
    PerturbationApplied,
    Proprioception,
    Selection,
    StateChange,
    dispatch_metric,
    output_event_kind,
)
from .selection import (
    CompositeSelector,
    FunctionalSelection,
    IndexSelection,
    LayerSelector,
    RandomSelection,
    UnitSelection,
    UnitSelector,
    _promote_to_selector,
)
from .contract import Subject, TaskContext, UnifiedModel
from .brainscore_model import BrainScoreModel
