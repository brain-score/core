from .metrics import Metric, Score
from .benchmarks import Benchmark
from .model_interface import (
    TaskContext,
    Subject,
    UnifiedModel,  # compatibility base for the legacy typed interface
    BrainScoreModel,
    UnitSelector,
    LayerSelector,
    CompositeSelector,
    Selection,
    Perturbation,
    StateChange,
    PerturbationApplied,
    UnitSelection,
    FunctionalSelection,
    RandomSelection,
    IndexSelection,
    EnvironmentStep,
    EnvironmentResponse,
    Message,
    InputEvent,
    OutputEvent,
    output_event_kind,
    dispatch_metric,
)
from ._evaluation.compatibility import (
    CompatibilityError, CompatibilityWarning, check_compatibility, check_io_catalog,
)
from ._evaluation.score_status import STATUS_NA, na_score, score_or_na
from ._evaluation.memory import MemoryError, check_memory, get_available_memory
from . import io_catalog
from .io_catalog import CatalogEntry

# Established imports share the actual implementation module and its state.
import sys as _sys
from ._model import behavior
from ._model import behavioral
from ._model import capability_config
from ._model import dispatch
from ._model import perturbation
from ._model import recording
from ._sessions import temporal
from ._sessions import helpers as streaming_helpers
from ._sessions import view as session_view
from ._evaluation import compatibility
from ._evaluation import execution_plan
from ._evaluation import memory
from ._evaluation import preflight
from ._evaluation import score_metadata
from ._evaluation import score_status
from ._evaluation import legacy_facade
from ._evaluation import nulls
from ._data import assembly_builder
from ._data import clock
from ._data import multimodal
from ._data import text
from ._runtime import environment_check
from ._runtime import extraction_cache
from ._runtime import hf_compat
from ._runtime import extensions

for _name, _module in (
    ('behavior', behavior),
    ('behavioral', behavioral),
    ('capability_config', capability_config),
    ('dispatch', dispatch),
    ('perturbation', perturbation),
    ('recording', recording),
    ('temporal', temporal),
    ('streaming_helpers', streaming_helpers),
    ('session_view', session_view),
    ('compatibility', compatibility),
    ('execution_plan', execution_plan),
    ('memory', memory),
    ('preflight', preflight),
    ('score_metadata', score_metadata),
    ('score_status', score_status),
    ('legacy_facade', legacy_facade),
    ('nulls', nulls),
    ('assembly_builder', assembly_builder),
    ('clock', clock),
    ('multimodal', multimodal),
    ('text', text),
    ('environment_check', environment_check),
    ('extraction_cache', extraction_cache),
    ('hf_compat', hf_compat),
    ('extensions', extensions),
    ('_env_check', environment_check),
):
    _sys.modules[__name__ + '.' + _name] = _module
del _sys, _name, _module

# Warn about dependency drift before scoring.
environment_check.warn_on_drift()
