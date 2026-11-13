from .metrics import Metric, Score
from .benchmarks import Benchmark
from .contract import (
    TaskContext,
    Subject,
    UnifiedModel,  # compatibility base for the legacy typed interface
)
from .events import (
    Selection,
    Perturbation,
    StateChange,
    PerturbationApplied,
    InputEvent,
    OutputEvent,
)
from .compatibility import (
    CompatibilityError, CompatibilityWarning, check_compatibility,
)
from .score_status import STATUS_NA, na_score, score_or_na
from .memory import MemoryError, check_memory, get_available_memory
from . import io_catalog
from .io_catalog import CatalogEntry
from . import environment_check

# Warn about dependency drift before scoring.
environment_check.warn_on_drift()
