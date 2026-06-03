from .metrics import Metric, Score
from .benchmarks import Benchmark
from .model_interface import (
    TaskContext,
    Subject,
    UnifiedModel,  # deprecated alias for Subject
    BrainScoreModel,
    UnitSelector,
    LayerSelector,
    CompositeSelector,
)
from .compatibility import CompatibilityError, CompatibilityWarning, check_compatibility, check_io_catalog
from .memory import MemoryError, check_memory, get_available_memory
from . import io_catalog
from .io_catalog import CatalogEntry