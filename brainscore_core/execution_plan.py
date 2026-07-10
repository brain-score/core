"""ExecutionPlan: a benchmark's declaration of its memory-relevant execution shape.

``check_memory`` uses this to ground its estimate in DECLARED shapes instead of
probing one stimulus and guessing the cardinality. It removes the order-of-
magnitude errors (a benchmark that processes 162k per-TR frames, not 300 videos;
a metric that sees a compressed / windowed width, not the raw one).

It does NOT make the estimate an oracle: the peak model on top of the declared
shapes is still heuristic (a coarse extraction-transient factor, sklearn-workspace
formulas, an allocator slack factor). So a plan makes the estimate DECLARED-grounded
and much better, not exact. Device VRAM for the forward pass is out of scope.

Kept dependency-free (dataclasses + typing only) so ``brainscore_core`` stays light.
"""
from dataclasses import dataclass
from typing import Any, Optional

_VALID_METRIC_CATEGORIES = {'pls', 'ridge', 'ridgecv', 'rsa', 'behavioral'}


def _positive_int(value, name):
    """Return value as a positive int, rejecting bools/floats/zero/negatives.

    ``operator.index`` accepts numpy integers but rejects floats. ``bool`` is
    rejected explicitly (``operator.index(True) == 1`` would otherwise slip
    ``True`` through as 1)."""
    import operator
    # reject Python bool AND numpy bool_ (both pass operator.index as 0/1)
    if isinstance(value, bool) or type(value).__name__ == 'bool_':
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        ivalue = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


@dataclass(frozen=True)
class ExecutionPlan:
    """Memory-relevant execution shape of a (model, benchmark) run.

    Declare this on a benchmark (``self.execution_plan = ExecutionPlan(...)``, or a
    zero-arg method/property returning one). Frozen: validated once at construction
    and immutable thereafter, so a consumer can't be handed a mutated invalid plan.

    Fields:
    - ``n_extraction_presentations``: rows actually held in the activation matrix,
      AFTER row expansion (video -> per-TR frames) or filtering (relevant-event
      IDs), and INCLUDING temporal multiplicity (S clips x T bins -> S*T). Fully
      benchmark-known; this is what ``len(stimulus_set)`` gets wrong.
    - ``feature_width``: per-row RAW recording width. Optional — the ONE
      model-dependent quantity, and stimulus-invariant, so leave it None and the
      pre-flight probes for it (using ``recording_target`` / ``probe_stimuli``
      below). Declare it (or pass ``feature_dim`` to check_memory) to skip probing.
    - ``metric_observations``: rows the metric actually fits over, AFTER any
      per-run sample exclusions / aggregation. Defaults to
      ``n_extraction_presentations``; declare it when they genuinely differ.
    - ``metric_feature_width``: feature width the metric SEES after PCA /
      compression / windowing (e.g. window x SVD-cap). May be an upper bound when
      it depends on the model's raw width; the estimate is DECLARED, not exact, so
      an upper bound is fine. Defaults to the resolved raw width.
    - ``metric_category``: override the identifier-based metric detection
      (``'pls' | 'ridge' | 'ridgecv' | 'rsa' | 'behavioral'``) when the identifier
      is misleading (e.g. an id without 'ridge' that scores with ridge).
    - ``runs_ceiling_metric``: whether the benchmark actually runs a neural-to-
      neural ceiling metric (many use a precomputed constant ceiling and don't).
      When False the ceiling workspace term is omitted.
    - ``recording_target``: region(s) to ``start_recording`` before the probe, so
      it records the benchmark's ACTUAL target layer (not just the first that
      happens to work). Only used when ``feature_width`` is probed.
    - ``probe_stimuli``: the input to run the probe through — a small stimulus set
      of the SAME shape the benchmark actually processes (e.g. one image frame for
      a benchmark that expands videos to frames), so an image-only model doesn't
      choke on a raw video row. Only used when ``feature_width`` is probed;
      defaults to the benchmark's stimulus set.
    - ``activation_dtype_bytes``: bytes per activation element (float32 = 4).
    """
    n_extraction_presentations: int
    feature_width: Optional[int] = None
    metric_observations: Optional[int] = None
    metric_feature_width: Optional[int] = None
    metric_category: Optional[str] = None
    runs_ceiling_metric: bool = True
    recording_target: Optional[Any] = None
    probe_stimuli: Optional[Any] = None
    activation_dtype_bytes: int = 4

    def __post_init__(self):
        # frozen dataclass: normalize/validate via object.__setattr__
        _set = object.__setattr__
        _set(self, 'n_extraction_presentations',
             _positive_int(self.n_extraction_presentations, 'n_extraction_presentations'))
        if self.feature_width is not None:
            _set(self, 'feature_width', _positive_int(self.feature_width, 'feature_width'))
        if self.metric_observations is not None:
            _set(self, 'metric_observations',
                 _positive_int(self.metric_observations, 'metric_observations'))
        if self.metric_feature_width is not None:
            _set(self, 'metric_feature_width',
                 _positive_int(self.metric_feature_width, 'metric_feature_width'))
        _set(self, 'activation_dtype_bytes',
             _positive_int(self.activation_dtype_bytes, 'activation_dtype_bytes'))
        if not isinstance(self.runs_ceiling_metric, bool):
            raise ValueError(
                f"runs_ceiling_metric must be a bool, got {self.runs_ceiling_metric!r}")
        if (self.metric_category is not None
                and self.metric_category not in _VALID_METRIC_CATEGORIES):
            raise ValueError(
                f"metric_category must be one of {sorted(_VALID_METRIC_CATEGORIES)} "
                f"or None, got {self.metric_category!r}")

    @property
    def resolved_metric_observations(self) -> int:
        """Metric row count, defaulting to the extraction presentation count."""
        return self.metric_observations or self.n_extraction_presentations

    def resolved_metric_feature_width(self, raw_feature_width: int) -> int:
        """Metric feature width: the declared compression/windowing target, else the
        raw (probed or declared) recording width passed in."""
        return self.metric_feature_width or raw_feature_width
