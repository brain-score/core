"""ExecutionPlan: a benchmark's declaration of its memory-relevant execution shape.

``check_memory`` uses this to produce a RELIABLE host-side estimate instead of
probing one stimulus and guessing. A benchmark that expands/filters its stimuli,
compresses features before the metric, or runs extraction on an accelerator can
declare exactly what the pre-flight needs — the four quantities that
``len(stimulus_set)`` and a single-stimulus probe cannot reveal.

Kept dependency-free (dataclasses + typing only) so ``brainscore_core`` stays light.
"""
from dataclasses import dataclass
from typing import Optional


def _positive_int(value, name):
    """Return value as a positive int, rejecting floats/zero/negatives.

    ``operator.index`` accepts numpy integers but rejects floats (1.9 does NOT
    silently truncate to 1)."""
    import operator
    try:
        ivalue = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


@dataclass
class ExecutionPlan:
    """Memory-relevant execution shape of a (model, benchmark) run.

    Declare this on a benchmark (``self.execution_plan = ExecutionPlan(...)``, or a
    zero-arg method/property returning one) to turn the memory pre-flight from a
    best-effort guess into a reliable host-side estimate.

    Fields:
    - ``n_extraction_presentations``: rows actually held in the activation matrix,
      AFTER row expansion (video -> per-TR frames) or filtering (relevant-event
      IDs), and INCLUDING temporal multiplicity (S clips x T bins -> S*T). This is
      what ``len(stimulus_set)`` gets wrong.
    - ``feature_width``: per-row extracted feature width (the RAW recording width
      the activation matrix is stored at).
    - ``metric_observations``: rows the metric fits over. Defaults to
      ``n_extraction_presentations``; declare separately when the benchmark
      aggregates (e.g. mean over frames / time) before the metric, so the two
      counts genuinely differ.
    - ``metric_feature_width``: feature width the metric SEES after any PCA /
      compression (e.g. Algonauts 38K -> 1K). Defaults to ``feature_width``.
    - ``activation_dtype_bytes``: bytes per activation element (float32 = 4).
    - ``extraction_on_device``: True if the activation matrix lives on an
      accelerator (GPU) rather than host RAM. The host-side estimate then excludes
      the raw matrix (only the metric's host arrays count); device VRAM is a
      separate budget the pre-flight deliberately does not model.
    """
    n_extraction_presentations: int
    feature_width: int
    metric_observations: Optional[int] = None
    metric_feature_width: Optional[int] = None
    activation_dtype_bytes: int = 4
    extraction_on_device: bool = False

    def __post_init__(self):
        self.n_extraction_presentations = _positive_int(
            self.n_extraction_presentations, 'n_extraction_presentations')
        self.feature_width = _positive_int(self.feature_width, 'feature_width')
        if self.metric_observations is not None:
            self.metric_observations = _positive_int(
                self.metric_observations, 'metric_observations')
        if self.metric_feature_width is not None:
            self.metric_feature_width = _positive_int(
                self.metric_feature_width, 'metric_feature_width')
        self.activation_dtype_bytes = _positive_int(
            self.activation_dtype_bytes, 'activation_dtype_bytes')
        if not isinstance(self.extraction_on_device, bool):
            raise ValueError(
                f"extraction_on_device must be a bool, got "
                f"{self.extraction_on_device!r}")

    @property
    def resolved_metric_observations(self) -> int:
        """Metric row count, defaulting to the extraction presentation count."""
        return self.metric_observations or self.n_extraction_presentations

    @property
    def resolved_metric_feature_width(self) -> int:
        """Metric feature width, defaulting to the raw extraction width."""
        return self.metric_feature_width or self.feature_width
