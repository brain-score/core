"""
Extraction-plan memory pre-check for Brain-Score.

Estimates whether a (model, benchmark) pair will fit in memory BEFORE the run,
so a benchmark can't OOM after hours of computation. The estimate is *computed*
from a plan — ``n_stimuli x n_features x dtype`` for the activation matrix plus
an analytic metric-workspace term — not extrapolated from a measured probe run.

Why not a measured probe (the previous design): running one probe extraction and
scaling its RSS to the full run is unsound. The benchmark frequently calls
``process()`` on a different / filtered stimulus set than the probe (e.g. Lahner
expands clips to frames), the plain-preprocessor path isn't cached, and a probe
crash disabled the guard entirely (fail-open). The only thing a probe genuinely
tells us is the recording layer's feature width — which is *stimulus-invariant*,
so a single-stimulus probe (or an explicit ``feature_dim`` / ``expected_feature_dim``
declaration) suffices, and everything else is arithmetic.

Cross-platform: CUDA (torch.cuda), MPS (psutil), CPU (psutil).
brainscore_core stays free of heavy dependencies -- torch is optional.
"""

import logging
import resource
import sys
import warnings
from typing import Optional, TYPE_CHECKING

from .execution_plan import ExecutionPlan, _positive_int as _validate_positive_int

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .model_interface import UnifiedModel


class MemoryError(Exception):
    """Raised when estimated memory exceeds available resources."""
    pass


def get_available_memory() -> int:
    """
    Returns available memory in bytes. Cross-platform.

    - NVIDIA GPU: uses torch.cuda.mem_get_info()
    - Apple Silicon MPS: uses psutil (MPS shares unified memory)
    - CPU only: uses psutil
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import psutil
            return psutil.virtual_memory().available
    except ImportError:
        pass

    import psutil
    return psutil.virtual_memory().available


def get_host_available_memory() -> int:
    """Available HOST (system) RAM in bytes, always via psutil.

    The extraction-plan estimate models host-side numpy/sklearn arrays (the
    activation matrix and the metric design matrix), so it must be compared
    against host RAM — not GPU VRAM. ``get_available_memory`` returns free VRAM
    when CUDA is present, which is the wrong budget for the host-side plan (and
    the plan does NOT model the GPU forward-pass working set — that is a separate,
    unmodeled OOM failure mode).
    """
    import psutil
    return psutil.virtual_memory().available


def get_peak_memory() -> int:
    """
    Returns peak memory usage in bytes since last reset.
    Cross-platform.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except ImportError:
        pass

    import psutil
    return psutil.Process().memory_info().rss


def reset_peak_memory() -> None:
    """Reset peak memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _get_peak_rss() -> int:
    """Cross-platform peak RSS (high-water mark) in bytes.

    Uses resource.getrusage which tracks the maximum RSS since process start.
    This captures transient spikes that psutil.Process().memory_info().rss misses
    (RSS is point-in-time, peak RSS is the all-time max).

    Linux: ru_maxrss is in KB. macOS: ru_maxrss is in bytes.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'linux':
        return maxrss * 1024
    return maxrss


def _detect_metric_category(benchmark) -> str:
    """
    Detect the benchmark's metric category from its identifier and attributes.

    Returns one of: 'ridgecv', 'ridge', 'pls', 'rsa', 'behavioral'.
    """
    bench_id = getattr(benchmark, 'identifier', '')
    bench_id_lower = bench_id.lower()

    # Check identifier suffixes (most reliable signal)
    if 'ridgecv' in bench_id_lower:
        return 'ridgecv'
    if 'ridge' in bench_id_lower:
        return 'ridge'
    if 'rdm' in bench_id_lower or 'rsa' in bench_id_lower:
        return 'rsa'
    if 'pls' in bench_id_lower:
        return 'pls'

    # Check for neural vs behavioral by looking for assemblies
    has_assembly = (hasattr(benchmark, '_assembly')
                    or hasattr(benchmark, 'train_assembly'))
    if not has_assembly:
        return 'behavioral'

    # Default: assume PLS (most common neural metric)
    return 'pls'


def _get_n_targets(benchmark) -> int:
    """Read actual neuroid/voxel count from the benchmark's assembly.

    If the benchmark fits per-subject (alpha_coord), returns the per-subject
    count (total / n_subjects) since subjects are fitted sequentially.
    """
    n_targets = None
    assembly = None
    for attr in ('_assembly', 'train_assembly', 'test_assembly'):
        assembly = getattr(benchmark, attr, None)
        if assembly is not None:
            sizes = getattr(assembly, 'sizes', None)
            if sizes is not None and 'neuroid' in sizes:
                n_targets = sizes['neuroid']
                break
    if n_targets is None:
        n_targets = getattr(benchmark, 'n_targets', 100)
        if n_targets is None:
            n_targets = 100
    n_targets = int(n_targets)

    # If benchmark fits per-subject, the metric only processes one subject's
    # neuroids at a time. Approximate per-subject count by dividing total.
    alpha_coord = getattr(benchmark, 'alpha_coord', None)
    if alpha_coord and assembly is not None:
        try:
            import numpy as np
            n_subjects = len(np.unique(assembly[alpha_coord].values))
            if n_subjects > 1:
                n_targets = int(np.ceil(n_targets / n_subjects))
        except Exception:
            pass  # fall back to full n_targets

    return int(n_targets)


def _get_n_stimuli(benchmark) -> int:
    """Get total stimulus count from the benchmark."""
    # Direct attribute
    stimulus_set = getattr(benchmark, 'stimulus_set', None)
    if stimulus_set is not None and len(stimulus_set) > 0:
        return len(stimulus_set)
    # From assemblies
    for attr in ('_assembly', 'train_assembly'):
        assembly = getattr(benchmark, attr, None)
        if assembly is not None:
            stim = getattr(assembly, 'stimulus_set', None)
            if stim is not None:
                return len(stim)
    return 0


def _get_n_alphas(benchmark) -> int:
    """Read the RidgeCV alpha grid size from the benchmark or its metric."""
    # Check for ALPHA_LIST or alpha_list on the benchmark's module
    for attr in ('_similarity_metric', 'similarity_metric', 'metric'):
        metric = getattr(benchmark, attr, None)
        if metric is not None:
            alphas = getattr(metric, 'alphas', None)
            if alphas is not None:
                return len(alphas)
    # Default RidgeCV grid in brain-score benchmarks
    return 115


SAFETY_FACTORS = {
    'pls': 1.5,
    'ridge': 2.0,
    'ridgecv': 3.0,
    'rsa': 2.0,
    'behavioral': 1.2,
}


def estimate_metric_memory(benchmark, n_features: Optional[int] = None,
                           n_stimuli: Optional[int] = None) -> int:
    """
    Estimate memory required for the benchmark's metric computation.

    Metric-aware: reads the actual target count and metric type from the
    benchmark to produce realistic estimates instead of using hardcoded defaults.

    Memory scaling by metric type:
    - PLS:      O(S * F * n_components + S * T)         ~small
    - Ridge:    O(F^2 + F * T + S * T)                  ~moderate
    - RidgeCV:  O(F^2 * A + S * T * A)                  ~large (A=n_alphas)
    - RSA:      O(S^2)                                   ~quadratic in stimuli
    - Behavioral: negligible

    Args:
        benchmark: the benchmark to estimate for.
        n_features: actual model feature dimension from probe. If None,
            falls back to benchmark.expected_feature_dim or 1000. This is
            critical because models with region_layer_map skip PCA, so
            features can be 9K (alexnet) to 148K+ (ViT-L).
        n_stimuli: number of observations the metric fits over. If None, reads
            the raw stimulus-set length — but callers that know the real
            processed cardinality (row-expanding benchmarks) must pass it, or the
            metric term silently uses the raw count while extraction uses the
            expanded one.

    Returns estimate in bytes.
    """
    category = _detect_metric_category(benchmark)
    if n_stimuli is None:
        n_stimuli = _get_n_stimuli(benchmark)
    n_targets = _get_n_targets(benchmark)
    if n_features is None:
        n_features = getattr(benchmark, 'expected_feature_dim', 1000)

    if category == 'behavioral' or n_stimuli == 0:
        return 0

    if category == 'rsa':
        # Dissimilarity matrix: S x S float64, plus model RDM
        rdm_size = n_stimuli * n_stimuli * 8
        return rdm_size * 2  # neural RDM + model RDM

    # sklearn converts float32 inputs to float64 internally.
    # All metric arrays use 8 bytes per element.
    bytes_per_element = 8

    # Shared: design and target matrices
    design = n_stimuli * n_features * bytes_per_element
    target = n_stimuli * n_targets * bytes_per_element

    if category == 'pls':
        # PLS.fit: centered copy of X + deflated copy + loadings/components
        # Cross-validation (10-fold): sortby creates a copy of train split
        n_components = 25
        centered = design  # centered copy of X
        components = n_features * n_components * bytes_per_element
        coef = n_features * n_targets * bytes_per_element  # coefficient matrix
        cv_sortby = int(n_stimuli * 0.9) * n_features * bytes_per_element  # sortby copy
        return design + target + centered + components + coef + cv_sortby

    if category == 'ridge':
        # Ridge: sklearn uses dual formulation when n_samples < n_features,
        # so Gram is min(S, F)², not F².
        # coef_dual = (S, T_per_subject), coef = X.T @ coef_dual = (F, T_per_subject)
        gram_dim = min(n_stimuli, n_features)
        gram = gram_dim * gram_dim * bytes_per_element
        centered = design  # centered copy of X
        coef = n_features * n_targets * bytes_per_element
        return gram + design + target + centered + coef

    if category == 'ridgecv':
        # RidgeCV: dual Gram + LOO predictions across A alphas
        n_alphas = _get_n_alphas(benchmark)
        gram_dim = min(n_stimuli, n_features)
        gram = gram_dim * gram_dim * bytes_per_element
        centered = design
        loo_predictions = n_stimuli * n_targets * n_alphas * bytes_per_element
        return gram + design + target + centered + loo_predictions

    # Fallback
    return design + target + design * 3


# Activations are extracted in float32; sklearn upcasts to float64 inside the
# metric (that upcast is accounted for separately in estimate_metric_memory).
ACTIVATION_DTYPE_BYTES = 4
# The forward-pass transient (raw inputs + per-batch intermediates held while the
# feature matrix accumulates) on top of the held activation matrix. A ceiling, not
# a measurement — ponytail: bump if a model's per-batch working set dwarfs its
# output matrix (very wide intermediate layers).
EXTRACTION_TRANSIENT_FACTOR = 1.5
# Slack over the analytic metric term for allocator fragmentation + temporaries.
PIPELINE_OVERHEAD = 1.2


def _get_benchmark_stimulus_set(benchmark):
    """Resolve the benchmark's stimulus set (direct attr or via an assembly)."""
    stimulus_set = getattr(benchmark, 'stimulus_set', None)
    if stimulus_set is not None:
        return stimulus_set
    for attr in ('_assembly', 'train_assembly'):
        assembly = getattr(benchmark, attr, None)
        if assembly is not None:
            stimulus_set = getattr(assembly, 'stimulus_set', None)
            if stimulus_set is not None:
                return stimulus_set
    return None


def _get_execution_plan(benchmark) -> Optional[ExecutionPlan]:
    """Return the benchmark's declared ExecutionPlan (attribute or zero-arg
    method/property), or None if it declares none."""
    plan = getattr(benchmark, 'execution_plan', None)
    if plan is None:
        return None
    if callable(plan):
        plan = plan()
    if plan is not None and not isinstance(plan, ExecutionPlan):
        raise TypeError(
            f"benchmark.execution_plan must be an ExecutionPlan (or return one), "
            f"got {type(plan).__name__}")
    return plan


def _probe_regions(model, benchmark):
    """Regions to try recording for the probe, in order.

    The benchmark's own ``.region`` if it exposes one; otherwise the model's
    ``region_layer_map`` keys tried ONE AT A TIME (never all at once — recording
    every region would sum several layers and inflate the feature width, and would
    request a modality a single-tower wrapper can't serve). ``[None]`` means the
    model has no recording concept (e.g. a behavioral candidate) — probe as-is.
    """
    region = getattr(benchmark, 'region', None)
    if region is not None:
        return [region]
    region_map = getattr(model, 'region_layer_map', None)
    if region_map:
        return list(region_map.keys())
    return [None]


def _probe_feature_dim(model, benchmark, stimulus_set):
    """Discover the recording-layer feature width from ONE stimulus.

    Feature width is stimulus-invariant (the layer's channel count), so one
    stimulus gives it without extrapolating any memory measurement. Records a
    SINGLE region at a time (see _probe_regions) and returns the first that yields
    a readable width. Returns (None, None) if none do — e.g. a candidate that only
    configures recording inside the benchmark's ``__call__`` and rejects a bare
    probe; the caller then skips with a loud warning (the guard is OFF, not "safe").

    Returns (n_features, region_used) or (None, None).
    """
    try:
        one = stimulus_set.iloc[:1] if hasattr(stimulus_set, 'iloc') else stimulus_set[:1]
    except Exception:
        one = stimulus_set
    can_record = hasattr(model, 'start_recording')
    time_bins = getattr(benchmark, 'timebins', None)
    last_error = None
    for region in _probe_regions(model, benchmark):
        if region is not None and can_record:
            try:
                model.start_recording(region, time_bins=time_bins)
            except Exception as e:
                last_error = e
                continue
        try:
            result = model.process(one)
        except Exception as e:
            last_error = e
            continue
        shape = getattr(result, 'shape', None)
        if shape is not None and len(shape) >= 2:
            return int(shape[-1]), region
        last_error = f"probe returned shape {shape!r} (need >=2 dims)"
    logger.warning(
        "Memory pre-flight could NOT run: no region yielded a readable feature "
        "width (last error: %s), so no estimate was made and the OOM guard is OFF "
        "for this run. Pass feature_dim=... to estimate anyway, or check_mem=False "
        "to silence.",
        last_error,
        exc_info=last_error if isinstance(last_error, BaseException) else None,
    )
    return None, None


def _resolve_feature_dim(model, benchmark, feature_dim, stimulus_set):
    """Resolve the model's per-stimulus feature width, preferring a declaration
    over a probe. Returns (n_features, source); n_features is None if unresolved."""
    if feature_dim is not None:
        return _validate_positive_int(feature_dim, 'feature_dim'), 'feature_dim argument'
    for obj, attr in ((benchmark, 'expected_feature_dim'), (model, 'feature_dim')):
        declared = getattr(obj, attr, None)
        if declared is not None:  # is-not-None, so a declared 0 errors (not skipped)
            name = f'{type(obj).__name__}.{attr}'
            return _validate_positive_int(declared, name), name
    dim, _region = _probe_feature_dim(model, benchmark, stimulus_set)
    return dim, 'single-stimulus probe'


def _get_n_presentations(benchmark, stimulus_set):
    """How many rows the model will actually process, and whether it is declared.

    ``len(stimulus_set)`` is only a rough proxy: benchmarks that expand each row
    (video -> per-TR frames in Algonauts / Lahner) process MORE, while ones that
    filter to relevant event IDs (time-resolved Lahner) process FEWER. A benchmark
    can declare the true count via ``expected_n_presentations``; otherwise we use
    the raw length and flag the whole estimate approximate.

    Returns (n_presentations, declared: bool).
    """
    declared = getattr(benchmark, 'expected_n_presentations', None)
    if declared is not None:
        return _validate_positive_int(declared, 'expected_n_presentations'), True
    return len(stimulus_set), False


def check_memory(
    model: 'UnifiedModel',
    benchmark,
    safety_factor: Optional[float] = None,
    feature_dim: Optional[int] = None,
) -> None:
    """
    Memory pre-flight with two paths:

    RELIABLE — when the benchmark declares an ``ExecutionPlan`` (attribute or
    zero-arg method; see brainscore_core.execution_plan). The plan states the real
    extraction cardinality, per-row feature width, metric observation count, metric
    (post-compression) feature width, dtype, and host/device placement, so the
    host-side estimate is exact — no probe, no guessing. Device VRAM for the forward
    pass remains a separate budget the pre-flight does not model.

    BEST-EFFORT — when no plan is declared: raise when the estimated peak exceeds
    host RAM (a strong signal, not a certainty — the estimate can over- or
    under-count), and otherwise be honest that a "fit" is not a safety guarantee.
    Every result on this path is flagged APPROXIMATE. Both key inputs are
    approximate in *both* directions:
      - ``n_presentations``: ``benchmark.expected_n_presentations`` if declared,
        else ``len(stimulus_set)`` — which over-counts benchmarks that filter rows
        (time-resolved Lahner keeps only relevant event IDs) and under-counts ones
        that expand rows (video->per-TR frames). The metric fits over this same
        count, which may itself differ from the extraction count (a benchmark can
        aggregate frames before the metric) — a distinction only the future API
        captures.
      - ``n_features``: ``feature_dim`` arg / ``benchmark.expected_feature_dim`` /
        ``model.feature_dim`` / a single-stimulus probe. The probed width is the
        RAW recording width; benchmarks that PCA/compress before the metric
        (Algonauts 38K->1K, TR-Lahner->512) see less, so the metric term over-
        counts there. Temporal ``S x T x F`` outputs are not modelled — the held
        matrix under-counts by T.

    The estimate is compared against HOST RAM only (the plan is host-side
    numpy/sklearn); the GPU forward-pass working set is a separate, unmodelled
    failure mode. Because the estimate can over- OR under-count, an over-budget
    result is a strong signal but NOT a certainty — it may be a false rejection,
    so the raise is override-able (``check_mem=False``) and every result is logged
    APPROXIMATE. Skips (loudly) when no feature width can be resolved — e.g. a
    candidate that only configures recording inside the benchmark's ``__call__``.

    :param model: the model to check
    :param benchmark: the benchmark to check against
    :param safety_factor: unused, kept for API compatibility.
    :param feature_dim: optional explicit per-stimulus feature width (positive
        int); skips the probe entirely (the fully-declarative path).
    """
    # Host RAM, not VRAM: the plan models host-side arrays (see get_host_available_memory).
    available = get_host_available_memory()
    category = _detect_metric_category(benchmark)

    import psutil
    baseline_rss = psutil.Process().memory_info().rss

    stimulus_set = _get_benchmark_stimulus_set(benchmark)
    if stimulus_set is None or len(stimulus_set) == 0:
        logger.info("Memory check not applicable: benchmark has no stimulus_set "
                    "(nothing to extract).")
        return

    plan = _get_execution_plan(benchmark)
    if plan is not None and feature_dim is None:
        # RELIABLE path: the benchmark declared its execution shape, so nothing is
        # probed or guessed. Extraction and metric counts/widths can genuinely
        # differ (aggregation before the metric; feature compression).
        source = 'ExecutionPlan'
        approximate = False
        cardinality_declared = True
        on_device = plan.extraction_on_device
        n_presentations = plan.n_extraction_presentations
        feature_width = plan.feature_width
        metric_obs = plan.resolved_metric_observations
        metric_fw = plan.resolved_metric_feature_width
        dtype_bytes = plan.activation_dtype_bytes
    else:
        # BEST-EFFORT path: probe / declared feature dim + raw (or expected) count.
        n_features, source = _resolve_feature_dim(
            model, benchmark, feature_dim, stimulus_set)
        if n_features is None:
            return  # unresolved feature dim; _probe_feature_dim already warned loudly
        approximate = True
        on_device = False
        n_presentations, cardinality_declared = _get_n_presentations(
            benchmark, stimulus_set)
        feature_width = metric_fw = n_features
        metric_obs = n_presentations
        dtype_bytes = ACTIVATION_DTYPE_BYTES

    n_targets = _get_n_targets(benchmark)

    # Extraction: the held activation matrix (float32 on host). When the plan says
    # extraction runs on an accelerator, the raw matrix is device-side and excluded
    # from the host budget (only the metric's host arrays count; device VRAM is a
    # separate, unmodelled budget). The transient factor is a coarse scalar on the
    # held matrix and is NOT an upper bound (a wide-input video model spikes above
    # output width) — advisory only.
    held_activations = 0 if on_device else n_presentations * feature_width * dtype_bytes
    extraction_transient = int(held_activations * EXTRACTION_TRANSIENT_FACTOR)

    # Metric workspace (float64 inside sklearn). metric_obs / metric_fw may differ
    # from the extraction counts (aggregation before the metric; feature compression).
    estimated_metric_memory = estimate_metric_memory(
        benchmark, n_features=metric_fw, n_stimuli=metric_obs)
    if category != 'behavioral':
        estimated_ceiling_memory = estimate_metric_memory(
            benchmark, n_features=n_targets, n_stimuli=metric_obs)
    else:
        estimated_ceiling_memory = 0
    metric_total = (estimated_metric_memory + estimated_ceiling_memory) * PIPELINE_OVERHEAD

    # The metric runs while the extracted activations are still resident.
    peak_from_extraction = baseline_rss + extraction_transient
    peak_from_metric = baseline_rss + held_activations + metric_total
    total_estimated = max(peak_from_extraction, peak_from_metric)

    total_system = available + baseline_rss
    if total_estimated > total_system:
        if approximate:
            detail = ("This estimate is APPROXIMATE and may over-count (feature "
                      "compression, row filtering) — if it is a false rejection, "
                      "set check_mem=False.")
        else:
            detail = ("This estimate is RELIABLE for host RAM via the declared "
                      "ExecutionPlan.") + (
                " Device/VRAM for the forward pass is a separate budget, not "
                "checked." if on_device else "")
        raise MemoryError(
            f"Estimated peak for '{model.identifier}' on "
            f"'{getattr(benchmark, 'identifier', 'unknown')}': "
            f"{total_estimated / 1e9:.1f} GB "
            f"(extraction peak: {peak_from_extraction / 1e9:.1f} GB, "
            f"metric peak: {peak_from_metric / 1e9:.1f} GB, "
            f"metric [{category}]: {estimated_metric_memory / 1e9:.1f} GB, "
            f"ceiling: {estimated_ceiling_memory / 1e9:.1f} GB, "
            f"n_features: {feature_width} [{source}], "
            f"n_presentations: {n_presentations}"
            f"{' [declared]' if cardinality_declared else ' [approx]'}). "
            f"Available host RAM: {total_system / 1e9:.1f} GB. " + detail
        )

    utilization = total_estimated / total_system if total_system > 0 else 0
    peak_source = "extraction" if peak_from_extraction >= peak_from_metric else "metric"
    if approximate:
        caveat = ("APPROXIMATE: stimulus expansion/filtering, temporal bins, "
                  "feature compression, and GPU memory are not modelled, so this "
                  "is not a guarantee either way. Declare an ExecutionPlan for a "
                  "reliable check.")
    else:
        caveat = ("RELIABLE for host RAM via the declared ExecutionPlan" +
                  (" (device/VRAM for the forward pass not modelled)."
                   if on_device else "."))
    logger.info(
        f"Memory estimate for '{model.identifier}' on "
        f"'{getattr(benchmark, 'identifier', 'unknown')}' "
        f"[{category}, {feature_width} features via {source}, "
        f"{n_presentations} presentations]: {total_estimated / 1e9:.1f} GB peak "
        f"(held activations: {held_activations / 1e9:.1f} GB; "
        f"metric: {estimated_metric_memory / 1e9:.1f} GB; "
        f"bottleneck: {peak_source}) — {total_system / 1e9:.1f} GB host RAM "
        f"({utilization:.0%} utilization). {caveat}"
    )
    if utilization > 0.8:
        warnings.warn(
            f"Memory utilization for '{model.identifier}' estimated at "
            f"{utilization:.0%} of available host RAM ({total_estimated / 1e9:.1f} / "
            f"{available / 1e9:.1f} GB). OOM risk is elevated.",
            ResourceWarning,
        )
