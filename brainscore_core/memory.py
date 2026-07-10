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


def estimate_metric_memory(benchmark, n_features: Optional[int] = None) -> int:
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

    Returns estimate in bytes.
    """
    category = _detect_metric_category(benchmark)
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


def _probe_feature_dim(model, stimulus_set) -> Optional[int]:
    """Discover the recording layer's feature width from ONE stimulus.

    Feature width is stimulus-invariant (it's the layer's channel count), so a
    single stimulus gives the true dimension without extrapolating any memory
    measurement. Returns None if the model can't process the stimulus — in which
    case the full benchmark (same ``process`` path) will fail fast too, so the
    memory guard has nothing to protect and the caller skips rather than fabricating
    a number.
    """
    try:
        one = stimulus_set.iloc[:1] if hasattr(stimulus_set, 'iloc') else stimulus_set[:1]
    except Exception:
        one = stimulus_set
    try:
        result = model.process(one)
    except Exception as e:
        logger.warning(
            f"Memory pre-flight could NOT run: the model failed to process a probe "
            f"stimulus ({type(e).__name__}: {e}), so its feature dimension is unknown "
            f"and no estimate was made. If the model configures itself inside the "
            f"benchmark (e.g. start_recording called in __call__), the probe can fail "
            f"for an otherwise-fine candidate — the OOM guard is then OFF for this run. "
            f"Pass feature_dim=... to estimate anyway, or check_mem=False to silence.",
            exc_info=True,
        )
        return None
    shape = getattr(result, 'shape', None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    logger.warning(
        "Memory pre-flight could NOT run: the probe returned a result with shape "
        "%r (need >=2 dims to read the feature width), so no estimate was made and "
        "the OOM guard is OFF for this run. Pass feature_dim=... to estimate anyway.",
        shape,
    )
    return None


def _resolve_feature_dim(model, benchmark, feature_dim, stimulus_set):
    """Resolve the model's per-stimulus feature width, preferring a declaration
    over a probe. Returns (n_features, source) or (None, source) if unresolved."""
    if feature_dim is not None:
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be a positive integer, got {feature_dim}")
        return int(feature_dim), 'feature_dim argument'
    for obj, attr in ((benchmark, 'expected_feature_dim'), (model, 'feature_dim')):
        declared = getattr(obj, attr, None)
        if declared:
            if int(declared) <= 0:
                raise ValueError(
                    f"{type(obj).__name__}.{attr} must be a positive integer, "
                    f"got {declared}")
            return int(declared), f'{type(obj).__name__}.{attr}'
    return _probe_feature_dim(model, stimulus_set), 'single-stimulus probe'


def _get_n_presentations(benchmark, stimulus_set):
    """How many rows the model will actually process, and whether it is declared.

    ``len(stimulus_set)`` is only a LOWER bound: benchmarks that expand each row
    (video -> per-TR frames in Algonauts, clip -> frames in Lahner) process far
    more presentations than the raw set has rows. A benchmark can declare the true
    count via ``expected_n_presentations``; otherwise we fall back to the raw
    length and flag the estimate as a lower bound.

    Returns (n_presentations, declared: bool).
    """
    declared = getattr(benchmark, 'expected_n_presentations', None)
    if declared:
        return int(declared), True
    return len(stimulus_set), False


def _ensure_recording(model, benchmark) -> None:
    """Best-effort: put the model into a recording state so the probe produces
    real activations.

    Benchmarks that expose ``.region`` are configured directly. Benchmarks that
    call ``start_recording`` inside ``__call__`` (e.g. Lahner) expose no region,
    so a naive probe would fail for an otherwise-fine layer-aware candidate and
    silently disable the guard. To avoid that fail-open, fall back to recording
    the model's own ``region_layer_map`` regions.
    """
    if not hasattr(model, 'start_recording'):
        return
    region = getattr(benchmark, 'region', None)
    time_bins = getattr(benchmark, 'timebins', None)
    try:
        if region is not None:
            model.start_recording(region, time_bins=time_bins)
            return
        region_map = getattr(model, 'region_layer_map', None)
        if region_map:
            model.start_recording(list(region_map.keys()), time_bins=time_bins)
    except Exception:
        pass  # best-effort; the probe still falls back to a clear warning


def check_memory(
    model: 'UnifiedModel',
    benchmark,
    safety_factor: Optional[float] = None,
    feature_dim: Optional[int] = None,
) -> None:
    """
    Best-effort memory pre-flight: raise before a run that is *clearly* over
    budget, and otherwise be honest that a "fit" is not a safety guarantee.

    This is NOT a reliable OOM oracle, and does not claim to be. A trustworthy
    estimate needs the benchmark to declare its execution plan; absent that, two
    inputs are only approximations:
      - ``n_presentations``: from ``benchmark.expected_n_presentations`` if
        declared, else ``len(stimulus_set)`` — a LOWER bound, because benchmarks
        that expand rows (video->per-TR frames) process many more.
      - ``n_features``: from ``feature_dim`` arg / ``benchmark.expected_feature_dim``
        / ``model.feature_dim`` / a single-stimulus probe. The probed width is the
        RAW recording width; benchmarks that PCA/compress features before the
        metric (Algonauts, TR-resolved Lahner) see a smaller width, so the metric
        term is an over-estimate there. Temporal outputs (S x T x F) are also not
        modelled — the held matrix is undercounted by T.

    Because of these, the estimate is compared against HOST RAM only (the plan is
    host-side numpy/sklearn); the GPU forward-pass working set is a separate,
    unmodelled failure mode. The check RAISES only when the estimate exceeds the
    budget (a definite problem), and for benchmarks that do NOT declare their plan
    it logs that a pass is inconclusive rather than reporting a false all-clear.

    Skips (with a loud warning) when no feature width can be resolved — e.g. a
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
        logger.info("Memory check skipped: benchmark has no stimulus_set")
        return

    _ensure_recording(model, benchmark)

    n_features, source = _resolve_feature_dim(
        model, benchmark, feature_dim, stimulus_set)
    if n_features is None:
        return  # unresolved feature dim; _probe_feature_dim already warned loudly

    n_presentations, cardinality_declared = _get_n_presentations(benchmark, stimulus_set)
    n_targets = _get_n_targets(benchmark)

    # Extraction plan: the held activation matrix (float32) plus a coarse
    # forward-pass transient. NOTE the transient factor is a rough scalar on the
    # OUTPUT matrix and is NOT an upper bound on the true transient — a wide-input
    # model (video (B,T,C,H,W)) can spike far above output width. Advisory only.
    held_activations = n_presentations * n_features * ACTIVATION_DTYPE_BYTES
    extraction_transient = int(held_activations * EXTRACTION_TRANSIENT_FACTOR)

    # Metric workspace (float64 inside sklearn — accounted for in the helper).
    estimated_metric_memory = estimate_metric_memory(benchmark, n_features=n_features)
    if category != 'behavioral':
        estimated_ceiling_memory = estimate_metric_memory(benchmark, n_features=n_targets)
    else:
        estimated_ceiling_memory = 0
    metric_total = (estimated_metric_memory + estimated_ceiling_memory) * PIPELINE_OVERHEAD

    # The metric runs while the extracted activations are still resident.
    peak_from_extraction = baseline_rss + extraction_transient
    peak_from_metric = baseline_rss + held_activations + metric_total
    total_estimated = max(peak_from_extraction, peak_from_metric)

    approximate = not (cardinality_declared and source != 'single-stimulus probe')
    total_system = available + baseline_rss
    if total_estimated > total_system:
        raise MemoryError(
            f"Estimated peak for '{model.identifier}' on "
            f"'{getattr(benchmark, 'identifier', 'unknown')}': "
            f"{total_estimated / 1e9:.1f} GB "
            f"(extraction peak: {peak_from_extraction / 1e9:.1f} GB, "
            f"metric peak: {peak_from_metric / 1e9:.1f} GB, "
            f"metric [{category}]: {estimated_metric_memory / 1e9:.1f} GB, "
            f"ceiling: {estimated_ceiling_memory / 1e9:.1f} GB, "
            f"n_features: {n_features} [{source}], "
            f"n_presentations: {n_presentations}"
            f"{' [declared]' if cardinality_declared else ' [lower bound]'}). "
            f"Available host RAM: {total_system / 1e9:.1f} GB. "
            f"Set check_mem=False to skip this check."
        )

    utilization = total_estimated / total_system if total_system > 0 else 0
    peak_source = "extraction" if peak_from_extraction >= peak_from_metric else "metric"
    logger.info(
        f"Memory estimate for '{model.identifier}' on "
        f"'{getattr(benchmark, 'identifier', 'unknown')}' "
        f"[{category}, {n_features} features via {source}, "
        f"{n_presentations} presentations]: {total_estimated / 1e9:.1f} GB peak "
        f"(held activations: {held_activations / 1e9:.1f} GB; "
        f"metric: {estimated_metric_memory / 1e9:.1f} GB; "
        f"bottleneck: {peak_source}) — {total_system / 1e9:.1f} GB host RAM "
        f"({utilization:.0%} utilization)"
    )
    if approximate:
        # Honest: an un-declared benchmark may expand stimuli / compress features
        # in ways the plan can't see, so a "fit" is NOT an all-clear.
        logger.info(
            f"Memory estimate for '{getattr(benchmark, 'identifier', 'unknown')}' "
            f"is APPROXIMATE (execution plan not declared): stimulus expansion, "
            f"temporal bins, and feature compression are not modelled, so passing "
            f"this check is not a guarantee against OOM. Declare "
            f"expected_n_presentations / expected_feature_dim for a reliable check."
        )
    if utilization > 0.8:
        warnings.warn(
            f"Memory utilization for '{model.identifier}' estimated at "
            f"{utilization:.0%} of available host RAM ({total_estimated / 1e9:.1f} / "
            f"{available / 1e9:.1f} GB). OOM risk is elevated.",
            ResourceWarning,
        )
