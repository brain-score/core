"""
Probe-based memory pre-check for Brain-Score.

Runs a small batch of stimuli through the model, measures peak memory
(including transient spikes), and extrapolates to the full benchmark.
Prevents OOM failures after hours of computation.

Cross-platform: CUDA (torch.cuda), MPS (psutil), CPU (psutil).
brainscore_core stays free of heavy dependencies -- torch is optional.
"""

import logging
import resource
import sys
import threading
import time
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


def _get_hwm_bytes() -> int:
    """Get the OS-level high-water mark RSS in bytes.

    Linux: ru_maxrss is in KB. macOS: ru_maxrss is in bytes.
    This is the peak RSS since process start (monotonically increasing).
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'linux':
        return maxrss * 1024
    return maxrss


def _measure_peak_rss(func, *args, **kwargs):
    """Run func while polling RSS in a background thread.

    Returns (result, peak_rss_bytes). The peak captures transient spikes
    (e.g., raw activations that are freed before func returns).
    """
    import psutil
    peak = [psutil.Process().memory_info().rss]
    stop = threading.Event()

    def poll():
        proc = psutil.Process()
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak[0]:
                    peak[0] = rss
            except Exception:
                pass
            time.sleep(0.005)  # 5ms polling

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        result = func(*args, **kwargs)
    finally:
        stop.set()
        thread.join(timeout=1.0)
    return result, peak[0]


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
    """Read actual neuroid/voxel count from the benchmark's assembly."""
    for attr in ('_assembly', 'train_assembly', 'test_assembly'):
        assembly = getattr(benchmark, attr, None)
        if assembly is not None:
            sizes = getattr(assembly, 'sizes', None)
            if sizes is not None and 'neuroid' in sizes:
                return sizes['neuroid']
    return getattr(benchmark, 'n_targets', 100)


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


def check_memory(
    model: 'UnifiedModel',
    benchmark,
    safety_factor: Optional[float] = None,
) -> None:
    """
    Run a probe extraction through the model and estimate whether the
    full benchmark will fit in memory.

    The probe runs the full stimulus set through the model to measure
    extraction overhead and discover the feature dimension. The metric
    cost is computed from benchmark metadata.

    Raises MemoryError if estimated total exceeds available memory.
    Logs a warning if utilization exceeds 80%.

    :param model: the model to check
    :param benchmark: the benchmark to check against
    :param safety_factor: no longer used (kept for API compatibility).
        Estimation is now based on measured extraction + computed metric.
    """
    available = get_available_memory()

    # Auto-detect safety factor from benchmark metric category
    category = _detect_metric_category(benchmark)
    if safety_factor is None:
        safety_factor = SAFETY_FACTORS.get(category, 1.5)

    # Get a probe stimulus from the benchmark's stimulus set.
    # Benchmarks store stimuli in different places:
    #   - benchmark.stimulus_set (some benchmarks expose directly)
    #   - benchmark._assembly.stimulus_set (NeuralBenchmark, PropertiesBenchmark)
    #   - benchmark.train_assembly.stimulus_set (TrainTestNeuralBenchmark)
    stimulus_set = getattr(benchmark, 'stimulus_set', None)
    if stimulus_set is None:
        for attr in ('_assembly', 'train_assembly'):
            assembly = getattr(benchmark, attr, None)
            if assembly is not None:
                stimulus_set = getattr(assembly, 'stimulus_set', None)
                if stimulus_set is not None:
                    break
    if stimulus_set is None or len(stimulus_set) == 0:
        logger.info("Memory check skipped: benchmark has no stimulus_set")
        return  # can't probe without stimuli

    # Configure model for recording if the benchmark specifies a region.
    # Benchmarks normally call start_recording() before running the model;
    # we replicate that here so the probe produces real activations.
    region = getattr(benchmark, 'region', None)
    if region is not None and hasattr(model, 'start_recording'):
        timebins = getattr(benchmark, 'timebins', None)
        model.start_recording(region, time_bins=timebins)

    import psutil
    baseline = psutil.Process().memory_info().rss

    # Run the full extraction as a probe. We use the complete stimulus set
    # (not a subset) because:
    # 1. The RSS delta measures the REAL extraction cost (image I/O, forward
    #    pass, xarray packaging, @store_xarray caching)
    # 2. The result gives us the actual feature dimension (n_features)
    # 3. The extraction is cached, so the actual scoring run benefits
    # 4. Stimulus subsets can fail due to metadata assertion mismatches
    try:
        result = model.process(stimulus_set)
    except Exception as e:
        logger.info(f"Memory check skipped: probe failed ({type(e).__name__}: {e})")
        return

    rss_after_probe = psutil.Process().memory_info().rss
    extraction_overhead = max(rss_after_probe - baseline, 0)

    # Read the actual feature dimension from the probe result.
    # Critical: models with region_layer_map skip PCA entirely, so features
    # can be 9K (alexnet) to 148K+ (ViT-L). Metric formulas depend on this.
    n_features = 1000  # fallback
    if result is not None:
        shape = getattr(result, 'shape', None)
        if shape is not None and len(shape) >= 2:
            n_features = shape[-1]

    n_stimuli = len(stimulus_set)

    # Metric memory: computed from benchmark metadata + measured n_features.
    # Uses float64 (8 bytes) because sklearn converts internally.
    estimated_metric_memory = estimate_metric_memory(benchmark, n_features=n_features)

    # Total = baseline (already consumed)
    #       + extraction_overhead (measured by probe — constant, not per-stimulus)
    #       + metric memory (computed from formula)
    total_estimated = baseline + extraction_overhead + estimated_metric_memory

    total_system = available + baseline
    if total_estimated > total_system:
        raise MemoryError(
            f"Estimated memory for '{model.identifier}' on "
            f"'{getattr(benchmark, 'identifier', 'unknown')}': "
            f"{total_estimated / 1e9:.1f} GB "
            f"(baseline: {baseline / 1e9:.1f} GB, "
            f"extraction: {extraction_overhead / 1e9:.1f} GB, "
            f"metric [{category}]: {estimated_metric_memory / 1e9:.1f} GB, "
            f"n_features: {n_features}, n_stimuli: {n_stimuli}). "
            f"Available: {total_system / 1e9:.1f} GB total. "
            f"Set check_mem=False to skip this check."
        )

    utilization = total_estimated / total_system if total_system > 0 else 0
    logger.info(
        f"Memory estimate for '{model.identifier}' on "
        f"'{getattr(benchmark, 'identifier', 'unknown')}' "
        f"[{category}, {n_features} features]: {total_estimated / 1e9:.1f} GB "
        f"(baseline: {baseline / 1e9:.1f} GB, "
        f"extraction: {extraction_overhead / 1e9:.1f} GB, "
        f"metric: {estimated_metric_memory / 1e9:.1f} GB) — "
        f"{total_system / 1e9:.1f} GB total system "
        f"({utilization:.0%} utilization)"
    )
    if utilization > 0.8:
        warnings.warn(
            f"Memory utilization for '{model.identifier}' estimated at "
            f"{utilization:.0%} of available ({total_estimated / 1e9:.1f} / "
            f"{available / 1e9:.1f} GB). OOM risk is elevated.",
            ResourceWarning,
        )
