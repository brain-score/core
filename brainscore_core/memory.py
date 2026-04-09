"""
Probe-based memory pre-check for Brain-Score.

Runs a single stimulus through the model, measures peak memory,
and extrapolates to the full benchmark. Prevents OOM failures
after hours of computation.

Cross-platform: CUDA (torch.cuda), MPS (psutil), CPU (psutil).
brainscore_core stays free of heavy dependencies -- torch is optional.
"""

import logging
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


def estimate_metric_memory(benchmark) -> int:
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

    Returns estimate in bytes.
    """
    category = _detect_metric_category(benchmark)
    n_stimuli = _get_n_stimuli(benchmark)
    n_targets = _get_n_targets(benchmark)
    n_features = getattr(benchmark, 'expected_feature_dim', 1000)

    if category == 'behavioral' or n_stimuli == 0:
        return 0

    if category == 'rsa':
        # Dissimilarity matrix: S x S float64, plus model RDM
        rdm_size = n_stimuli * n_stimuli * 8
        return rdm_size * 2  # neural RDM + model RDM

    # Shared: design and target matrices
    # Design: (S, F) float32;  Target: (S, T) float32
    design = n_stimuli * n_features * 4
    target = n_stimuli * n_targets * 4

    if category == 'pls':
        # PLS: design + target + deflated matrices (~2x design) + components
        n_components = 25
        components = n_features * n_components * 4
        return design + target + design * 2 + components

    if category == 'ridge':
        # Ridge: normal equations F x F + design + target
        gram = n_features * n_features * 4
        return gram + design + target

    if category == 'ridgecv':
        # RidgeCV: grid search over A alphas, sklearn allocates
        # (S, T, A) for leave-one-out predictions + (F, F) gram per alpha
        n_alphas = _get_n_alphas(benchmark)
        gram = n_features * n_features * 4
        loo_predictions = n_stimuli * n_targets * n_alphas * 4
        return gram + design + target + loo_predictions

    # Fallback
    return design + target + design * 3


def check_memory(
    model: 'UnifiedModel',
    benchmark,
    safety_factor: Optional[float] = None,
    max_probe_stimuli: int = 1,
) -> None:
    """
    Run a single-stimulus probe through the model and extrapolate
    whether the full benchmark will fit in memory.

    Raises MemoryError if estimated peak exceeds available memory.
    Logs a warning if estimated peak exceeds 80% of available memory.

    :param model: the model to check
    :param benchmark: the benchmark to check against
    :param safety_factor: multiplier for memory estimate. If None (default),
        auto-selects based on benchmark metric type: PLS=1.5, ridge=2.0,
        ridgecv=3.0, RSA=2.0, behavioral=1.2.
    :param max_probe_stimuli: number of stimuli in probe (default 1)
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

    probe = stimulus_set.iloc[:max_probe_stimuli]

    # Configure model for recording if the benchmark specifies a region.
    # Benchmarks normally call start_recording() before running the model;
    # we replicate that here so the probe produces real activations.
    region = getattr(benchmark, 'region', None)
    if region is not None and hasattr(model, 'start_recording'):
        timebins = getattr(benchmark, 'timebins', None)
        model.start_recording(region, time_bins=timebins)

    # Two-probe approach: measure RSS after 1 stimulus, then after a small
    # batch, to compute the marginal per-stimulus cost. This captures the
    # raw activation storage that accumulates during extraction (before PCA
    # compresses it). A single probe only shows the tiny post-PCA result.
    import psutil
    baseline = psutil.Process().memory_info().rss

    try:
        result = model.process(probe)
    except Exception as e:
        logger.info(f"Memory check skipped: probe failed ({type(e).__name__}: {e})")
        return

    rss_after_1 = psutil.Process().memory_info().rss

    # Second probe with a larger batch to measure marginal per-stimulus cost
    n_probe_2 = min(10, len(stimulus_set))
    if n_probe_2 > max_probe_stimuli:
        probe_2 = stimulus_set.iloc[max_probe_stimuli:n_probe_2]
        try:
            model.process(probe_2)
        except Exception:
            pass  # fall back to single-probe estimate
        rss_after_n = psutil.Process().memory_info().rss
        n_new = n_probe_2 - max_probe_stimuli
        marginal_per_stimulus = max(rss_after_n - rss_after_1, 0) / n_new if n_new > 0 else 0
    else:
        marginal_per_stimulus = 0

    forward_pass_peak = max(rss_after_1 - baseline, 0)
    activation_bytes = getattr(result, 'nbytes', 0) if result is not None else 0
    # Use the measured marginal if available, otherwise fall back to result.nbytes
    per_stimulus_cost = max(marginal_per_stimulus, activation_bytes)

    n_stimuli = len(stimulus_set)
    stored_activations = int(per_stimulus_cost * n_stimuli)
    estimated_scoring_memory = forward_pass_peak + stored_activations
    estimated_metric_memory = estimate_metric_memory(benchmark)

    # Include the pre-scoring baseline (Python runtime + loaded model + libraries).
    # This memory is already consumed and must be counted toward OOM risk.
    total_estimated = baseline + (estimated_scoring_memory + estimated_metric_memory) * safety_factor

    if total_estimated > available + baseline:
        # Compare against total system memory (available + what we already hold)
        raise MemoryError(
            f"Estimated memory for '{model.identifier}' on "
            f"'{getattr(benchmark, 'identifier', 'unknown')}': "
            f"{total_estimated / 1e9:.1f} GB "
            f"(baseline: {baseline / 1e9:.1f} GB, "
            f"forward pass: {forward_pass_peak / 1e9:.1f} GB, "
            f"activations: {stored_activations / 1e9:.1f} GB "
            f"[{per_stimulus_cost / 1e6:.1f} MB/stim x {n_stimuli}], "
            f"metric: {estimated_metric_memory / 1e9:.1f} GB, "
            f"safety: {safety_factor}x). "
            f"Available: {(available + baseline) / 1e9:.1f} GB total. "
            f"Options: use a larger instance or "
            f"set check_mem=False to skip this check."
        )

    # Utilization against total system memory
    total_system = available + baseline
    utilization = total_estimated / total_system if total_system > 0 else 0
    logger.info(
        f"Memory estimate for '{model.identifier}' on "
        f"'{getattr(benchmark, 'identifier', 'unknown')}' "
        f"[{category}]: {total_estimated / 1e9:.1f} GB "
        f"(baseline: {baseline / 1e9:.1f} GB, "
        f"forward pass: {forward_pass_peak / 1e9:.1f} GB, "
        f"activations: {stored_activations / 1e9:.1f} GB "
        f"[{per_stimulus_cost / 1e6:.1f} MB/stim x {n_stimuli}], "
        f"metric: {estimated_metric_memory / 1e9:.1f} GB, "
        f"safety: {safety_factor}x) — "
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
