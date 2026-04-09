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


def estimate_metric_memory(benchmark) -> int:
    """
    Estimate memory required for the benchmark's metric (regression).

    For static benchmarks: negligible.
    For naturalistic benchmarks with ridge regression against
    ~20,000 cortical vertices: can be 2-3 GB per fold.

    Returns estimate in bytes.
    """
    n_stimuli = len(getattr(benchmark, 'stimulus_set', []))
    n_targets = getattr(benchmark, 'n_targets', 100)
    n_features = getattr(benchmark, 'expected_feature_dim', 1000)

    # Design matrix: (n_stimuli, n_features) float32
    design = n_stimuli * n_features * 4
    # Target matrix: (n_stimuli, n_targets) float32
    target = n_stimuli * n_targets * 4
    # SVD/Cholesky intermediate: ~3x design matrix
    intermediate = design * 3

    return design + target + intermediate


def check_memory(
    model: 'UnifiedModel',
    benchmark,
    safety_factor: float = 1.5,
    max_probe_stimuli: int = 1,
) -> None:
    """
    Run a single-stimulus probe through the model and extrapolate
    whether the full benchmark will fit in memory.

    Raises MemoryError if estimated peak exceeds available memory.
    Logs a warning if estimated peak exceeds 80% of available memory.

    :param model: the model to check
    :param benchmark: the benchmark to check against
    :param safety_factor: multiplier for memory estimate (default 1.5x)
    :param max_probe_stimuli: number of stimuli in probe (default 1)
    """
    available = get_available_memory()

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

    # Measure memory for one stimulus
    reset_peak_memory()
    baseline = get_peak_memory()

    try:
        model.process(probe)
    except Exception as e:
        # If the probe itself fails, let the benchmark handle the error later.
        # Don't mask real errors behind a memory check.
        logger.info(f"Memory check skipped: probe failed ({type(e).__name__}: {e})")
        return

    peak = get_peak_memory()
    per_stimulus_cost = max(peak - baseline, 0)

    # Extrapolate to full benchmark
    batch_size = getattr(benchmark, 'batch_size', 1)
    # Memory scales with batch size, not total stimuli (batches are sequential)
    estimated_model_memory = per_stimulus_cost * batch_size
    estimated_metric_memory = estimate_metric_memory(benchmark)
    total_estimated = (estimated_model_memory + estimated_metric_memory) * safety_factor

    if total_estimated > available:
        raise MemoryError(
            f"Estimated memory for '{model.identifier}' on "
            f"'{getattr(benchmark, 'identifier', 'unknown')}': "
            f"{total_estimated / 1e9:.1f} GB "
            f"(model: {estimated_model_memory / 1e9:.1f} GB per batch, "
            f"metric: {estimated_metric_memory / 1e9:.1f} GB, "
            f"safety: {safety_factor}x). "
            f"Available: {available / 1e9:.1f} GB. "
            f"Options: reduce batch_size, use a larger instance, or "
            f"set check_mem=False to skip this check."
        )

    utilization = total_estimated / available
    logger.info(
        f"Memory estimate for '{model.identifier}' on "
        f"'{getattr(benchmark, 'identifier', 'unknown')}': "
        f"{total_estimated / 1e9:.1f} GB "
        f"(model: {estimated_model_memory / 1e9:.1f} GB, "
        f"metric: {estimated_metric_memory / 1e9:.1f} GB, "
        f"safety: {safety_factor}x) — "
        f"{available / 1e9:.1f} GB available "
        f"({utilization:.0%} utilization)"
    )
    if utilization > 0.8:
        warnings.warn(
            f"Memory utilization for '{model.identifier}' estimated at "
            f"{utilization:.0%} of available ({total_estimated / 1e9:.1f} / "
            f"{available / 1e9:.1f} GB). OOM risk is elevated.",
            ResourceWarning,
        )
