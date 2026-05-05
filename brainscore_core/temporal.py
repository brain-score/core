"""
Temporal binning for naturalistic benchmarks.

The core pattern (see [[Unified Model Interface - Vision and Goals]]
§"Temporal Binning: How Timestamps Travel Through the Pipeline"):

1. Preprocessor expands a clip into individual frames/tokens with
   ``clip_id`` and ``frame_time_ms`` coordinates in the stimulus set.
2. The activations_model (PytorchWrapper, TextWrapper, etc.) treats each
   row as an independent presentation and extracts per-frame activations.
   Timestamps ride through unchanged as presentation coordinates.
3. After extraction, ``temporal_bin()`` reads ``clip_id`` and
   ``frame_time_ms`` from the output assembly, groups frames by clip,
   averages activations within each requested time window, and returns
   a ``NeuroidAssembly`` with a ``time_bin`` dimension.

This file provides only step 3. Steps 1 and 2 are model/preprocessor
concerns — a video preprocessor knows how to split a clip into
timestamped rows; the existing PytorchWrapper and TextWrapper already
preserve arbitrary stimulus-set columns as presentation coordinates.

Key property: zero changes to PytorchWrapper or TextWrapper. Timestamps
are carried by the StimulusSet (via its DataFrame columns), which is
already the conventional metadata channel. No "sidecar" dataclass.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, walk_coords,
)


TimeBin = Tuple[float, float]
# Time bins are represented as (start_ms, end_ms) tuples. Open at the top
# (end is exclusive) so adjacent bins like (0, 500) and (500, 1000) don't
# double-count frames on the boundary.


def add_time_bin_axis(
    assembly: NeuroidAssembly,
    time_bin_start_ms: Optional[float] = None,
    time_bin_end_ms: Optional[float] = None,
) -> NeuroidAssembly:
    """Promote a 2D ``(presentation, neuroid)`` assembly into a 3D
    ``(presentation, time_bin, neuroid)`` assembly with ``time_bin=1``.

    Lets benchmark code consume image-wrapper output (PytorchWrapper,
    VLMVisionWrapper) the same way it consumes video/audio/per-token-text
    wrappers. The axis is purely structural — no aggregation happens —
    so the values are unchanged and the cost is one ``expand_dims`` call.

    Optional ``time_bin_start_ms`` / ``time_bin_end_ms`` attach absolute
    ms coords on the new axis. Useful when a still-image stimulus has a
    known on-screen window (e.g., 100 ms presentation onset → offset).

    No-op when the assembly already carries a ``time_bin`` dim.
    """
    if 'time_bin' in assembly.dims:
        return assembly
    promoted = assembly.expand_dims('time_bin', axis=1)
    if time_bin_start_ms is not None:
        promoted = promoted.assign_coords(
            time_bin_start_ms=('time_bin', [float(time_bin_start_ms)]))
    if time_bin_end_ms is not None:
        promoted = promoted.assign_coords(
            time_bin_end_ms=('time_bin', [float(time_bin_end_ms)]))
    return promoted


def temporal_bin(
    assembly: NeuroidAssembly,
    time_bins: Sequence[TimeBin],
    *,
    time_coord: str = 'frame_time_ms',
    group_coord: str = 'clip_id',
    aggregation: str = 'mean',
) -> NeuroidAssembly:
    """Group per-frame activations by clip, bin by timestamp.

    Input assembly is expected to have dims ``('presentation', 'neuroid')``
    with ``time_coord`` and ``group_coord`` both varying along the
    presentation dimension. Each row is one frame/token of one clip.

    Output has dims ``('presentation', 'time_bin', 'neuroid')`` where
    presentation is one-entry-per-clip and time_bin is one-entry-per-window.

    Args:
        assembly: 2-D ``NeuroidAssembly`` indexed by frame (not by clip).
        time_bins: List of ``(start_ms, end_ms)`` tuples. End is exclusive.
        time_coord: Name of the presentation-dimension coordinate giving
            each frame's timestamp in milliseconds.
        group_coord: Name of the coordinate identifying which clip a
            frame belongs to. Frames with the same value are pooled.
        aggregation: How to combine activations of frames within one bin.
            Supported: ``'mean'`` (default), ``'first'`` (take the earliest
            frame in the bin), ``'none'`` (raise on ambiguity).

    Returns:
        A ``NeuroidAssembly`` of shape
        ``(n_clips, n_time_bins, n_neuroids)``. The clip-level
        presentation coordinate values are the unique ``group_coord``
        values in the order they first appear in the input.

    Raises:
        ValueError: If required coordinates are missing, or if a
            ``(clip, time_bin)`` pair has no frames and aggregation
            does not cover that case.

    Notes:
        - Bins are half-open: ``start <= t < end``.
        - Empty (clip, bin) cells currently produce ``NaN``. Callers
          (typically benchmarks or downstream metrics) decide how to
          handle NaNs — there is no one-size-fits-all rule.
    """
    # walk_coords is the canonical way to list all coords in this codebase
    # (xarray 2022.3.0 / NeuroidAssembly hides non-dim coords from
    # ``.coords`` when dims share a MultiIndex). Names not in this list
    # really are missing.
    all_coord_names = [name for name, _, _ in walk_coords(assembly)]
    if time_coord not in all_coord_names:
        raise ValueError(
            f"temporal_bin requires a '{time_coord}' coordinate on the "
            f"input assembly's presentation dimension. "
            f"Available coords: {all_coord_names}."
        )
    if group_coord not in all_coord_names:
        raise ValueError(
            f"temporal_bin requires a '{group_coord}' coordinate on the "
            f"input assembly's presentation dimension. "
            f"Available coords: {all_coord_names}."
        )
    if aggregation not in ('mean', 'first', 'none'):
        raise ValueError(
            f"aggregation={aggregation!r} not in supported "
            f"('mean', 'first', 'none').")

    # Pull the time and group arrays; both should be length n_presentations.
    times = np.asarray(assembly[time_coord].values)
    groups = np.asarray(assembly[group_coord].values)
    data = assembly.values  # (n_presentations, n_neuroids)

    # Unique clips, preserving order-of-first-appearance.
    # np.unique returns sorted — we want first-appearance instead.
    _, first_idx = np.unique(groups, return_index=True)
    clip_order = groups[np.sort(first_idx)]

    n_clips = len(clip_order)
    n_time_bins = len(time_bins)
    n_neuroids = data.shape[1]

    output = np.full((n_clips, n_time_bins, n_neuroids), np.nan,
                     dtype=data.dtype)

    # Build per-clip row masks once.
    clip_to_rows = {clip: np.where(groups == clip)[0] for clip in clip_order}

    for ci, clip in enumerate(clip_order):
        rows = clip_to_rows[clip]
        clip_times = times[rows]
        for bi, (start, end) in enumerate(time_bins):
            bin_mask = (clip_times >= start) & (clip_times < end)
            if not bin_mask.any():
                continue  # leave NaN
            bin_rows = rows[bin_mask]
            if aggregation == 'mean':
                output[ci, bi] = data[bin_rows].mean(axis=0)
            elif aggregation == 'first':
                earliest_offset = np.argmin(times[bin_rows])
                output[ci, bi] = data[bin_rows[earliest_offset]]
            elif aggregation == 'none':
                if len(bin_rows) > 1:
                    raise ValueError(
                        f"aggregation='none' but bin {bi} for clip "
                        f"{clip} has {len(bin_rows)} frames.")
                output[ci, bi] = data[bin_rows[0]]

    # Coord construction:
    # - presentation dim: one entry per clip. We add stimulus_id as a
    #   second coord to force a MultiIndex — otherwise gather_indexes
    #   in NeuroidAssembly.__init__ promotes a single coord to a plain
    #   Index and the coord name is lost (accessing result['clip_id']
    #   would fail).
    # - time_bin dim: multiple coords (center_ms, start_ms, end_ms)
    #   trigger a MultiIndex here too, which preserves the names.
    # - neuroid dim: carry over from input.
    time_bin_centers = [0.5 * (s + e) for (s, e) in time_bins]
    time_bin_bounds = list(time_bins)

    # Copy neuroid-dim coords from input
    neuroid_coords = {}
    for coord_name, dims, values in walk_coords(assembly):
        if set(dims) == {'neuroid'}:
            neuroid_coords[coord_name] = ('neuroid', values)

    coords = {
        'clip_id': ('presentation', list(clip_order)),
        # Mirror as stimulus_id so gather_indexes builds a MultiIndex on
        # the presentation dim and both names remain accessible.
        'stimulus_id': ('presentation', list(clip_order)),
        'time_bin_center_ms': ('time_bin', time_bin_centers),
        'time_bin_start_ms': ('time_bin', [s for (s, _) in time_bin_bounds]),
        'time_bin_end_ms': ('time_bin', [e for (_, e) in time_bin_bounds]),
        **neuroid_coords,
    }

    return NeuroidAssembly(
        output,
        coords=coords,
        dims=['presentation', 'time_bin', 'neuroid'],
    )


def expand_clip_to_frames(
    clip_stimulus_set,
    sample_times_ms: Sequence[float],
    *,
    frame_extractor=None,
    clip_id_col: str = 'stimulus_id',
    video_col: str = 'video_path',
    output_image_col: str = 'image_file_name',
) -> 'StimulusSet':  # noqa: F821
    """Helper for vision preprocessors: expand clips into timestamped frames.

    Takes a stimulus set where each row is a clip (one ``video_path``
    per row) and returns a new stimulus set where each row is a single
    frame with ``clip_id`` and ``frame_time_ms`` columns.

    ``frame_extractor`` is a callable ``(video_path, time_ms) -> Path``
    that produces an image file for the requested frame. It is the
    caller's job to supply this (OpenCV / ffmpeg / etc.) — we stay
    dependency-free here.

    Args:
        clip_stimulus_set: Input StimulusSet, one row per clip.
        sample_times_ms: Sequence of timestamps (in ms) at which to
            extract frames. Same sampling applied to every clip.
        frame_extractor: Callable that extracts one frame from a video
            at a given timestamp. Returns a path (str or Path).
        clip_id_col: Column in input set that identifies each clip.
            Becomes ``clip_id`` in output.
        video_col: Column in input set giving the video file path.
        output_image_col: Column to write frame image paths into.

    Returns:
        An expanded StimulusSet with one row per (clip, sample_time)
        pair. Columns include: ``stimulus_id`` (unique per frame),
        ``clip_id`` (preserved from input row), ``frame_time_ms``,
        ``output_image_col`` (path to extracted frame), and all other
        original columns (duplicated per frame).
    """
    if frame_extractor is None:
        raise ValueError(
            "expand_clip_to_frames requires a frame_extractor "
            "callable; we don't bundle one to keep brainscore_core "
            "dependency-free. Typical implementations use OpenCV "
            "(cv2.VideoCapture) or ffmpeg.")

    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    expanded_rows = []
    stimulus_paths = {}
    for _, row in clip_stimulus_set.iterrows():
        clip_id = row[clip_id_col]
        video_path = row[video_col]
        for t_ms in sample_times_ms:
            frame_path = str(frame_extractor(video_path, t_ms))
            frame_stim_id = f'{clip_id}_t{int(t_ms)}'
            new_row = dict(row)
            new_row['stimulus_id'] = frame_stim_id
            new_row['clip_id'] = clip_id
            new_row['frame_time_ms'] = float(t_ms)
            new_row[output_image_col] = frame_path
            expanded_rows.append(new_row)
            stimulus_paths[frame_stim_id] = frame_path

    df = pd.DataFrame(expanded_rows)
    new_set = StimulusSet(df)
    new_set.identifier = f'{clip_stimulus_set.identifier}-frames'
    new_set.stimulus_paths = stimulus_paths
    return new_set


# ═════════════════════════════════════════════════════════════════
# Time-series CV + HRF correction for naturalistic fMRI benchmarks.
# ═════════════════════════════════════════════════════════════════
#
# For benchmarks that score model features against a *continuous* fMRI
# time series (movie-watching, naturalistic speech), two pieces must
# differ from the image-benchmark default:
#
# 1. Cross-validation folds must be contiguous blocks of time, not
#    random stripes. Adjacent TRs in a BOLD time series are temporally
#    correlated by autocorrelation and share stimulus-driven shared
#    variance; random KFold leaks information from train to test and
#    produces inflated scores. EPFL's V-JEPA paper uses 15 s contiguous
#    blocks; Bonner et al. (2021), Toneva & Wehbe (2019), and TRIBEv2
#    use the same block-CV pattern.
#
# 2. Model features must be convolved with a hemodynamic response
#    function (HRF) before regressing against BOLD. The canonical SPM
#    double-gamma HRF peaks at ~5 s with an undershoot around 15 s.
#    Failing to HRF-convolve forces the regressor to learn an implicit
#    delay which conflates model and BOLD timing.
#
# Lahner2024 in its current form does NOT need either: the packaged
# assembly is pre-computed GLM betas (one value per clip, already
# HRF-deconvolved at packaging time). These utilities are for the
# future naturalistic time-series benchmarks (IBC, Courtois NeuroMod,
# Algonauts 2023) that M8/M12 target.


def contiguous_block_cv(
    n_samples: int,
    block_size_samples: int,
    n_splits: Optional[int] = None,
):
    """Yield (train_idx, test_idx) splits where each test set is a
    contiguous block of consecutive indices.

    For time-series data where adjacent samples share variance (BOLD
    autocorrelation, naturalistic stimulus structure), random KFold
    leaks information and inflates scores. Contiguous blocks preserve
    the temporal structure of held-out data.

    The default partition is sequential, non-overlapping: fold ``i``
    holds out samples ``[i*block_size, (i+1)*block_size)``. Samples that
    don't fit evenly (``n_samples % block_size != 0``) fall into the
    last fold's train set on all but the final split, and vice versa
    on the final split — every sample appears in exactly one test set.

    Args:
        n_samples: Total number of time points.
        block_size_samples: Length of each held-out block, in samples.
        n_splits: Number of folds. Defaults to
            ``ceil(n_samples / block_size_samples)`` so every sample is
            held out in exactly one fold.

    Yields:
        ``(train_idx, test_idx)`` pairs of ``np.ndarray[int]``.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if block_size_samples <= 0:
        raise ValueError(
            f"block_size_samples must be positive, got {block_size_samples}")
    if block_size_samples > n_samples:
        raise ValueError(
            f"block_size_samples ({block_size_samples}) must not exceed "
            f"n_samples ({n_samples})")

    # Default: cover every sample exactly once by ceiling-dividing.
    default_n_splits = (n_samples + block_size_samples - 1) // block_size_samples
    n_splits = n_splits or default_n_splits

    if n_splits <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")

    for fold in range(n_splits):
        test_start = fold * block_size_samples
        test_end = min(test_start + block_size_samples, n_samples)
        if test_end <= test_start:
            # Happens if the user asked for more folds than blocks fit —
            # skip rather than emit an empty test set.
            continue
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n_samples),
        ])
        yield train_idx, test_idx


def double_gamma_hrf(
    duration_sec: float,
    sampling_rate_hz: float,
    peak_delay: float = 6.0,
    peak_dispersion: float = 1.0,
    undershoot_delay: float = 16.0,
    undershoot_dispersion: float = 1.0,
    peak_undershoot_ratio: float = 1 / 6.0,
) -> np.ndarray:
    """Sample the canonical SPM double-gamma HRF.

    ``h(t) = peak(t) - (1/6) * undershoot(t)``, where each component is
    a gamma density with shape/scale parameters. Defaults reproduce the
    SPM canonical (peak ≈ 5 s, undershoot ≈ 15 s, peak:undershoot = 6:1).

    The returned kernel is normalized to unit sum so convolution
    preserves the feature scale.

    Args:
        duration_sec: Length of the sampled kernel. 32 s covers the
            canonical peak + undershoot; longer is fine but wasteful.
        sampling_rate_hz: Sample rate of the output (typically equal to
            the fMRI TR, e.g. 1 / 1.5 s ≈ 0.667 Hz, or the model's
            feature sample rate).
        peak_delay/peak_dispersion: Gamma parameters for the positive
            peak component. SPM default: delay=6, dispersion=1.
        undershoot_delay/undershoot_dispersion: Gamma parameters for the
            undershoot. SPM default: delay=16, dispersion=1.
        peak_undershoot_ratio: Scale factor on the undershoot relative
            to the peak. SPM default: 1/6.

    Returns:
        1-D ``np.ndarray`` of length ``round(duration_sec * sampling_rate_hz)``,
        normalized to sum to 1.
    """
    from scipy.special import gamma as gamma_fn

    n = int(round(duration_sec * sampling_rate_hz))
    if n <= 0:
        raise ValueError(
            f"HRF duration × sample rate must be > 0; got {duration_sec}s "
            f"at {sampling_rate_hz} Hz.")

    t = np.arange(n, dtype=np.float64) / sampling_rate_hz
    # Gamma PDF with shape=delay, scale=1/dispersion. Equivalent to
    # scipy.stats.gamma(a=delay, scale=1/dispersion).pdf(t).
    peak = (t ** (peak_delay - 1) * (peak_dispersion ** peak_delay)
            * np.exp(-peak_dispersion * t)) / gamma_fn(peak_delay)
    undershoot = (t ** (undershoot_delay - 1)
                  * (undershoot_dispersion ** undershoot_delay)
                  * np.exp(-undershoot_dispersion * t)) / gamma_fn(undershoot_delay)
    h = peak - peak_undershoot_ratio * undershoot
    total = h.sum()
    if total == 0:
        raise ValueError(
            "HRF summed to zero — check duration/sampling parameters.")
    return h / total


def hrf_convolve(
    features: np.ndarray,
    sampling_rate_hz: float,
    hrf: Optional[np.ndarray] = None,
    hrf_duration_sec: float = 32.0,
) -> np.ndarray:
    """Convolve model features with an HRF along the time axis.

    Aligns the timing of model features to BOLD responses for regression
    against fMRI time series. The returned array has the same shape as
    the input — only the past influences the present (causal truncation
    of the ``np.convolve`` full mode).

    Args:
        features: ``(n_time, n_features)`` array. Rows are time points
            sampled at ``sampling_rate_hz``.
        sampling_rate_hz: Sample rate of the feature time series.
        hrf: Optional pre-computed 1-D HRF kernel. Must be sampled at
            the same rate as ``features``. If None, the canonical SPM
            double-gamma is used.
        hrf_duration_sec: Duration of the default HRF kernel when
            ``hrf`` is None. 32 s is standard (covers peak + undershoot).

    Returns:
        ``(n_time, n_features)`` convolved features, same dtype as input.
    """
    if features.ndim != 2:
        raise ValueError(
            f"hrf_convolve expects 2-D (time, features) input; got shape "
            f"{features.shape}")

    n_time, n_features = features.shape
    if n_time == 0:
        return features.copy()

    if hrf is None:
        hrf = double_gamma_hrf(hrf_duration_sec, sampling_rate_hz)

    if hrf.ndim != 1:
        raise ValueError(f"hrf must be 1-D; got shape {hrf.shape}")

    # Causal convolution: truncate 'full' result to the length of the
    # original time axis. Equivalent to 'same' mode for power-of-2 lengths
    # but explicit so behavior is unambiguous for any n.
    out = np.zeros_like(features)
    for j in range(n_features):
        conv = np.convolve(features[:, j], hrf, mode='full')[:n_time]
        out[:, j] = conv
    return out
