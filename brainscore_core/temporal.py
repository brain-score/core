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
