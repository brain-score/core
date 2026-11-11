"""Streaming presentations, bounded windows and timing policies.

Each is a plain generator over an iterable of inputs. Hand the generator to
``Session`` as its source: no session subclass is involved.
"""

from __future__ import annotations
from typing import Any, Iterable, Iterator, Optional
from ..streaming import StreamEvent


def windowed_stream(frames: Iterable[Any], *, fps: float, window_ms: float,
                    stride_ms: Optional[float] = None, window_to_stimuli=None,
                    stats: Optional[dict] = None) -> Iterator[StreamEvent]:
    """Consume an open-ended frame feed in fixed windows.

    ``window_plan`` in ``brainscore.temporal`` tiles a known duration. A real feed has no
    known duration, so windowing here is incremental: pull frames from an iterator,
    hold at most one window's worth, emit when the window fills, slide by the stride,
    keep going until the feed runs dry.

    A window batches ``window_ms`` of material into one model call. At most one
    window is held, so the frame buffer does not grow with the feed.

    ``frames`` is any iterable of per-frame payloads; it is never listed, so a lazy
    decoder stays lazy. ``window_to_stimuli(frames, start_ms, end_ms)`` converts one
    window into whatever the subject consumes -- that is the caller's business,
    because a video model and a frame-aggregation model want different things from
    the same feed. ``stats``, when given, is filled with ``max_frames_held`` and
    ``windows_emitted``.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if window_ms <= 0:
        raise ValueError(f"window_ms must be > 0, got {window_ms}")
    stride_ms = window_ms if stride_ms is None else stride_ms
    if stride_ms <= 0:
        raise ValueError(f"stride_ms must be > 0, got {stride_ms}")
    return _windows(iter(frames), float(fps), float(window_ms), float(stride_ms),
                    window_to_stimuli or _default_window_to_stimuli,
                    stats if stats is not None else {})


def _windows(frames, fps, window_ms, stride_ms, window_to_stimuli, stats) -> Iterator[StreamEvent]:
    window_frames = max(1, int(round(window_ms * fps / 1000.0)))
    stride_frames = max(1, int(round(stride_ms * fps / 1000.0)))
    stats.update(max_frames_held=0, windows_emitted=0)
    buffer: list[Any] = []
    exhausted = False
    window_index = 0
    while True:
        while len(buffer) < window_frames and not exhausted:
            try:
                buffer.append(next(frames))
            except StopIteration:
                exhausted = True
        stats['max_frames_held'] = max(stats['max_frames_held'], len(buffer))
        if not buffer:
            return
        # a trailing partial window is still real material -- emit it rather than drop it
        window = list(buffer[:window_frames])
        start_ms = window_index * stride_ms
        end_ms = start_ms + (len(window) / fps) * 1000.0
        stimuli = window_to_stimuli(window, start_ms, end_ms)
        event = StreamEvent(
            channel="vision:image", payload=stimuli, t_ms=start_ms,
            meta={"window_index": window_index, "window_start_ms": start_ms,
                  "window_end_ms": end_ms, "n_frames": len(window),
                  "partial": len(window) < window_frames},
        )
        buffer = buffer[stride_frames:]
        window_index += 1
        stats['windows_emitted'] += 1
        yield event


def _default_window_to_stimuli(frames, start_ms, end_ms):
    """One row per frame, timestamped within the window.

    Deliberately simple: it keeps the frames addressable so a frame-aggregation model
    works out of the box. A native-video model should pass its own converter that
    packs the window into a single clip row.
    """
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    n = len(frames)
    step = (end_ms - start_ms) / n if n else 0.0
    return StimulusSet(pd.DataFrame({
        "stimulus_id": [f"w{int(start_ms)}_f{i}" for i in range(n)],
        "image_file_name": list(frames),
        "t_ms": [start_ms + i * step for i in range(n)],
    }))
