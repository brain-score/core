"""Streaming presentations, bounded windows and timing policies.

Each is a plain generator over an iterable of inputs. Hand the generator to
``Session`` as its source: no session subclass is involved.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Sequence, Union
from ..streaming import Session, StreamEvent, _with_trial_memory
from .payloads import (
    _neural_channels,
    _stimulus_input_columns,
    _table_row_events,
)


def streaming_stimulus_session(stimulus_set, record: Union[str, Sequence[str]] = "IT",
                               time_bins=None, task_context=None) -> Session:
    """Open-loop session that creates one stimulus event at a time.

    ``stimulus_session`` lists every event up front. This one keeps the caller's
    stimulus table but builds its events lazily from a row iterator, so event
    buffering stays bounded. Truly generated feeds that do not already exist as
    a table use ``windowed_stream`` instead.

    Trials are marked with ``begin`` / ``end`` as ``task_context.trial_memory`` says
    (``none`` without a task context), exactly as ``table_session`` does.

    ``streaming=True`` is what routes ``participate`` to the one-at-a-time driver.
    """
    columns = _stimulus_input_columns(stimulus_set)

    def events() -> Iterator[StreamEvent]:
        # One event per input column; a several-column row marks its
        # events with meta['table_row'] so the row is presented once.
        for position, (row_index, row) in enumerate(stimulus_set.iterrows()):
            yield from _table_row_events(columns, position, row_index, row)

    memory = 'none' if task_context is None else task_context.trial_memory
    session = Session(_with_trial_memory(events(), memory), _neural_channels(record),
                      task_context, stimulus_set=stimulus_set, time_bins=time_bins,
                      streaming=True, record=record)
    session.one_answer_per_trial = memory == 'none'
    return session


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
            channel="vision", payload=stimuli, t_ms=start_ms,
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


@dataclass
class PacingReport:
    """What ``paced_stream`` measured. Pass one in, read it back after the run."""
    policy: str = 'lag'
    stride_ms: float = 0.0
    windows_dropped: int = 0
    windows_delivered: int = 0
    max_lag_ms: float = 0.0
    processing_ms: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """Processing seconds per second of material. < 1.0 keeps up."""
        material_ms = (self.windows_delivered + self.windows_dropped) * self.stride_ms
        return (self.processing_ms / material_ms) if material_ms else 0.0

    def report(self) -> dict:
        return {
            'policy': self.policy,
            'windows_delivered': self.windows_delivered,
            'windows_dropped': self.windows_dropped,
            'max_lag_ms': round(self.max_lag_ms, 1),
            'realtime_factor': round(self.realtime_factor, 3),
            'kept_up': self.realtime_factor < 1.0 and self.windows_dropped == 0,
        }


_POLICIES = ('drop', 'lag', 'error')


def paced_stream(windows: Iterable[StreamEvent], *, window_ms: float,
                 stride_ms: Optional[float] = None, policy: str = 'lag',
                 clock=None, sleep=None,
                 report: Optional[PacingReport] = None) -> Iterator[StreamEvent]:
    """Run windows (from ``windowed_stream``) against a wall clock.

    Batch and windowed inputs manage availability. This adds the constraint that
    makes streaming hard in practice: the feed advances whether or not the model
    is finished. A window covers ``window_ms`` of world time, so if the model takes
    longer than that to process one, it is falling behind and something must give.

    The policy is explicit rather than implicit, because all three answers are
    legitimate and they measure different things:
      * ``'drop'``   -- skip whatever arrived while busy and stay current. What a live
                        system does. Coverage is lost; latency is bounded.
      * ``'lag'``    -- process everything and fall behind. What an offline run does.
                        Coverage is complete; latency grows without bound.
      * ``'error'``  -- refuse to continue. For a benchmark that is only meaningful if
                        real-time was actually achieved.

    The clock is injected so this is testable without sleeping: ``clock()`` returns
    seconds, ``sleep(seconds)`` paces the feed. Defaults are the real ones. Each
    window is held back until the world has produced it. Counters go to ``report``
    (a ``PacingReport``); its ``realtime_factor`` is the honest headline.
    """
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
    import time as _time
    report = report if report is not None else PacingReport()
    report.policy = policy
    report.stride_ms = float(window_ms if stride_ms is None else stride_ms)
    return _paced(iter(windows), float(window_ms), policy, clock or _time.monotonic,
                  sleep or _time.sleep, report)


def _paced(windows, window_ms, policy, clock, sleep, report) -> Iterator[StreamEvent]:
    t0 = None
    for event in windows:
        if t0 is None:
            t0 = clock()
        elapsed_ms = (clock() - t0) * 1000.0
        lag_ms = elapsed_ms - event.meta['window_start_ms']
        if lag_ms > window_ms and policy != 'lag':
            if policy == 'error':
                raise RuntimeError(
                    f"real-time budget missed: {lag_ms:.0f} ms behind at window "
                    f"{event.meta['window_index']} (window is "
                    f"{window_ms:.0f} ms). Model is too slow for this "
                    f"feed; use policy='drop' to skip, or 'lag' to run offline.")
            report.windows_dropped += 1
            continue
        report.max_lag_ms = max(report.max_lag_ms, lag_ms)
        # pace: do not hand over a window before the world has produced it
        wait_s = (event.meta['window_start_ms'] - elapsed_ms) / 1000.0
        if wait_s > 0:
            sleep(wait_s)
        report.windows_delivered += 1
        last_emit_at = clock()
        yield event
        # time spent by the caller since the window came out = processing time
        report.processing_ms += (clock() - last_emit_at) * 1000.0
