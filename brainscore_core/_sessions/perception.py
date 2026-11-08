"""Streaming presentations, bounded windows and timing policies."""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union
from ..streaming import Session, StreamEvent
from ..supported_data_standards.brainio.assemblies import NeuroidAssembly
from .payloads import (
    _collect_requested_channels,
    _neural_channels,
    _stimulus_input_columns,
    _table_row_events,
)


class StreamingStimulusSetSession(Session):
    """Open-loop session that creates one stimulus event at a time.

    ``StimulusSetSession`` materializes every event in a deque up front. This one
    retains the caller's stimulus table but builds its event queue lazily from a
    row iterator, so event buffering stays bounded. Truly generated feeds that do
    not already exist as a table use ``WindowedStreamSession`` instead.

    ``streaming = True`` is what routes ``participate`` to the one-at-a-time driver.
    """

    streaming = True

    def __init__(self, stimulus_set, record: Union[str, Sequence[str]] = "IT", time_bins=None):
        self.stimulus_set = stimulus_set
        self.record = record
        self.time_bins = time_bins
        self.requested_output_channels = _neural_channels(record)
        self.emitted: list[StreamEvent] = []
        self._pending: list[StreamEvent] = []
        self._rows = enumerate(stimulus_set.iterrows())
        self._columns = _stimulus_input_columns(stimulus_set)
        self.max_events_held = 0     # observability: proves nothing is pre-drained

    def next_input(self) -> Optional[StreamEvent]:
        while not self._pending:
            try:
                position, (row_index, row) = next(self._rows)
            except StopIteration:
                return None
            # One event per input column; a several-column row marks its
            # events with meta['table_row'] so the row is presented once.
            self._pending.extend(_table_row_events(self._columns, position,
                                                   row_index, row))
        self.max_events_held = max(self.max_events_held, len(self._pending))
        return self._pending.pop(0)

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)

    def collect(
        self, channel: Optional[str] = None,
    ) -> Union[NeuroidAssembly, dict[str, NeuroidAssembly]]:
        """Collect per-stimulus outputs; omitted channel collects all requests."""
        if channel is None:
            return _collect_requested_channels(self)
        payloads = [e.payload for e in self.emitted if e.channel == channel]
        if not payloads:
            raise ValueError(f"No emitted events for channel {channel!r}.")
        if len(payloads) == 1:
            return payloads[0]
        import xarray as xr
        return xr.concat(payloads, dim="presentation")


class WindowedStreamSession(Session):
    """Consume an open-ended frame feed in fixed windows.

    ``window_plan`` in ``temporal.py`` tiles a known duration. A real feed has no
    known duration, so windowing here is incremental: pull frames from an iterator,
    hold at most one window's worth, emit when the window fills, slide by the stride,
    keep going until the feed runs dry.

    A window batches ``window_ms`` of material into one model call. The session
    holds at most one window, so its frame buffer does not grow with the feed.

    ``frames`` is any iterable of per-frame payloads; it is never listed, so a lazy
    decoder stays lazy. ``window_to_stimuli(frames, start_ms, end_ms)`` converts one
    window into whatever the subject consumes -- that is the caller's business, not
    the session's, because a video model and a frame-aggregation model want different
    things from the same feed.
    """

    streaming = True

    def __init__(self, frames, *, fps: float, window_ms: float,
                 stride_ms: Optional[float] = None, record: Union[str, Sequence[str]] = "IT",
                 window_to_stimuli=None, time_bins=None):
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if window_ms <= 0:
            raise ValueError(f"window_ms must be > 0, got {window_ms}")
        stride_ms = window_ms if stride_ms is None else stride_ms
        if stride_ms <= 0:
            raise ValueError(f"stride_ms must be > 0, got {stride_ms}")
        self._frames = iter(frames)
        self.fps = float(fps)
        self.window_ms = float(window_ms)
        self.stride_ms = float(stride_ms)
        self.record = record
        self.time_bins = time_bins
        self.requested_output_channels = _neural_channels(record)
        self.emitted: list[StreamEvent] = []
        self._window_frames = max(1, int(round(self.window_ms * self.fps / 1000.0)))
        self._stride_frames = max(1, int(round(self.stride_ms * self.fps / 1000.0)))
        self._buffer: list[Any] = []
        self._frames_consumed = 0
        self._window_index = 0
        self._exhausted = False
        self._window_to_stimuli = window_to_stimuli or _default_window_to_stimuli
        self.max_frames_held = 0          # observability: bounded by the window
        self.windows_emitted = 0

    def next_input(self) -> Optional[StreamEvent]:
        while len(self._buffer) < self._window_frames and not self._exhausted:
            try:
                self._buffer.append(next(self._frames))
                self._frames_consumed += 1
            except StopIteration:
                self._exhausted = True
        self.max_frames_held = max(self.max_frames_held, len(self._buffer))
        if not self._buffer:
            return None
        # a trailing partial window is still real material -- emit it rather than drop it
        window = list(self._buffer[:self._window_frames])
        start_ms = self._window_index * self.stride_ms
        end_ms = start_ms + (len(window) / self.fps) * 1000.0
        stimuli = self._window_to_stimuli(window, start_ms, end_ms)
        event = StreamEvent(
            channel="vision", payload=stimuli, t_ms=start_ms,
            meta={"window_index": self._window_index, "window_start_ms": start_ms,
                  "window_end_ms": end_ms, "n_frames": len(window),
                  "partial": len(window) < self._window_frames},
        )
        self._buffer = self._buffer[self._stride_frames:]
        if self._exhausted and len(self._buffer) < self._window_frames:
            # no more full windows can form; drain what remains next call, then stop
            if not self._buffer:
                self._exhausted = True
        self._window_index += 1
        self.windows_emitted += 1
        return event

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)

    def collect(
        self, channel: Optional[str] = None,
    ) -> Union[NeuroidAssembly, dict[str, NeuroidAssembly]]:
        """Collect window outputs; omitted channel collects all requests."""
        if channel is None:
            return _collect_requested_channels(self)
        payloads = [e.payload for e in self.emitted if e.channel == channel]
        if not payloads:
            raise ValueError(f"No emitted events for channel {channel!r}.")
        if len(payloads) == 1:
            return payloads[0]
        import xarray as xr
        return xr.concat(payloads, dim="presentation")


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


class RealTimeStreamSession(Session):
    """A windowed feed that runs against a wall clock.

    Batch and windowed sessions manage input availability. This session adds the constraint
    that makes streaming hard in practice: the feed advances whether or not the model
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
    seconds, ``sleep(seconds)`` paces the feed. Defaults are the real ones.

    ``realtime_factor`` is the honest headline: processing seconds per second of
    material. Below 1.0 means it keeps up.
    """

    streaming = True
    _POLICIES = ('drop', 'lag', 'error')

    def __init__(self, inner: 'WindowedStreamSession', *, policy: str = 'lag',
                 clock=None, sleep=None):
        if policy not in self._POLICIES:
            raise ValueError(f"policy must be one of {self._POLICIES}, got {policy!r}")
        self.inner = inner
        self.policy = policy
        self.record = inner.record
        self.time_bins = inner.time_bins
        self.requested_output_channels = inner.requested_output_channels
        self.emitted: list[StreamEvent] = []
        import time as _time
        self._clock = clock or _time.monotonic
        self._sleep = sleep or _time.sleep
        self._t0 = None
        self._last_emit_at = None
        self.windows_dropped = 0
        self.windows_delivered = 0
        self.max_lag_ms = 0.0
        self.processing_ms = 0.0

    def next_input(self) -> Optional[StreamEvent]:
        now = self._clock()
        if self._t0 is None:
            self._t0 = now
        else:
            # time spent by the caller since the last window came out = processing time
            self.processing_ms += (now - self._last_emit_at) * 1000.0

        event = self.inner.next_input()
        while event is not None:
            elapsed_ms = (self._clock() - self._t0) * 1000.0
            lag_ms = elapsed_ms - event.meta['window_start_ms']
            if lag_ms > self.inner.window_ms and self.policy != 'lag':
                if self.policy == 'error':
                    raise RuntimeError(
                        f"real-time budget missed: {lag_ms:.0f} ms behind at window "
                        f"{event.meta['window_index']} (window is "
                        f"{self.inner.window_ms:.0f} ms). Model is too slow for this "
                        f"feed; use policy='drop' to skip, or 'lag' to run offline.")
                self.windows_dropped += 1
                event = self.inner.next_input()
                continue
            self.max_lag_ms = max(self.max_lag_ms, lag_ms)
            # pace: do not hand over a window before the world has produced it
            wait_s = (event.meta['window_start_ms'] - elapsed_ms) / 1000.0
            if wait_s > 0:
                self._sleep(wait_s)
            self.windows_delivered += 1
            self._last_emit_at = self._clock()
            return event

        self._last_emit_at = self._clock()
        return None

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)

    @property
    def realtime_factor(self) -> float:
        """Processing seconds per second of material. < 1.0 keeps up."""
        material_ms = (self.windows_delivered + self.windows_dropped) * self.inner.stride_ms
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
