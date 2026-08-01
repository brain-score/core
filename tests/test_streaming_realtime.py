"""Tier 3 real-time streaming: the feed advances whether or not the model is ready.

Tiers 1 and 2 change when inputs must exist. This tier adds wall-clock pressure, which
is the constraint that actually decides whether a streaming claim is real. The three
policies measure genuinely different things and the tests pin each:

  drop  -> stays current, loses coverage      (what a live system does)
  lag   -> complete coverage, unbounded delay (what an offline run does)
  error -> refuses to pretend                 (for a real-time-only benchmark)

The clock is injected, so none of this sleeps.
"""
import pytest

from brainscore_core.streaming_helpers import (
    RealTimeStreamSession,
    WindowedStreamSession,
)


class _FakeClock:
    """Manual clock in seconds. `work(ms)` is the model spending time."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        if seconds > 0:
            self.t += seconds

    def work(self, ms):
        self.t += ms / 1000.0


def _feed(n):
    return (f'frame_{i}.png' for i in range(n))


def _windowed(n_frames=90, window_ms=1000):
    # 30 fps, 1000 ms windows -> 30 frames each
    return WindowedStreamSession(_feed(n_frames), fps=30, window_ms=window_ms,
                                 record='IT')


@pytest.mark.unit
def test_fast_model_keeps_up_and_drops_nothing():
    clock = _FakeClock()
    rt = RealTimeStreamSession(_windowed(), policy='drop',
                               clock=clock, sleep=clock.sleep)
    while True:
        event = rt.next_input()
        if event is None:
            break
        clock.work(100)          # 100 ms to process 1000 ms of material
    report = rt.report()
    assert report['windows_dropped'] == 0
    assert report['windows_delivered'] == 3
    assert report['realtime_factor'] < 1.0
    assert report['kept_up'] is True


@pytest.mark.unit
def test_slow_model_with_drop_policy_skips_to_stay_current():
    clock = _FakeClock()
    rt = RealTimeStreamSession(_windowed(n_frames=300), policy='drop',
                               clock=clock, sleep=clock.sleep)
    delivered = 0
    while True:
        event = rt.next_input()
        if event is None:
            break
        delivered += 1
        clock.work(3000)         # 3 s to process 1 s of material -> 3x too slow
    report = rt.report()
    assert report['windows_dropped'] > 0, "a 3x-too-slow model should be dropping"
    assert report['realtime_factor'] > 1.0
    assert report['kept_up'] is False
    assert delivered == report['windows_delivered']


@pytest.mark.unit
def test_lag_policy_never_drops_even_when_slow():
    """Offline semantics: complete coverage, delay allowed to grow."""
    clock = _FakeClock()
    inner = _windowed(n_frames=150)          # 5 windows
    rt = RealTimeStreamSession(inner, policy='lag', clock=clock, sleep=clock.sleep)
    while True:
        event = rt.next_input()
        if event is None:
            break
        clock.work(5000)                      # hopelessly slow
    report = rt.report()
    assert report['windows_dropped'] == 0, "lag policy must not drop material"
    assert report['windows_delivered'] == 5
    assert report['max_lag_ms'] > 0


@pytest.mark.unit
def test_error_policy_refuses_when_the_budget_is_missed():
    clock = _FakeClock()
    rt = RealTimeStreamSession(_windowed(n_frames=300), policy='error',
                               clock=clock, sleep=clock.sleep)
    with pytest.raises(RuntimeError, match="real-time budget missed"):
        while True:
            event = rt.next_input()
            if event is None:
                break
            clock.work(4000)


@pytest.mark.unit
def test_realtime_factor_is_processing_over_material():
    clock = _FakeClock()
    rt = RealTimeStreamSession(_windowed(n_frames=90), policy='lag',
                               clock=clock, sleep=clock.sleep)
    while True:
        event = rt.next_input()
        if event is None:
            break
        clock.work(500)          # 0.5 s per 1 s window -> factor ~0.5
    assert rt.realtime_factor == pytest.approx(0.5, abs=0.05)


@pytest.mark.unit
def test_pacing_does_not_deliver_a_window_before_the_world_produced_it():
    """A fast model must WAIT, not race ahead of the feed."""
    clock = _FakeClock()
    rt = RealTimeStreamSession(_windowed(n_frames=90), policy='lag',
                               clock=clock, sleep=clock.sleep)
    rt.next_input()                       # window 0 at t=0
    rt.next_input()                       # window 1 starts at 1000 ms
    assert clock.t >= 1.0, (
        f"clock only advanced to {clock.t}s; the second window was handed over "
        "before the feed could have produced it")


@pytest.mark.unit
def test_invalid_policy_fails_fast():
    with pytest.raises(ValueError, match="policy must be one of"):
        RealTimeStreamSession(_windowed(), policy='whatever')
