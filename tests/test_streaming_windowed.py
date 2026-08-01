"""Tier 2 windowed streaming: an open-ended feed consumed in fixed windows.

Tier 1 proved the perception path can run one stimulus at a time, but batch size
collapses to 1. Tier 2 restores batching without ever holding the whole feed: pull
frames, hold at most one window, emit, slide.

The properties that matter, and that these tests pin:
  * memory held is bounded by the window, not by the length of the feed
  * the feed is never listed (a lazy decoder stays lazy)
  * every window carries the timing a benchmark needs to align it to brain time
  * a trailing partial window is emitted, not silently dropped
"""
import numpy as np
import pytest

from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    WindowedStreamSession,
    _drive_neural_session_streaming,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly


class _RecordingSubject:
    def __init__(self):
        self.window_sizes = []

    def start_recording(self, region, time_bins=None, recording_type=None):
        pass

    def process(self, stimuli, multi_modality=False):
        self.window_sizes.append(len(stimuli))
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(
            np.ones((len(ids), 2)),
            coords={'stimulus_id': ('presentation', ids),
                    'neuroid_id': ('neuroid', ['n0', 'n1'])},
            dims=['presentation', 'neuroid'])


def _counting_feed(n, counter):
    """A generator that records how many frames were actually pulled."""
    for i in range(n):
        counter['pulled'] += 1
        yield f'frame_{i:03d}.png'


@pytest.mark.unit
def test_windows_batch_frames_instead_of_one_at_a_time():
    counter = {'pulled': 0}
    # 30 fps, 1000 ms window -> 30 frames per window; 90 frames -> 3 windows
    session = WindowedStreamSession(_counting_feed(90, counter), fps=30,
                                    window_ms=1000, record='IT')
    subject = _RecordingSubject()
    _drive_neural_session_streaming(subject, session)
    assert subject.window_sizes == [30, 30, 30]
    assert session.windows_emitted == 3


@pytest.mark.unit
def test_memory_held_is_bounded_by_the_window_not_the_feed():
    """The whole point: a 10x longer feed must not hold 10x more."""
    held = []
    for n_frames in (60, 600):
        counter = {'pulled': 0}
        session = WindowedStreamSession(_counting_feed(n_frames, counter), fps=30,
                                        window_ms=1000, record='IT')
        _drive_neural_session_streaming(_RecordingSubject(), session)
        held.append(session.max_frames_held)
    assert held[0] == held[1] == 30, (
        f"frames held grew with feed length ({held}); the feed is being buffered")


@pytest.mark.unit
def test_feed_is_consumed_lazily():
    """A lazy decoder must stay lazy -- the session must not list() the iterator."""
    counter = {'pulled': 0}
    session = WindowedStreamSession(_counting_feed(300, counter), fps=30,
                                    window_ms=1000, record='IT')
    # pull exactly one window, then stop
    first = session.next_input()
    assert first is not None
    assert counter['pulled'] == 30, (
        f"pulled {counter['pulled']} frames to build one 30-frame window; "
        "the feed was drained ahead of demand")


@pytest.mark.unit
def test_each_window_carries_alignment_timing():
    counter = {'pulled': 0}
    session = WindowedStreamSession(_counting_feed(90, counter), fps=30,
                                    window_ms=1000, record='IT')
    windows = []
    event = session.next_input()
    while event is not None:
        windows.append(event)
        event = session.next_input()
    assert [w.meta['window_start_ms'] for w in windows] == [0.0, 1000.0, 2000.0]
    assert windows[0].meta['window_end_ms'] == pytest.approx(1000.0)
    # t_ms is the window start, which is what a benchmark aligns against brain time
    assert [w.t_ms for w in windows] == [0.0, 1000.0, 2000.0]


@pytest.mark.unit
def test_overlapping_windows_slide_by_stride():
    counter = {'pulled': 0}
    # 1000 ms window, 500 ms stride -> 50% overlap
    session = WindowedStreamSession(_counting_feed(90, counter), fps=30,
                                    window_ms=1000, stride_ms=500, record='IT')
    starts = []
    event = session.next_input()
    while event is not None:
        starts.append(event.meta['window_start_ms'])
        event = session.next_input()
    assert starts[:3] == [0.0, 500.0, 1000.0]


@pytest.mark.unit
def test_trailing_partial_window_is_emitted_not_dropped():
    """70 frames at 30/window = 2 full + 10 left over. Those 10 are real material."""
    counter = {'pulled': 0}
    session = WindowedStreamSession(_counting_feed(70, counter), fps=30,
                                    window_ms=1000, record='IT')
    subject = _RecordingSubject()
    _drive_neural_session_streaming(subject, session)
    assert subject.window_sizes == [30, 30, 10]
    partial = [e for e in session.emitted if e.channel == 'neural:IT']
    assert len(partial) == 3


@pytest.mark.unit
def test_custom_window_converter_is_used():
    """A native-video model packs a window into ONE clip row, not N frame rows."""
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    def as_single_clip(frames, start_ms, end_ms):
        return StimulusSet(pd.DataFrame({
            'stimulus_id': [f'clip_{int(start_ms)}'],
            'video_path': ['/tmp/clip.mp4'],
            'n_frames': [len(frames)],
        }))

    counter = {'pulled': 0}
    session = WindowedStreamSession(_counting_feed(60, counter), fps=30, window_ms=1000,
                                    record='IT', window_to_stimuli=as_single_clip)
    subject = _RecordingSubject()
    _drive_neural_session_streaming(subject, session)
    assert subject.window_sizes == [1, 1]


@pytest.mark.unit
def test_invalid_window_config_fails_fast():
    for kwargs in ({'fps': 0, 'window_ms': 1000},
                   {'fps': 30, 'window_ms': 0},
                   {'fps': 30, 'window_ms': 1000, 'stride_ms': 0}):
        with pytest.raises(ValueError):
            WindowedStreamSession(iter([]), record='IT', **kwargs)
