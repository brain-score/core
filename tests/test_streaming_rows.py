"""A closed-loop step is a row: channel -> payload. Rows and events convert both ways."""
import numpy as np

from brainscore_core.streaming import (
    CONTROL_CHANNEL, NamedStreams, StreamEvent, events_row, row_events,
)


def test_row_round_trips_through_events_with_two_cameras_and_reward():
    frame_a, frame_b = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)
    row = {
        'vision:image': NamedStreams({'wrist': frame_a, 'exterior': frame_b}),
        'proprioception': {'joint_position': [0.0, 1.0]},
        'instruction': 'pick up the cup',
        'reward': 1.5,
    }
    events = row_events(row, 40.0, meta={'reward': {'discount': 0.9}}, step_num=3)
    assert [e.channel for e in events] == [
        'vision:image', 'vision:image', 'proprioception', 'instruction', 'reward']
    cameras = [e for e in events if e.channel == 'vision:image']
    assert {e.meta['camera'] for e in cameras} == {'wrist', 'exterior'}
    assert {e.t_ms for e in events} == {40.0}
    assert events[-1].meta == {'step_num': 3, 'discount': 0.9}
    # Control events are not row entries.
    events.insert(0, StreamEvent(CONTROL_CHANNEL, 'begin', 40.0))
    back = events_row(events)
    assert set(back) == set(row)
    np.testing.assert_array_equal(back['vision:image']['wrist'], frame_a)
    np.testing.assert_array_equal(back['vision:image']['exterior'], frame_b)
    assert back['proprioception'] == row['proprioception']
    assert back['instruction'] == 'pick up the cup' and back['reward'] == 1.5


def test_two_unnamed_events_on_one_channel_are_an_error():
    import pytest
    with pytest.raises(ValueError, match='Two events'):
        events_row([StreamEvent('text', 'a', None), StreamEvent('text', 'b', None)])


def test_a_dict_payload_is_one_stream_not_named_streams():
    events = row_events({'observation': {'frame': 1, 'ascii': 'x'}})
    assert len(events) == 1 and events[0].payload == {'frame': 1, 'ascii': 'x'}
