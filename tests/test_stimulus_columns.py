"""One table says which channel a stimulus column feeds (``io_catalog.stimulus_columns``).

A column registered with ``register_channel(entry, columns=...)`` feeds its channel
in a table session, in pre-flight and in the subject's projection of the table.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from brainscore_core import io_catalog
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.contract import Subject
from brainscore_core.extensions import CatalogEntry, register_channel
from brainscore_core.responses import PART_CHANNELS, part_columns
from brainscore_core.streaming_helpers import declared_only
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


@pytest.fixture
def odor(monkeypatch):
    """An external input family that owns the column ``odor_name``, and its output."""
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    register_channel(CatalogEntry('odor', 'input', 'StimulusSet column', 'an odorant name', 'test',
                                  payload_type=str), columns=('odor_name',))
    register_channel(CatalogEntry('odor_rating', 'output', 'a subject answer', 'a float', 'test',
                                  payload_type=float))


class Smeller(Subject):
    """Answers each odor it receives with the length of the odorant name."""

    def __init__(self, in_channels=frozenset({'odor'})):
        self._in = frozenset(in_channels)
        self.received = []
        self.table_columns = None

    @property
    def identifier(self):
        return 'smeller'

    @property
    def in_channels(self):
        return self._in

    @property
    def out_channels(self):
        return frozenset({'odor_rating'})

    def participate(self, session):
        session.declare_inputs(self.in_channels)
        self.table_columns = list(session.stimuli.columns)
        while (event := session.next_input()) is not None:
            if event.channel == 'odor':
                self.received.append(event)
                session.respond(StreamEvent('odor_rating', float(len(event.payload)),
                                            event.t_ms, meta=dict(event.meta)))

    def reset(self):
        self.received = []


def _odor_table():
    return StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b', 'c'],
                                     'odor_name': ['rose', 'lemon', 'smoke'],
                                     'intensity': [1, 2, 3]}))


def test_registered_column_feeds_its_channel_in_a_table_session(odor):
    assert io_catalog.column_channel('odor_name') == 'odor'
    session = stimulus_session(_odor_table(), ['odor_rating'])
    subject = Smeller()
    subject.participate(session)

    assert [event.payload for event in subject.received] == ['rose', 'lemon', 'smoke']
    assert [event.meta['column'] for event in subject.received] == ['odor_name'] * 3
    assert [event.payload for event in session.collect('odor_rating')] == [4.0, 5.0, 5.0]
    assert 'odor_name' in subject.table_columns


def test_preflight_sees_a_registered_column_channel(odor):
    benchmark = SimpleNamespace(identifier='odor-benchmark', required_input_channels={'odor'},
                                requested_output_channels={'odor_rating'})
    check_channel_compatibility(Smeller(), benchmark)


def test_a_subject_that_does_not_declare_the_channel_loses_the_column(odor):
    table = _odor_table()
    assert list(declared_only(table, {'text'}).columns) == ['stimulus_id', 'intensity']
    assert declared_only(table, {'odor'}) is table

    session = stimulus_session(table, ['odor_rating'])
    subject = Smeller(in_channels={'text'})
    subject.participate(session)
    assert subject.received == [] and 'odor_name' not in subject.table_columns


def test_columns_feed_only_a_new_input_family(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    with pytest.raises(ValueError, match='Only input channels'):
        register_channel(CatalogEntry('gaze', 'output', 'x', 'x', 'x'), columns=('gaze_x',))
    with pytest.raises(ValueError, match="names its form"):
        register_channel(io_catalog.get('vision'), columns=('frame',), replace=True)
    with pytest.raises(TypeError, match='sequence of column names'):
        register_channel(CatalogEntry('touch', 'input', 'x', 'x', 'x'), columns='touch_site')
    assert io_catalog.column_channel('gaze_x') is None
    assert io_catalog.column_channel('frame') is None


def test_registering_the_same_entry_again_changes_nothing(odor):
    entry = io_catalog.get('odor')
    register_channel(entry, columns=('odor_name',))
    assert io_catalog.get('odor') is entry and io_catalog.column_channel('odor_name') == 'odor'


def test_prompt_part_columns_are_the_columns_of_the_part_channel():
    table = io_catalog.stimulus_columns()
    for kind, channel in PART_CHANNELS.items():
        assert part_columns(kind) == tuple(c for c, fed in table.items() if fed == channel)
    # The order a prompt looks in: the first present column wins.
    assert part_columns('text') == ('sentence', 'text')
    assert part_columns('image')[:3] == ('image_file_name', 'image_path', 'filename')
    assert part_columns('video')[:2] == ('video_path', 'video_file_name')
    assert part_columns('audio')[:3] == ('audio_path', 'audio_file_name', 'audio_file')
