"""The ``behavior`` family is closed (Response Format Standard, sections 11.2 and 12.3 step 3).

Its addresses are one list in the catalog; a request or a declaration outside
it is malformed. Only core defines the family. A task takes the core response
classes by identity, never a subclass.
"""
import pytest

from brainscore_core import io_catalog
from brainscore_core.compatibility import validate_channels
from brainscore_core.contract import TaskContext
from brainscore_core.extensions import register_channel
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming import Session, StreamEvent

ADDRESSES = ('behavior', 'behavior:choice', 'behavior:probabilities',
             'behavior:log_probabilities', 'behavior:rating', 'behavior:rt', 'behavior:text')


def test_the_closed_list_is_the_formats_that_exist():
    # A closed list: core alone defines the behavior formats, so a new format edits this test.
    assert io_catalog.CHANNEL_FORMS['behavior'] == (
        None, 'choice', 'probabilities', 'log_probabilities', 'rating', 'rt', 'text'), (
        'the behavior formats changed. A new behavior format is added to CHANNEL_FORMS in '
        'brainscore_core/io_catalog.py, with its record schema in '
        'brainscore_core/response_formats.py and a line in this list.')
    for channel in ADDRESSES:
        assert io_catalog.form_problem(channel) is None


@pytest.mark.parametrize('channel', ['behavior:logprobs', 'behavior:answer', 'behavior:Choice',
                                     'behavior:confidence'])
def test_an_address_outside_the_list_is_refused_everywhere(channel):
    assert 'closed list' in io_catalog.form_problem(channel)
    with pytest.raises(ValueError, match='closed list'):
        validate_channels({channel}, io_catalog.OUTPUT, 'subject out_channels')
    with pytest.raises(ValueError, match='closed list'):
        Session((), [channel])
    session = Session((), ['behavior:choice'])
    with pytest.raises(ValueError, match='not requested'):
        session.respond(StreamEvent(channel, 'cat', None))
    unrequested = Session(())            # no outputs requested: still checked
    with pytest.raises(ValueError, match='closed list'):
        unrequested.respond(StreamEvent(channel, 'cat', None))
    assert unrequested.emitted == []


def test_log_probabilities_is_an_address_a_session_and_a_subject_can_name():
    assert io_catalog.form_problem('behavior:log_probabilities') is None
    validate_channels({'behavior:log_probabilities'}, io_catalog.OUTPUT, 'subject out_channels')
    assert Session((), ['behavior:log_probabilities']).outputs == ('behavior:log_probabilities',)


def test_bare_vision_keeps_its_own_message():
    assert 'names its form' in io_catalog.form_problem('vision')


@pytest.mark.parametrize('name', ['behavior', 'behavior:log_probabilities'])
def test_the_behavior_family_cannot_be_registered_or_replaced(monkeypatch, name):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    entry = io_catalog.CatalogEntry(name, io_catalog.OUTPUT, 'a plugin', 'anything', 'a plugin')
    for replace in (False, True):
        with pytest.raises(ValueError, match='defined by core'):
            register_channel(entry, replace=replace)
    assert io_catalog.get('behavior').owner == 'core'


def test_other_families_can_still_be_registered(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    entry = io_catalog.CatalogEntry('gaze', io_catalog.OUTPUT, 'a plugin', 'x, y', 'a plugin')
    assert register_channel(entry) is entry


@pytest.mark.parametrize('base, arguments', [(Choice, (['cat', 'dog'],)), (Scale, (1, 7)), (Text, ())])
def test_a_response_class_is_checked_by_identity(base, arguments):
    class Renamed(base):
        pass
    Renamed.__name__ = base.__name__      # the same name does not pass
    with pytest.raises(TypeError, match='not a subclass'):
        TaskContext(response=Renamed(*arguments))
    assert TaskContext(response=base(*arguments)).response is not None
