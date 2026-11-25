"""The act forms of the motor family: motor:key and motor:pointer (Contract Rules 3.7, 5.15)."""
import pytest

from brainscore_core import io_catalog
from brainscore_core.io_catalog import CatalogEntry, OUTPUT
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import EnvironmentSource


def problems(channel, payload):
    return io_catalog.validate(channel, payload, event=StreamEvent(channel, payload, None))


@pytest.mark.parametrize('channel, payload', [
    ('motor:key', [{'act': 'press', 'key': 'Space'}, {'act': 'wait', 'ms': 50},
                   {'act': 'release', 'key': 'Space'}, {'act': 'type', 'text': 'yes'}]),
    ('motor:pointer', [{'act': 'move', 'x': 10, 'y': 20.5}, {'act': 'down'},
                       {'act': 'stroke', 'points': [[30, 40], [50, 60]], 'force': 0.5}, {'act': 'up'}]),
    ('motor:key', [{'act': 'look', 'id': 'after-start'}]),
    ('motor:key', []),          # no act this step
    ('motor:pointer', None),    # an answer with no act
    ('motor', 3),               # the bare channel is the environment's to check
])
def test_a_list_of_acts_of_the_form_passes(channel, payload):
    assert problems(channel, payload) == []


@pytest.mark.parametrize('channel, payload, named', [
    ('motor:key', {'act': 'press', 'key': 'Space'}, 'list of acts'),
    ('motor:key', [{'act': 'move', 'x': 1, 'y': 2}], 'move'),             # a pointer act on the key form
    ('motor:pointer', [{'act': 'press', 'key': 'Space'}], 'press'),
    ('motor:pointer', [{'act': 'move', 'x': 1}], 'move takes'),
    ('motor:pointer', [{'act': 'move', 'x': float('nan'), 'y': 2}], 'finite'),
    ('motor:key', [{'act': 'wait', 'ms': -5}], '0 or more'),
    ('motor:key', [{'act': 'press', 'key': ''}], 'nonempty'),
    ('motor:pointer', [{'act': 'look', 'id': 'a\nb'}], 'one line'),
    ('motor:pointer', [{'act': 'stroke', 'points': [[1, 2, 3]]}], 'pairs'),
    ('motor:pointer', [{'act': 'stroke', 'points': [[1, 2]], 'force': 2}], 'force'),
])
def test_a_wrong_act_is_named(channel, payload, named):
    found = problems(channel, payload)
    assert found and named in ' '.join(found)


def test_only_the_listed_forms_are_motor_channels():
    assert io_catalog.form_problem('motor:key') is None and io_catalog.form_problem('motor') is None
    assert 'motor:pointer' in io_catalog.form_problem('motor:mouse')


def test_core_defines_the_motor_family():
    with pytest.raises(ValueError, match='defined by core'):
        io_catalog.register_channel(CatalogEntry('motor', OUTPUT, 'x', 'x', 'x'), replace=True)


def test_a_session_on_motor_key_refuses_a_wrong_act_before_the_environment():
    class Screen:
        def __init__(self):
            self.actions = []

        def reset(self):
            return {'observation': 0}

        def step(self, outputs):
            self.actions.append(outputs['motor:key'])
            return {'observation': 1}, 'terminal'
    env = Screen()
    session = Session(EnvironmentSource(env, action_channel='motor:key'), ('motor:key',)).declare_inputs(
        {'control', 'observation'})
    while (event := session.next_input()) is not None and event.payload != 'respond':
        pass
    with pytest.raises(ValueError, match='move'):
        session.respond(StreamEvent('motor:key', [{'act': 'move', 'x': 1, 'y': 2}], None))
    assert env.actions == []
