import pytest
from brainscore_core import io_catalog
from brainscore_core.extensions import StreamEvent


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))


def test_environment_action_validation_precedes_step():
    from brainscore_core.streaming_helpers import EnvironmentSource
    from brainscore_core.streaming import Session
    class Environment:
        def clock(self):
            return {'t_ms': 125., 'time_source': 'sensor', 'time_inferred': False}
        def reset(self):
            return {'observation': 1}
        def step(self, action):
            pytest.fail('invalid action reached the environment')
    def validate(action):
        raise ValueError('out of bounds')
    source = EnvironmentSource(Environment(), action_validator=validate)
    session = Session(source, ('motor',)).declare_inputs({'control', 'observation'})
    while (event := session.next_input()).channel != 'control' or event.payload != 'respond':
        pass
    assert source.input_events[0].t_ms == 125
    with pytest.raises(ValueError, match='out of bounds'):
        session.respond(StreamEvent('motor', [99], 125))
    assert session.emitted == []
