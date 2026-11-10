import pytest
from brainscore_core import io_catalog
from brainscore_core.extensions import StreamEvent


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', {})


def test_environment_action_validation_precedes_step():
    from brainscore_core.streaming_helpers import EnvironmentSession
    from brainscore_core.events import EnvironmentStep
    class Environment:
        def reset(self):
            return EnvironmentStep(context={'t_ms': 125., 'time_source': 'sensor', 'time_inferred': False})
        def step(self, action):
            pytest.fail('invalid action reached the environment')
    def validate(action):
        raise ValueError('out of bounds')
    session = EnvironmentSession(Environment(), action_validator=validate)
    session.next_input()
    assert session.input_events[0].t_ms == 125
    with pytest.raises(ValueError, match='out of bounds'):
        session.emit(StreamEvent('motor', [99], 125))
    assert session.emitted == []
