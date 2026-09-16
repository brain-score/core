import pytest
from brainscore_core import io_catalog
from brainscore_core.capabilities import Capability, capability_registry, register_capability
from brainscore_core.extensions import CatalogEntry, StreamEvent, register_channel
from brainscore_core.model_interface import BrainScoreModel


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', {})
    previous = dict(capability_registry)
    yield
    capability_registry.clear()
    capability_registry.update(previous)


def test_external_input_output_state_reset_and_session():
    for name, kind in (('test_request', 'input'), ('test_result', 'output')):
        register_channel(CatalogEntry(name, kind, 'StreamEvent', 'int', 'test',
                                     shape_validator=lambda x: isinstance(x, int)))
    setups = []
    class Tool(Capability):
        identifier = 'external-test'
        input_channels = {'test_request'}
        output_channels = {'test_result'}
        def enabled_for(self, model):
            return bool(model.capability_config.get(self.identifier))
        def setup(self, model):
            setups.append(model.identifier)
            return {'calls': 0}
        def handles(self, model, event, **kwargs):
            return isinstance(event, StreamEvent) and event.channel == 'test_request'
        def process(self, model, event, **kwargs):
            state = model.capability_state(self.identifier)
            state['calls'] += 1
            return StreamEvent('test_result', state['calls'] + event.payload, event.t_ms)
        def supports_session(self, model, channels):
            return set(channels) == {'test_result'}
        def interact(self, model, session):
            while (event := session.next_input()) is not None:
                session.emit(model.process(event))
    register_capability(Tool())
    model = BrainScoreModel('external', capability_config={'external-test': True})
    assert model.process(StreamEvent('test_request', 10, 5)).payload == 11
    assert len(setups) == 1
    model.reset()
    assert model.process(StreamEvent('test_request', 10, 5)).payload == 11
    assert len(setups) == 2 and 'test_result' in model.out_channels
    with pytest.raises(ValueError):
        model.process(StreamEvent('test_request', 'bad', 5))
    with pytest.raises(NotImplementedError):
        model.check_session_support({'test_result', 'motor'})


def test_external_modality_routes_without_host_modification():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    register_channel(CatalogEntry('depth_sensor', 'input', 'StimulusSet', 'depth', 'test'),
                     columns=['depth_sample'])
    calls = []
    def extract(model, stimuli, **kwargs):
        calls.append(stimuli)
        return 'measured'
    model = BrainScoreModel('depth', preprocessors={'depth_sensor': extract})
    assert model.process(StimulusSet({'depth_sample': ['fixture']})) == 'measured'
    assert len(calls) == 1
    with pytest.raises(ValueError, match='already belongs'):
        register_channel(CatalogEntry('another_sensor', 'input', 'StimulusSet', 'depth', 'test'),
                         columns=['depth_sample'])
    with pytest.raises(ValueError, match='already belongs'):
        register_channel(CatalogEntry('another_sensor', 'input', 'StimulusSet', 'depth', 'test'),
                         columns=['image_path'])


def test_nested_robot_camera_validation_precedes_policy():
    import numpy as np
    from brainscore_core.events import CameraFrame, EnvironmentStep
    model = BrainScoreModel('robot', action_fn=lambda step: pytest.fail('policy ran'))
    with pytest.raises(ValueError):
        model.process(EnvironmentStep(observation={'cameras': {
            'wrist': CameraFrame(rgb=np.zeros((8, 8, 3), dtype=np.float32))}}))


def test_session_benchmark_checks_execution_support_before_compute():
    from types import SimpleNamespace
    from brainscore_core.compatibility import check_channel_compatibility, CompatibilityError
    model = BrainScoreModel('motor-only', action_fn=lambda step: None)
    benchmark = SimpleNamespace(identifier='session', uses_session=True,
        required_input_channels=set(), requested_output_channels={'motor'})
    check_channel_compatibility(model, benchmark)
    model.check_session_support = lambda channels: (_ for _ in ()).throw(NotImplementedError('no driver'))
    with pytest.raises(CompatibilityError, match='no driver'):
        check_channel_compatibility(model, benchmark)


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
