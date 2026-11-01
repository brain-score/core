"""Built-in session handlers for BrainScoreModel's parts.

Each part of a BrainScoreModel (region map, answer or readout, policy,
intervention) handles some channels. These capabilities declare those
channels and run the matching sessions, registered the same way as any added
capability, so ``interact``, ``check_session_support`` and the declared
channels all read one registry. They sort after added capabilities
(``order`` 10000), so an added capability that claims the same outputs wins.
"""

from .base import Capability
from .registry import register_capability
from .. import streaming_helpers as _drivers


class _SessionCapability(Capability):
    """A capability that only runs sessions; it handles no ``process`` events."""

    order = 10000

    def handles(self, model, event, **kwargs) -> bool:
        del model, event, kwargs
        return False

    def process(self, model, event, **kwargs):  # pragma: no cover - never handled
        raise NotImplementedError(self.identifier)


class NeuralSessions(_SessionCapability):
    identifier = 'neural-sessions'

    def enabled_for(self, model) -> bool:
        return bool(model.region_layer_map)

    def declared_output_channels(self, model):
        return {f'neural:{region}' for region in model.region_layer_map}

    def supports_session(self, model, channels) -> bool:
        return bool(channels) and all(c.startswith('neural:') for c in channels)

    def interact(self, model, session) -> None:
        if getattr(session, 'streaming', False):
            _drivers._drive_neural_session_streaming(model, session)
        else:
            _drivers._drive_neural_session_via_process(model, session,
                                                       allow_multi_modality=True)


class BehaviorSessions(_SessionCapability):
    identifier = 'behavior-sessions'

    def enabled_for(self, model) -> bool:
        return (model._behavioral_readout_layer is not None
                or model._generation_fn is not None)

    def declared_input_channels(self, model):
        return {'instruction'} if model._generation_fn is not None else set()

    def declared_output_channels(self, model):
        return {'behavior'}

    def supports_session(self, model, channels) -> bool:
        return set(channels) == {'behavior'}

    def interact(self, model, session) -> None:
        if getattr(session, 'streaming', False):
            _drivers._drive_behavior_session_streaming(model, session)
        else:
            _drivers._drive_behavior_session_via_process(model, session)


class MotorSessions(_SessionCapability):
    identifier = 'motor-sessions'

    def enabled_for(self, model) -> bool:
        return model._action_fn is not None

    def declared_input_channels(self, model):
        return {'observation'}

    def declared_output_channels(self, model):
        return {'motor'}

    def supports_session(self, model, channels) -> bool:
        return set(channels) == {'motor'}

    def interact(self, model, session) -> None:
        _drivers._drive_environment_session_via_process(model, session)


class PerturbationSessions(_SessionCapability):
    identifier = 'perturbation-sessions'

    def enabled_for(self, model) -> bool:
        return model._state_change_fn is not None

    def declared_input_channels(self, model):
        return {'lesion', 'stimulation'}

    def declared_output_channels(self, model):
        return {'perturbation'}

    def supports_session(self, model, channels) -> bool:
        return set(channels) == {'perturbation'}

    def interact(self, model, session) -> None:
        _drivers._drive_state_change_session_via_process(model, session)


for _capability in (NeuralSessions(), BehaviorSessions(), MotorSessions(),
                    PerturbationSessions()):
    register_capability(_capability)
