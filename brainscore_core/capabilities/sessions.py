"""Built-in session handlers for BrainScoreModel's parts.

Each part of a BrainScoreModel (region map, answer or readout, policy,
intervention) handles some channels. These capabilities declare those
channels and run the matching sessions, registered the same way as any added
capability, so ``participate``, ``check_session_support`` and the declared
channels all read one registry. They sort after added capabilities
(``order`` 10000), so an added capability that claims the same outputs wins.
"""

from .base import Capability
from .registry import register_capability
from brainscore_core._sessions import helpers as _drivers


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
        # A region map names where to record; only a wired extractor can
        # record there, so a map without one declares nothing (rule 2.4).
        return bool(model.region_layer_map) and bool(model._extractors)

    def declared_output_channels(self, model):
        return {f'neural:{region}' for region in model.region_layer_map}

    def supports_session(self, model, channels) -> bool:
        return bool(channels) and all(c.startswith('neural:') for c in channels)

    def participate(self, model, session) -> None:
        task = getattr(session, 'task_context', None)
        if task is not None and task.task_type != 'passive':
            # The session's task decides what the subject does (rule 4.1):
            # record during the task; the answers are not sent.
            _drivers.drive_behavior_and_neural_session(model, session)
            return
        # A passive session records activity even after a task was configured.
        with model._behavioral.suspended():
            if getattr(session, 'streaming', False):
                _drivers._drive_neural_session_streaming(model, session)
            else:
                _drivers.drive_neural_session(model, session, allow_multi_modality=True)


class BehaviorSessions(_SessionCapability):
    identifier = 'behavior-sessions'

    def enabled_for(self, model) -> bool:
        return (model._behavioral_readout_layer is not None
                or model._generation_fn is not None
                or model._reply_fn is not None or model._likelihood_fn is not None)

    def declared_input_channels(self, model):
        replies = (model._generation_fn, model._reply_fn, model._likelihood_fn)
        return {'instruction'} if any(part is not None for part in replies) else set()

    def declared_output_channels(self, model):
        # Each part declares the kinds of answer it can give (rule 2.4).
        channels = {'behavior'}
        if model._behavioral_readout_layer is not None:
            channels |= {'behavior:choice', 'behavior:probabilities', 'behavior:rating'}
        if model._generation_fn is not None:
            channels |= {'behavior:choice'}
        if model._reply_fn is not None:
            channels |= {'behavior:choice', 'behavior:rating', 'behavior:confidence',
                         'behavior:text'}
        if model._likelihood_fn is not None:
            channels |= {'behavior:choice', 'behavior:probabilities'}
        return channels

    def supports_session(self, model, channels) -> bool:
        # Behavior alone, or behavior plus regions recorded during the same answers.
        behavior = {c for c in channels if c == 'behavior' or c.startswith('behavior:')}
        recorded = set(channels) - behavior
        return bool(behavior) and behavior <= self.declared_output_channels(model) and recorded <= {
            f'neural:{region}' for region in model.region_layer_map}

    def participate(self, model, session) -> None:
        requested = set(_drivers._requested_output_channels(session))
        task = getattr(session, 'task_context', None)
        if requested != {'behavior'} or (task is not None and task.response is not None):
            _drivers.drive_behavior_and_neural_session(model, session)
        elif getattr(session, 'streaming', False):
            _drivers._drive_behavior_session_streaming(model, session)
        else:
            _drivers.drive_behavior_session(model, session)


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

    def participate(self, model, session) -> None:
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

    def participate(self, model, session) -> None:
        _drivers._drive_state_change_session_via_process(model, session)


for _capability in (NeuralSessions(), BehaviorSessions(), MotorSessions(),
                    PerturbationSessions()):
    register_capability(_capability)
