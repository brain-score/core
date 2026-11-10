"""Contract conformance for ``participate(session)`` (Contract Rules v3, rule 1).

A subject that implements only ``participate`` must run through every public
path: the static typed helpers, the legacy-benchmark facade and a closed-loop
environment. No class carries a second evaluation method.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.contract import Subject, TaskContext, UnifiedModel
from brainscore_core.environment import DiscreteSpec, MappingSpec, TextSpec
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    behavioral_response, neural_response, run_environment,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _images(n=2):
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_path': [f'/tmp/s{i}.png' for i in range(n)],
    }))
    stimuli.identifier = f'participate-images-{n}'
    return stimuli


def _neural(n):
    return NeuroidAssembly(np.arange(n, dtype=float)[:, None],
                           coords={'stimulus_id': ('presentation', [f's{i}' for i in range(n)]),
                                   'neuroid_id': ('neuroid', [0])},
                           dims=['presentation', 'neuroid'])


class _ParticipateOnly(Subject):
    """Implements only the contract: no process, look_at or start_recording."""

    identifier = 'participate-only'
    in_channels = {'vision', 'observation'}
    out_channels = {'neural:IT', 'behavior', 'motor'}

    def __init__(self):
        self.inputs = []
        self.payloads = []

    def participate(self, session):
        if 'motor' in session.requested_output_channels:
            # Closed loop: answer each respond request until the episode ends.
            while (event := session.next_input()) is not None:
                if event.channel != 'control':
                    self.inputs.append(event)
                elif event.payload == 'respond':
                    session.respond(StreamEvent('motor', len(self.inputs) % 3, None))
            return
        while (event := session.next_input()) is not None:
            self.inputs.append(event)
        n = len(session.stimulus_set)
        for channel in session.requested_output_channels:
            payload = (_neural(n) if channel.startswith('neural:') else
                       BehavioralAssembly(np.zeros((n, 1)), dims=['presentation', 'choice']))
            self.payloads.append(payload)
            session.respond(StreamEvent(channel, payload, None))


def test_subject_has_exactly_one_abstract_evaluation_method():
    assert Subject.__abstractmethods__ == {
        'identifier', 'in_channels', 'out_channels', 'participate'}
    assert not hasattr(Subject, 'process')
    # Typed bases add conveniences, never a second abstract evaluation method.
    assert {'participate', 'process'}.isdisjoint(UnifiedModel.__abstractmethods__)
    for cls in (Subject, UnifiedModel, LegacyBenchmarkFacade):
        assert not hasattr(cls, 'interact'), cls


def test_static_neural_helper_runs_a_participate_only_subject():
    subject = _ParticipateOnly()
    out = neural_response(subject, _images(3), record='IT')
    assert out is subject.payloads[0]
    assert len([e for e in subject.inputs if e.channel != 'control']) == 3


def test_static_behavior_helper_runs_a_participate_only_subject():
    subject = _ParticipateOnly()
    context = TaskContext(task_type='probabilities', label_set=['a', 'b'],
                          metadata={'stimulus_set': _images(2)})
    out = behavioral_response(subject, context)
    assert out is subject.payloads[0]


def test_legacy_facade_runs_a_participate_only_subject():
    subject = _ParticipateOnly()
    facade = LegacyBenchmarkFacade(subject, visual_degrees=8)
    facade.start_recording('IT', time_bins=[(70, 170)])
    out = facade.look_at(_images(2), number_of_trials=1)
    np.testing.assert_array_equal(out.values, [[0.], [1.]])
    assert not hasattr(subject, 'look_at') and not hasattr(subject, 'process')


class _Environment:
    """Three observations, then a terminal one."""

    def __init__(self):
        self.actions = []
        self.closed = 0

    def observation_spec(self):
        return MappingSpec({'task': TextSpec()})

    def action_spec(self):
        return DiscreteSpec(3)

    def _step(self):
        return {'observation': {'task': 'move'}}

    def reset(self, *, seed=None, options=None):
        return self._step()

    def step(self, outputs):
        self.actions.append(outputs['motor'])
        return self._step(), 'terminal' if len(self.actions) == 3 else None

    def close(self):
        self.closed += 1


def test_closed_loop_runs_a_participate_only_subject():
    subject, env = _ParticipateOnly(), _Environment()
    closed_loop = SimpleNamespace(identifier='closed-loop', uses_session=True,
                                  required_input_channels={'observation'},
                                  requested_output_channels={'motor'})
    match = check_channel_compatibility(subject, closed_loop)
    assert match.consumed_inputs == {'observation'} and match.kind == 'full'
    applied = run_environment(subject, env)
    assert env.actions == [1, 2, 0] and env.closed == 1
    assert applied == [1, 2, 0]
