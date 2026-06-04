"""Tests for the communicative Message type + output-type metric dispatch.

Message is the symmetry-closing member: valid as both an InputEvent and an
OutputEvent, the named multi-agent enabler. process(Message) routes to the
agent's action_fn.
"""
from typing import get_args

import pytest

from brainscore_core.model_interface import (
    BrainScoreModel, Message, EnvironmentResponse, EnvironmentStep,
    PerturbationApplied, Selection, Perturbation,
    InputEvent, OutputEvent, output_event_kind, dispatch_metric,
)


class TestMessageType:
    def test_fields_and_defaults(self):
        m = Message(content='hello')
        assert m.content == 'hello'
        assert m.sender is None and m.recipient is None and m.metadata == {}
        m2 = Message(content=[1, 2], sender='a', recipient='b', metadata={'k': 1})
        assert m2.sender == 'a' and m2.recipient == 'b' and m2.metadata['k'] == 1

    def test_message_is_in_both_unions(self):
        # the dual membership is the whole point — emit on one side, consume on the other
        assert Message in get_args(InputEvent)
        assert Message in get_args(OutputEvent)


class TestOutputEventKind:
    def test_environment_and_message_and_perturbation(self):
        assert output_event_kind(EnvironmentResponse(action=[0.0])) == 'environment'
        assert output_event_kind(Message(content='x')) == 'message'
        pa = PerturbationApplied(
            handle_id='h', target=Selection(layer='L', indices=[0]),
            perturbation=Perturbation(kind='zero'))
        assert output_event_kind(pa) == 'perturbation'

    def test_assembly_kinds_via_duck_typing(self):
        # core can't import brainio's concrete classes; classification is by
        # MRO class name, so locally-named stand-ins resolve correctly.
        class NeuroidAssembly:  # noqa: N801
            pass

        class BehavioralAssembly:  # noqa: N801
            pass

        assert output_event_kind(NeuroidAssembly()) == 'neural'
        assert output_event_kind(BehavioralAssembly()) == 'behavioral'

    def test_unknown(self):
        assert output_event_kind(object()) == 'unknown'


class TestDispatchMetric:
    def test_dispatch_by_kind(self):
        metric_map = {'message': 'M', 'behavioral': 'B'}
        assert dispatch_metric(Message(content='x'), metric_map) == 'M'

    def test_missing_kind_raises(self):
        with pytest.raises(KeyError, match='neural'):
            dispatch_metric(EnvironmentResponse(action=[0.0]), {'neural': 'N'})


class TestProcessMessage:
    def test_routes_to_action_fn_returning_message(self):
        agent = BrainScoreModel(
            'responder', None, {}, {}, None,
            action_fn=lambda step: Message(
                content=f'reply<{step.observation.content}>', sender='responder'))
        out = agent.process(Message(content='ping', sender='peer'))
        assert isinstance(out, Message)
        assert out.content == 'reply<ping>' and out.sender == 'responder'

    def test_accepts_environment_response_return(self):
        agent = BrainScoreModel('a', None, {}, {}, None,
                                action_fn=lambda step: EnvironmentResponse(action=1))
        assert isinstance(agent.process(Message(content='x')), EnvironmentResponse)

    def test_no_action_fn_raises(self):
        model = BrainScoreModel('m', None, {}, {}, None)
        with pytest.raises(NotImplementedError, match='action_fn'):
            model.process(Message(content='x'))

    def test_bad_return_type_raises(self):
        agent = BrainScoreModel('a', None, {}, {}, None,
                                action_fn=lambda step: 'raw string')
        with pytest.raises(TypeError, match='Message or EnvironmentResponse'):
            agent.process(Message(content='x'))
