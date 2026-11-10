"""Compare installed public APIs and fixed measurements during a pure refactor.

Run in CI against both wheels. This deliberately imports no model framework.
"""
import importlib
import inspect
import json
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
import sys

import numpy as np
import xarray as xr

from brainscore_core.environment import ArraySpec, MappingSpec
from brainscore_core.model_interface import Subject, TaskContext
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import (
    EnvironmentSource, behavioral_response, drain_session_inputs, neural_response, session_stimuli,
)
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

manifest = json.loads(Path(sys.argv[1]).read_text())
paths = ['brainscore_core'] + ['brainscore_core.' + name for name in manifest]
# Populate every public module attribute before taking either inventory.
modules = {name: importlib.import_module(name) for name in paths}
apis = {}
for name, module in modules.items():
    public = {key: value for key, value in vars(module).items() if not key.startswith('_')}
    signatures = {}
    for key, value in public.items():
        if inspect.isfunction(value) or inspect.isclass(value):
            try:
                signature = inspect.signature(value)
                # This singleton's repr contains a process-specific address.
                parameters = [parameter.replace(default='<dataclasses.MISSING>')
                              if parameter.default is MISSING else parameter
                              for parameter in signature.parameters.values()]
                signatures[key] = str(signature.replace(parameters=parameters))
            except (TypeError, ValueError):
                pass
    apis[name] = {'names': sorted(public), 'signatures': signatures}


def encode(value):
    """Compare numeric bytes, coordinates and event fields; omit measured latency."""
    if isinstance(value, np.ndarray):
        content = {'values': encode(value.tolist())} if value.dtype.hasobject else {'bytes': value.tobytes().hex()}
        return {'dtype': str(value.dtype), 'shape': list(value.shape), **content}
    if isinstance(value, xr.DataArray):
        return {'type': type(value).__module__ + '.' + type(value).__name__,
                'data': encode(value.values), 'dims': list(value.dims),
                'attrs': encode(value.attrs),
                'coordinates': {key: encode(coord.values) for key, coord in value.coords.items()}}
    if is_dataclass(value):
        return {'type': type(value).__module__ + '.' + type(value).__name__,
                'fields': {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items() if key != 'latency_ms'}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Extractor:
    identifier = backbone_id = 'fixture-extractor'

    def __call__(self, stimuli, layers):
        return NeuroidAssembly(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'stimulus_index': ('presentation', [0, 1]),
                    'neuroid_id': ('neuroid', ['fixture.0', 'fixture.1']),
                    'layer': ('neuroid', ['fixture', 'fixture'])},
            dims=['presentation', 'neuroid'],
        )


class NeuralSubject(Subject):
    """Records one region with the fixture extractor; defined here, so it is the same in both wheels."""
    identifier = 'fixture-neural'
    in_channels = {'text'}
    out_channels = {'neural:language_system'}

    def participate(self, session):
        events = drain_session_inputs(session)
        output = Extractor()(session_stimuli(session, events), ['fixture'])
        session.respond(StreamEvent('neural:language_system', output, 0.0))


class AnswerSubject(Subject):
    """Answers 'yes' to each task stimulus."""
    identifier = 'fixture-answer'
    in_channels = {'text', 'instruction'}
    out_channels = {'behavior'}

    def participate(self, session):
        drain_session_inputs(session)
        ids = list(session.task_context.fitting_stimuli['stimulus_id'])
        output = BehavioralAssembly(['yes'] * len(ids), coords={'stimulus_id': ('presentation', ids)},
                                    dims=['presentation'])
        session.respond(StreamEvent('behavior', output, 0.0))


class PolicySubject(Subject):
    """Adds 0.25 to the observed state at each step."""
    identifier = 'fixture-policy'
    in_channels = {'observation'}
    out_channels = {'motor'}

    def participate(self, session):
        row = {}
        while (event := session.next_input()) is not None:
            if event.channel != 'control':
                row[event.channel] = event.payload
            elif event.payload == 'respond':
                action = row['observation']['state'] + np.float32(0.25)
                session.respond(StreamEvent('motor', action, None))


stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], 'sentence': ['one', 'two']})
stimuli.identifier = 'layout-control'
neural = NeuralSubject()
neural_values = neural_response(neural, stimuli, record='language_system')
answer = AnswerSubject()
task = TaskContext(task_type='label', label_set=['yes', 'no'], fitting_stimuli=stimuli,
                   instruction='Respond yes or no')
behavior_values = behavioral_response(answer, task)


class Environment:
    def observation_spec(self):
        return MappingSpec({'state': ArraySpec((1,), 'float32')})

    def action_spec(self):
        return ArraySpec((1,), 'float32')

    def reset(self, *, seed=None):
        self.count = 0
        return self.observation()

    def observation(self):
        return {'observation': {'state': np.array([self.count], dtype=np.float32)}}

    def step(self, outputs):
        self.count += 1
        return self.observation(), 'terminal' if self.count == 2 else None


policy = PolicySubject()
source = EnvironmentSource(Environment(), seed=0, require_specs=True)
with Session(source, ('motor',)) as session:
    policy.participate(session)
    environment_values = {
        'inputs': encode(source.input_events),
        'proposals': encode(session.emitted),
        'receipts': encode(source.action_events),
        'actions': encode(session.collect('motor')),
        'complete': session.complete,
    }

assert not {'torch', 'transformers', 'tensorflow', 'jax'} & sys.modules.keys()
Path(sys.argv[2]).write_text(json.dumps({
    'apis': apis,
    'measurements': {'neural': encode(neural_values), 'behavior': encode(behavior_values),
                     'environment': environment_values},
}, indent=2, sort_keys=True) + '\n')
print('Captured public imports/signatures and neural, behavioral, environment outputs')
