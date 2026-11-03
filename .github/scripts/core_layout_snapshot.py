"""Compare installed public APIs and fixed measurements during a pure refactor.

Run in CI against both wheels. This deliberately imports no model framework.
"""
import importlib
import inspect
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys

import numpy as np
import xarray as xr

from brainscore_core.environment import ArraySpec, MappingSpec
from brainscore_core.events import EnvironmentResponse, EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel, TaskContext
from brainscore_core.streaming_helpers import (
    EnvironmentSession, behavioral_response, neural_response,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
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
                signatures[key] = str(inspect.signature(value))
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


stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], 'sentence': ['one', 'two']})
stimuli.identifier = 'layout-control'
neural = BrainScoreModel(identifier='fixture-neural', extractor={'text': Extractor()},
                         region_layer_map={'language_system': 'fixture'})
neural_values = neural_response(neural, stimuli, record='language_system')
answer = BrainScoreModel(identifier='fixture-answer', supported_modalities={'text'},
                         answer=lambda stimulus_row, instruction, label_set: 'yes')
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
        return EnvironmentStep(
            observation={'state': np.array([self.count], dtype=np.float32)},
            step_num=self.count, is_first=self.count == 0, is_last=self.count == 2,
        )

    def step(self, action):
        self.count += 1
        return self.observation()


policy = BrainScoreModel(identifier='fixture-policy', policy=lambda step: EnvironmentResponse(
    action=step.observation['state'] + np.float32(0.25)))
with EnvironmentSession(environment=Environment(), seed=0, require_specs=True) as session:
    policy.participate(session)
    environment_values = {
        'inputs': encode(session.input_events),
        'proposals': encode(session.emitted),
        'receipts': encode(session.action_events),
        'actions': encode(session.collect()),
        'complete': session.complete,
    }

assert not {'torch', 'transformers', 'tensorflow', 'jax'} & sys.modules.keys()
Path(sys.argv[2]).write_text(json.dumps({
    'apis': apis,
    'measurements': {'neural': encode(neural_values), 'behavior': encode(behavior_values),
                     'environment': environment_values},
}, indent=2, sort_keys=True) + '\n')
print('Captured public imports/signatures and neural, behavioral, environment outputs')
