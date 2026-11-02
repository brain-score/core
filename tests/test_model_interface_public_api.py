"""Public API guard for ``brainscore_core.model_interface``.

The model-interface module is a compatibility facade: collaborators can import
these names from here even as the implementation moves into smaller modules.
This test snapshots the pre-decomposition public surface.
"""

import brainscore_core.model_interface as model_interface


EXPECTED_PUBLIC_NAMES = {
    'ABC',
    'Any',
    'BrainScoreModel',
    'Callable',
    'CameraFrame',
    'CompositeSelector',
    'Dict',
    'EnvironmentResponse',
    'EnvironmentStep',
    'Extractor',
    'FunctionalSelection',
    'IndexSelection',
    'InputEvent',
    'LayerSelector',
    'List',
    'Message',
    'Optional',
    'OutputEvent',
    'Perturbation',
    'PerturbationApplied',
    'Proprioception',
    'RandomSelection',
    'Selection',
    'Set',
    'StateChange',
    'Subject',
    'TYPE_CHECKING',
    'TaskContext',
    'Tuple',
    'UnifiedModel',
    'Union',
    'UnitSelection',
    'UnitSelector',
    'abstractmethod',
    'dataclass',
    'dispatch_metric',
    'field',
    'output_event_kind',
    'replace',
    'warnings',
}


def test_model_interface_public_surface_snapshot():
    public_names = {name for name in dir(model_interface)
                    if not name.startswith('_')}

    assert public_names == EXPECTED_PUBLIC_NAMES
    assert issubclass(model_interface.UnifiedModel, model_interface.Subject)


def test_brainscore_model_dispatch_order_snapshot():
    assert [
        (event_type.__name__, handler_name)
        for event_type, handler_name in model_interface.BrainScoreModel._INPUT_HANDLERS
    ] == [
        ('StateChange', '_dispatch_state_change'),
        ('EnvironmentStep', '_handle_environment_step'),
        ('Message', '_handle_message'),
    ]


# -- Public observation surface: ``extractor`` + ``detect_modalities`` --------
# Core-only fakes (no framework). Observation tools read these instead of the
# private ``_extractors`` / ``_detect_modalities``; neither is a Subject member.

import subprocess
import sys
from types import MappingProxyType

import pytest

from brainscore_core.model_interface import BrainScoreModel, Subject


class _Columns:
    def __init__(self, columns):
        self.columns = columns


class _FakeExtractor:
    """Extractor with the layer-aware signature and an episode reset."""

    def __init__(self, output):
        self.output, self.calls, self.resets = output, [], 0

    def __call__(self, stimuli, layers):
        self.calls.append((stimuli, list(layers)))
        return self.output

    def reset(self):
        self.resets += 1


def _forms():
    vision, text = _FakeExtractor('v-out'), _FakeExtractor('t-out')
    return {
        'extractor': (BrainScoreModel('m', extractor={'vision': vision, 'text': text}),
                      {'vision': vision, 'text': text}),
        'activations_model_mapping': (
            BrainScoreModel('m', activations_model={'vision': vision, 'text': text}),
            {'vision': vision, 'text': text}),
        'shared_activations_model': (
            BrainScoreModel('m', preprocessors={'vision': lambda x: x},
                            activations_model=vision),
            {'vision': vision}),
    }


@pytest.mark.parametrize('form', ['extractor', 'activations_model_mapping',
                                  'shared_activations_model'])
def test_extractor_is_read_only_normalized_mapping(form):
    model, expected = _forms()[form]
    mapping = model.extractor
    assert isinstance(mapping, MappingProxyType)
    assert set(mapping) == set(expected)
    assert all(mapping[k] is expected[k] for k in expected)  # same objects, no copies
    with pytest.raises(TypeError):
        mapping['audio'] = _FakeExtractor('a')
    with pytest.raises(TypeError):
        del mapping['vision']
    with pytest.raises(AttributeError):
        model.extractor = {}
    assert dict(model.extractor) == dict(mapping)  # unchanged by attempts


def test_extractor_is_not_a_subject_member():
    assert not hasattr(Subject, 'extractor')
    assert not hasattr(Subject, 'detect_modalities')


def test_canonical_and_old_forms_declare_same_channels():
    forms = _forms()
    new, old = forms['extractor'][0], forms['activations_model_mapping'][0]
    before = (new.in_channels, new.out_channels, new.supported_modalities)
    new.extractor  # reading the mapping must not change declarations
    assert (new.in_channels, new.out_channels, new.supported_modalities) == before
    assert before == (old.in_channels, old.out_channels, old.supported_modalities)
    assert new.supported_modalities == {'vision', 'text'}


@pytest.mark.parametrize('columns', [
    ['image_file_name'], ['image_path'], ['filename'], ['sentence'], ['text'],
    ['video_path'], ['audio_path'], ['mystery_col'], ['image_file_name', 'sentence'],
])
@pytest.mark.parametrize('form', ['extractor', 'activations_model_mapping',
                                  'shared_activations_model'])
def test_detect_modalities_matches_private_alias(form, columns):
    model = _forms()[form][0]
    stimuli = _Columns(columns)
    assert model.detect_modalities(stimuli) == model._detect_modalities(stimuli)
    assert model.detect_modalities(stimuli) == model._dispatcher.detect_modalities(stimuli)


def test_detect_modalities_private_alias_is_same_function():
    assert BrainScoreModel._detect_modalities is BrainScoreModel.detect_modalities


def test_extractor_mapping_survives_reset_and_reuse():
    vision = _FakeExtractor('v-out')
    model = BrainScoreModel('m', extractor={'vision': vision})
    mapping = model.extractor
    stimuli = _Columns(['image_path'])
    assert model.process(stimuli) == 'v-out'
    model.reset()
    assert vision.resets == 1
    assert model.extractor['vision'] is vision and dict(mapping) == {'vision': vision}
    assert model.process(stimuli) == 'v-out'
    assert [c[0] for c in vision.calls] == [stimuli, stimuli]


def test_extractor_access_imports_no_framework():
    code = (
        "import sys\n"
        "from brainscore_core.model_interface import BrainScoreModel\n"
        "m = BrainScoreModel('m', extractor={'text': lambda stimuli, layers: None})\n"
        "dict(m.extractor); m.detect_modalities(type('S', (), {'columns': ['sentence']})())\n"
        "bad = [n for n in ('torch', 'transformers', 'tensorflow', 'jax') if n in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)
