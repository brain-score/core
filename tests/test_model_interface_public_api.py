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
    assert model_interface.UnifiedModel is model_interface.Subject


def test_brainscore_model_dispatch_order_snapshot():
    assert [
        (event_type.__name__, handler_name)
        for event_type, handler_name in model_interface.BrainScoreModel._INPUT_HANDLERS
    ] == [
        ('StateChange', '_dispatch_state_change'),
        ('EnvironmentStep', '_handle_environment_step'),
        ('Message', '_handle_message'),
    ]
