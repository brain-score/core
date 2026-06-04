"""OutputEvent union contract.

`process()` returns one of a declared set of output shapes, symmetric with
InputEvent. These tests pin the contract so a new output type is added as a
union member (no Subject ABC change). They are pure introspection — no model
weights, no BrainIO import — so they run anywhere.
"""
import inspect
from typing import ForwardRef, get_args

from brainscore_core.model_interface import (
    Subject,
    BrainScoreModel,
    InputEvent,
    OutputEvent,
    EnvironmentResponse,
    PerturbationApplied,
)


def _member_names(union):
    """Names of a Union's members, handling both real classes and ForwardRefs."""
    names = []
    for arg in get_args(union):
        if isinstance(arg, ForwardRef):
            names.append(arg.__forward_arg__)
        else:
            names.append(getattr(arg, '__name__', str(arg)))
    return names


def test_output_event_has_the_current_members():
    names = _member_names(OutputEvent)
    assert names == [
        'NeuroidAssembly',       # neural recording
        'BehavioralAssembly',    # behavioral readout / generation
        'EnvironmentResponse',   # embodied step
        'PerturbationApplied',   # state-change acknowledgement
        'Message',               # communicative event (also an InputEvent member)
    ]


def test_output_event_includes_concrete_runtime_types():
    """The two members that are real classes today are present by identity —
    these are exactly what the embodied / state-change dispatch paths return
    (proven at runtime in test_environment_step.py / test_state_change.py)."""
    args = get_args(OutputEvent)
    assert EnvironmentResponse in args
    assert PerturbationApplied in args


def test_output_event_assembly_members_are_forward_refs():
    """The xarray assembly types stay forward refs so core needs no hard
    BrainIO import; resolving them is the benchmark layer's job."""
    fwd = [a for a in get_args(OutputEvent) if isinstance(a, ForwardRef)]
    assert {f.__forward_arg__ for f in fwd} == {'NeuroidAssembly', 'BehavioralAssembly'}


def test_subject_process_returns_output_event():
    assert Subject.process.__annotations__.get('return') is OutputEvent


def test_brainscoremodel_process_returns_output_event():
    assert BrainScoreModel.process.__annotations__.get('return') is OutputEvent


def test_process_input_is_annotated_input_event():
    """The ABC's input parameter is named input_event (not the old `stimuli`)
    and annotated InputEvent — the signature is honest about the generic input."""
    sig = inspect.signature(Subject.process)
    params = list(sig.parameters)
    assert params == ['self', 'input_event']
    assert Subject.process.__annotations__.get('input_event') is InputEvent
