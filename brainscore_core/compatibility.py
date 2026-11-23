"""
Pre-flight compatibility checker for Brain-Score (Contract Rules v3, rule 5).

Compatibility is decided from declarations alone: what the benchmark sends
and asks for, and what the subject says it takes in, produces and requires.
A subject is compatible with a benchmark when all four hold:

1. every requested output is in ``subject.out_channels``;
2. at least one benchmark input is in ``subject.in_channels``;
3. every channel in ``subject.required_channels`` is among the benchmark's
   inputs;
4. every requested output that ``subject.output_inputs`` lists can be answered:
   the subject takes in at least one of its listed inputs from the benchmark.
   A ``perturbation`` record answers an intervention, so it is answered only
   from the intervention inputs (the catalog's ``StateChange`` channels:
   ``lesion``, ``stimulation``, ``pharmacological``) the subject takes in. A
   subject that lists ``perturbation`` from another input has a malformed
   declaration (``ValueError``, at pre-flight).

The benchmark declares the inputs its human participants received. A subject
that takes in only some of them still runs: the streams it cannot take in
never enter it. :func:`check_compatibility` returns the match as data
(:class:`CompatibilityMatch`) so the score records which inputs were used.
An incompatible pair raises :class:`CompatibilityError`; ``score()`` returns
the N/A result for it (``brainscore_core.score_status.na_score``). A malformed
declaration raises ``ValueError``: it is a defect, not an incompatible pair. A
compatible pair whose run is refused when it starts (a task, outputs or settings
the subject cannot run) raises :class:`ExecutionNotFeasibleError`, an error. On an
old benchmark, which states no requested output, a refusal before any input of the
benchmark reached the subject is raised as :class:`CompatibilityError` (N/A) by the
legacy facade.

Benchmark inputs are ``required_input_channels`` when declared, otherwise the
channels of ``required_modalities`` plus ``accepted_modalities``. A benchmark with no
input channels is rejected. Requested
outputs are ``requested_output_channels`` when declared, otherwise
``neural:<region>`` for a benchmark with a ``region``.

Benchmarks that still set ``available_modalities`` emit a
``DeprecationWarning`` directing the maintainer to register separate variants.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set

from brainscore_core import io_catalog
from brainscore_core.contract import required_traits_for
from brainscore_core.io_catalog import canonical_modality, modalities_to_input_channels
from brainscore_core.contract import Subject
from brainscore_core.streaming import CONTROL_CHANNEL, channel_declared, parse_channel


class CompatibilityError(Exception):
    """Raised when the declarations of a subject and a benchmark do not fit.

    The declaration check raises it, before any compute, and ``score()`` turns it
    into the N/A result. The legacy facade raises it too, for a refusal of an old
    benchmark's task or recording before any input of the benchmark reached the
    subject. ``channels`` are the channels that cause the
    mismatch (for example the requested outputs the subject does not produce);
    empty when the reason is not a channel.
    """

    def __init__(self, message: str = '', channels=()):
        super().__init__(message)
        self.channels = frozenset(channels)


class ExecutionNotFeasibleError(Exception):
    """Raised when a subject or benchmark refuses one combination of task, outputs and settings.

    The pair passed the declaration check, and the refusal comes when a session
    or a benchmark run starts, before model computation. It is an error of the
    run; ``channels`` and ``task_type`` name what was refused. On an old benchmark, a
    refusal before any input reached the subject is N/A (the legacy facade raises it
    as :class:`CompatibilityError`).

    A refusal is raised before model computation where the fact is knowable at
    session start. A reply part that can detect an unsupported setting or
    stimulus part only when it is called raises it at that call.
    """

    def __init__(self, message: str = '', channels=(), task_type=None):
        super().__init__(message)
        self.channels = frozenset(channels)
        self.task_type = task_type


class CompatibilityWarning(UserWarning):
    """Issued when a model can run but will not use all available modalities."""
    pass


def _declares_channels(benchmark) -> bool:
    """A benchmark that declares its inputs or its requested outputs drives sessions itself."""
    return any(getattr(benchmark, name, None) is not None
               for name in ('required_input_channels', 'requested_output_channels'))


def ensure_legacy_benchmark_modalities(benchmark, modalities):
    """Supply a domain default only when a legacy benchmark has no input contract."""
    if (not _declares_channels(benchmark)
            and not getattr(benchmark, 'required_modalities', None)
            and not getattr(benchmark, 'accepted_modalities', None)):
        benchmark.required_modalities = set(modalities)
    return benchmark


@dataclass(frozen=True)
class CompatibilityMatch:
    """How a subject matches a benchmark, returned by pre-flight.

    ``consumed_inputs`` are the benchmark inputs the subject takes in and
    ``dropped_inputs`` the ones it cannot, which never enter it. ``kind`` is
    ``'full'`` when nothing is dropped, otherwise ``'partial'``.
    """
    subject: str
    benchmark: str
    consumed_inputs: FrozenSet[str]
    dropped_inputs: FrozenSet[str]
    requested_output_channels: FrozenSet[str]

    @property
    def kind(self) -> str:
        return 'partial' if self.dropped_inputs else 'full'

    def as_attrs(self) -> Dict[str, object]:
        """Plain-data form for ``score.attrs['compatibility']``."""
        return {
            'match': self.kind,
            'consumed_inputs': sorted(self.consumed_inputs),
            'dropped_inputs': sorted(self.dropped_inputs),
            'requested_output_channels': sorted(self.requested_output_channels),
        }


def _looks_like_channels(value) -> bool:
    """A nonempty collection of channel-name strings, not unrelated metadata."""
    return (isinstance(value, (set, frozenset, list, tuple)) and bool(value)
            and all(isinstance(item, str) for item in value))


# The channel that says what to do; it is not a stimulus (rule 5.2).
_INSTRUCTION = 'instruction'


def check_channel_compatibility(subject: Subject, benchmark) -> CompatibilityMatch:
    """Check the four compatibility conditions and return the match.

    The subject must take in at least one benchmark input that is a stimulus:
    an instruction alone does not make a pair applicable (rule 5.2).

    Decides from declarations only (rule 5.3). Raises ``TypeError`` for an object
    that is not a :class:`Subject`. Raises
    :class:`CompatibilityError` before compute when valid declarations do not
    fit, and ``ValueError`` when a declaration is malformed (a field spelled the
    way benchmarks do not read, no inputs, a bad channel name, direction or form,
    an inconsistent ``output_inputs`` or ``required_channels``, an unknown
    ``trial_memory``): that is a defect, not an N/A pair. Both declarations are
    validated before they are compared. Whether one part of the subject answers
    all requested outputs together is decided by the subject when a session
    starts; a refusal there raises :class:`ExecutionNotFeasibleError`.
    Every declared channel must be registered in the catalog with the right
    direction.
    """
    if Subject not in type(subject).__mro__:     # real inheritance, not Subject.register
        raise TypeError(
            f'{type(subject).__name__} is not a Subject. Every subject is an instance of '
            'brainscore_core.contract.Subject, whose base class runs the session duties around '
            'participate; subclass Subject (an old vision or language model: load it with '
            'load_model, which wraps it in its adapter).')
    subject.participate     # the read refuses a participate that cannot be called
    subject_in = set(subject.in_channels)
    subject_out = set(subject.out_channels)
    subject_required = set(subject.required_channels)
    # The control signals belong to the session, not to what a benchmark presents.
    bench_inputs = benchmark_input_channels(benchmark) - {CONTROL_CHANNEL}
    bench_requested = benchmark_requested_output_channels(benchmark)
    subject_id = subject.identifier
    benchmark_id = _benchmark_identifier(benchmark)

    # Undeclared legacy benchmarks are allowed. A side spelled the way a subject
    # or protocol spells it would otherwise go unchecked, so name the right field.
    for resolved, field_name, near_names in (
            (bench_inputs, 'required_input_channels', ('input_channels', 'in_channels')),
            (bench_requested, 'requested_output_channels', ('output_channels', 'out_channels'))):
        if resolved:
            continue
        misspelled = [name for name in near_names
                      if _looks_like_channels(getattr(benchmark, name, None))]
        if misspelled:
            raise ValueError(
                f"Benchmark '{benchmark_id}' sets {', '.join(misspelled)}, which "
                f"benchmarks do not read, so it was not checked. Use {field_name}."
            )

    if not bench_inputs:
        raise ValueError(
            f"Benchmark '{benchmark_id}' declares no input channels. A benchmark "
            f"states what it sends: set required_input_channels, or keep the "
            f"stimulus_set and task it presents as attributes (rule 5.1)."
            + _no_stimulus_column_hint(getattr(benchmark, "stimulus_set", None))
        )

    # Both declarations are validated completely before any comparison, so a
    # defect (ValueError) is never hidden behind an unrelated mismatch.
    for channels, direction, owner in (
            (subject_in, io_catalog.INPUT, f"subject '{subject_id}' in_channels"),
            (subject_out, io_catalog.OUTPUT, f"subject '{subject_id}' out_channels"),
            (subject_required, io_catalog.INPUT, f"subject '{subject_id}' required_channels"),
            (bench_inputs, io_catalog.INPUT, f"benchmark '{benchmark_id}' input channels"),
            (bench_requested, io_catalog.OUTPUT,
             f"benchmark '{benchmark_id}' requested_output_channels")):
        validate_channels(channels, direction, owner)
    undeclared_required = subject_required - subject_in
    if undeclared_required:
        raise ValueError(
            f"Subject '{subject_id}' requires {_format_channels(undeclared_required)}, which "
            f"are not in its in_channels {_format_channels(subject_in)}.")
    output_inputs = subject.output_inputs
    _validate_output_inputs(output_inputs, subject_in, subject_out, subject_id)
    if 'memory:trial_history' in getattr(subject, 'traits', ()) and CONTROL_CHANNEL not in subject_in:
        raise ValueError(
            f"Subject '{subject_id}' declares memory:trial_history but not the "
            f"'{CONTROL_CHANNEL}' input, so it cannot know where a trial starts. "
            f"Add '{CONTROL_CHANNEL}' to in_channels.")
    required_traits = required_traits_for(getattr(benchmark, 'required_traits', None),
                                           getattr(benchmark, 'trial_memory', 'none'))

    missing_outputs = bench_requested - subject_out
    if missing_outputs:
        message = (
            f"Subject '{subject_id}' cannot produce output channels requested by "
            f"benchmark '{benchmark_id}': {_format_channels(missing_outputs)}. "
            f"Subject out_channels: {_format_channels(subject_out)}."
        )
        regions = sorted(address for family, address in map(parse_channel, missing_outputs)
                         if family == 'neural' and address is not None)
        if regions:
            message += (
                f" It has no layer mapping for region '{regions[0]}'. Name one of the "
                f"subject's own layers for this region in its region_layer_map."
            )
        raise CompatibilityError(message, channels=missing_outputs)

    consumed = {channel for channel in bench_inputs if channel_declared(channel, subject_in)}
    if not consumed:
        raise CompatibilityError(
            f"Subject '{subject_id}' takes in none of the inputs benchmark "
            f"'{benchmark_id}' sends: {_format_channels(bench_inputs)}. "
            f"Subject in_channels: {_format_channels(subject_in)}.",
            channels=bench_inputs)
    if not consumed - {_INSTRUCTION}:
        # Rule 5.2: an instruction alone does not make a pair applicable; it says what
        # to do, it is not a stimulus.
        stimuli = bench_inputs - {_INSTRUCTION}
        raise CompatibilityError(
            f"Subject '{subject_id}' takes in only the instruction of benchmark "
            f"'{benchmark_id}', none of its stimulus inputs {_format_channels(stimuli)}; "
            f"an instruction alone does not make a pair applicable. "
            f"Subject in_channels: {_format_channels(subject_in)}.",
            channels=stimuli)

    unmet_requirements = subject_required - bench_inputs
    if unmet_requirements:
        raise CompatibilityError(
            f"Subject '{subject_id}' hard-requires input channels "
            f"{_format_channels(unmet_requirements)} but benchmark "
            f"'{benchmark_id}' sends {_format_channels(bench_inputs)}.",
            channels=unmet_requirements)

    for output in sorted(bench_requested):
        needed = _answering_inputs(output, output_inputs, subject_in)
        if needed is None:
            continue
        if not needed & consumed:
            raise CompatibilityError(
                f"Subject '{subject_id}' answers {output!r} only from the inputs "
                f"{_format_channels(needed) if needed else '(none it takes in)'}, but it takes in only "
                f"{_format_channels(consumed)} of what benchmark '{benchmark_id}' sends.",
                channels=needed or {output})

    missing_traits = required_traits - set(subject.traits) if required_traits else set()
    if missing_traits:
        raise CompatibilityError(
            f"Benchmark '{benchmark_id}' requires subject traits {sorted(missing_traits)}; "
            f"subject '{subject_id}' declares {sorted(subject.traits) or 'none'}."
        )

    return CompatibilityMatch(
        subject=subject_id,
        benchmark=benchmark_id,
        consumed_inputs=frozenset(consumed),
        dropped_inputs=frozenset(bench_inputs - consumed),
        requested_output_channels=frozenset(bench_requested),
    )


def _no_stimulus_column_hint(stimuli) -> str:
    """For a kept table with no catalog column: its columns and the accepted ones."""
    if stimuli is None:
        return ""
    accepted: Dict[str, list] = {}
    for column, channel in io_catalog.stimulus_columns().items():
        accepted.setdefault(channel, []).append(column)
    names = "; ".join(f"{channel}: {', '.join(columns)}" for channel, columns in sorted(accepted.items()))
    return (f" Its stimulus_set has the columns {sorted(map(str, stimuli.columns))}, none of which "
            f"feeds an input. Name the stimulus column by one of the accepted names ({names}).")


def _answering_inputs(output, output_inputs, subject_in):
    """The inputs ``output`` can be answered from, or None when any input will do (rule 4)."""
    if output in output_inputs:
        return set(output_inputs[output])
    if output == 'perturbation':
        return {channel for channel in subject_in if _is_intervention(channel)}
    return None


def _is_intervention(channel) -> bool:
    """An input channel whose payload is a ``StateChange`` (``lesion``, ``stimulation``, ...)."""
    from brainscore_core.events import StateChange
    if not io_catalog.has(channel):
        return False
    entry = io_catalog.get(channel)
    return entry.direction == io_catalog.INPUT and entry.payload_type is StateChange


def _validate_output_inputs(output_inputs, in_channels, out_channels, subject_id) -> None:
    """Every listed output is declared, and every input listed for it is declared."""
    for output, inputs in output_inputs.items():
        if output not in out_channels:
            raise ValueError(
                f"Subject '{subject_id}' lists {output!r} in output_inputs, but not in "
                f"out_channels {_format_channels(set(out_channels))}.")
        if not inputs:
            raise ValueError(
                f"Subject '{subject_id}' lists {output!r} in output_inputs with no input. "
                "A listed output must name at least one input.")
        undeclared = set(inputs) - set(in_channels)
        if undeclared:
            raise ValueError(
                f"Subject '{subject_id}' lists the inputs {_format_channels(undeclared)} for "
                f"{output!r} in output_inputs, but not in in_channels "
                f"{_format_channels(set(in_channels))}.")
        if output == 'perturbation':
            other = {channel for channel in inputs if not _is_intervention(channel)}
            if other:
                raise ValueError(
                    f"Subject '{subject_id}' lists 'perturbation' in output_inputs from "
                    f"{_format_channels(other)}. A perturbation record answers an intervention: "
                    "list only intervention inputs (lesion, stimulation, pharmacological).")


def benchmark_input_channels(benchmark) -> Set[str]:
    """Every input channel the benchmark sends."""
    return benchmark_required_input_channels(benchmark) | benchmark_accepted_input_channels(benchmark)


def benchmark_required_input_channels(benchmark) -> Set[str]:
    """Return the input channels a benchmark sends.

    Its ``required_input_channels`` when declared; else the v1.5
    ``required_modalities``; else what its ``stimulus_set`` presents under its
    ``task`` (``task_input_channels``, with ``instruction`` when the task gives one).
    """
    declared = getattr(benchmark, "required_input_channels", None)
    if declared is not None:
        return set(declared)
    modalities = set(getattr(benchmark, "required_modalities", set()))
    if modalities:
        return modalities_to_input_channels(modalities)
    stimuli = getattr(benchmark, "stimulus_set", None)
    if stimuli is None:
        return set()
    from brainscore_core._sessions.view import task_input_channels
    return set(task_input_channels(stimuli, getattr(benchmark, "task", None)))


def benchmark_accepted_input_channels(benchmark) -> Set[str]:
    """Return v1.5 any-of input channels from accepted_modalities."""
    if getattr(benchmark, "required_input_channels", None) is not None:
        return set()
    modalities = set(getattr(benchmark, "accepted_modalities", set()))
    return modalities_to_input_channels(modalities)


def benchmark_requested_output_channels(benchmark) -> Set[str]:
    """Return v2 requested output channels, with a v1.5 region fallback."""
    declared = getattr(benchmark, "requested_output_channels", None)
    if declared is not None:
        return set(declared)
    region = getattr(benchmark, "region", None)
    if region is None:
        return set()
    return {f"neural:{region}"}


def validate_channels(channels: Set[str], direction: str, owner: str, *,
                      allow_unregistered: bool = False) -> None:
    """Validate channel names and direction against the shared catalog.

    Exploratory experiments may allow unregistered families with a warning.
    Benchmark scoring requires registration. An unknown family, invalid
    syntax, direction, form or address is a malformed declaration and raises
    ValueError.
    """
    for channel in sorted(channels, key=repr):
        try:
            family, address = parse_channel(channel)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{owner} declares invalid channel '{channel}': {error}."
            ) from error

        try:
            entry = io_catalog.get(family)
        except KeyError as error:
            if allow_unregistered:
                warnings.warn(
                    f"{owner} declares unregistered channel '{channel}'. "
                    'This experiment may run, but benchmark scoring requires '
                    'a catalog channel or register_channel() registration.',
                    CompatibilityWarning, stacklevel=2)
                continue
            known = sorted(entry.name for entry in io_catalog.all_entries())
            raise ValueError(
                f"{owner} declares unknown channel '{channel}': family '{family}' is not in "
                f"the channel catalog; known families: {known}. Correct the name, or "
                "register the family with register_channel()."
            ) from error

        if entry.direction not in (direction, io_catalog.BOTH):
            raise ValueError(
                f"{owner} declares channel '{channel}' as {direction}, but "
                f"the registry marks family '{family}' as {entry.direction}."
            )

        problem = io_catalog.form_problem(channel)
        if problem:
            raise ValueError(f"{owner} declares {problem}.")

        if address is not None and entry.addressing is None:
            raise ValueError(
                f"{owner} declares addressed channel '{channel}', but family "
                f"'{family}' is not addressable."
            )


def _format_channels(channels: Set[str]) -> str:
    return (
        "{"
        + ", ".join(repr(channel) for channel in sorted(channels, key=repr))
        + "}"
    )


def _benchmark_identifier(benchmark) -> str:
    return getattr(benchmark, "identifier", "?")


def check_compatibility(model: Subject, benchmark) -> CompatibilityMatch:
    """Fast pre-flight check; returns the match or raises CompatibilityError.

    One rule for every subject (see the module docstring). Emits a
    :class:`DeprecationWarning` if the benchmark still sets the deprecated
    ``available_modalities`` field, before any compatibility error.
    """
    bench_required: Set[str] = {canonical_modality(m)
                                for m in getattr(benchmark, 'required_modalities', set())}
    bench_available_attr = getattr(benchmark, 'available_modalities', None)
    if bench_available_attr is not None:
        bench_available: Set[str] = {canonical_modality(m) for m in bench_available_attr}
        # Only warn if it adds something beyond required (otherwise it is a
        # redundant declaration, not a multi-input claim).
        if bench_available - bench_required:
            warnings.warn(
                f"Benchmark '{_benchmark_identifier(benchmark)}' sets "
                f"`available_modalities={bench_available}`, which is deprecated. "
                f"Declare the benchmark's inputs in `required_input_channels` (or keep "
                f"its stimuli as `stimulus_set`), and register each other input form "
                f"as a separate benchmark variant.",
                DeprecationWarning,
                stacklevel=2,
            )

    match = check_channel_compatibility(model, benchmark)
    return match
