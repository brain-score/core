"""The channel catalog: the one list of channel families a session accepts.

A channel is a name for one kind of input or output of a subject. The name has
a family and an optional address (``family:address``), for example
``'neural:IT'`` or ``'vision:image'``. Every family has one entry here with its
direction, its payload contract and, when it has one, a payload type and a
validator (Contract Rules 3.2 and 5).

The catalog is part of the contract. Pre-flight refuses a channel that is not
registered. A session checks every event against the entry of its family and
raises an error for a payload that does not match. A family with a ``CHANNEL_FORMS``
entry (``vision``) has no bare channel: each channel names a form.

The catalog also holds the one table that says which channel a stimulus-table
column feeds (``stimulus_columns``, ``column_channel``). A table session sends
each such column on its channel; pre-flight, the subject's projection of a table
and ``BrainScoreModel`` read the same table.

``check_payload`` is a loose check on ``.ndim`` and ``.dtype``, so that core needs
no heavy dependency. ``validate`` runs every check for one event.

``register_channel`` adds a family and, for an input family, the columns that
feed it. Code outside core imports it from ``brainscore_core.extensions``.
"""
from dataclasses import dataclass
from collections.abc import Iterable
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .streaming import parse_channel

INPUT = "input"
OUTPUT = "output"
BOTH = "both"
# ``video`` is a modality name that folds into ``vision`` for lookups
# (``canonical_modality``). A visual input channel names its form instead:
# ``vision:image`` or ``vision:video`` (see ``MODALITY_TO_INPUT_CHANNEL``).
MODALITY_ALIASES = {
    "video": "vision",
}


def canonical_modality(modality: str) -> str:
    """Fold modality aliases to their canonical name (e.g. ``video`` -> ``vision``)."""
    return MODALITY_ALIASES.get(modality, modality)


# A visual input names its form: an image or a video (Contract Rules 3.7, 5.9).
MODALITY_TO_INPUT_CHANNEL = {
    "vision": "vision:image",
    "text": "text",
    "audio": "audio",
    "video": "vision:video",
}

_DIRECTIONS = (INPUT, OUTPUT, BOTH)
ShapeValidator = Callable[[Any], object]


@dataclass(frozen=True)
class CatalogEntry:
    """One channel family and its payload contract.

    ``name``, ``direction``, ``payload_contract``, ``meta_keys``, ``addressing``,
    ``expected_ndim``, ``expected_dtype``, ``shape_validator`` and ``payload_type``
    are read by pre-flight and by sessions.
    ``carried_by``, ``handled_by``, ``owner`` and ``materializers`` are notes for
    readers; no code reads them.
    """
    name: str                          # channel family, e.g. 'text', 'neural'
    direction: str                     # INPUT, OUTPUT, or BOTH
    carried_by: str                    # note: what carries it (a table column, a task, a policy)
    payload_contract: str              # human-readable dtype / shape / units
    handled_by: str                    # note: which harness or wrapper interprets the payload
    meta_keys: Tuple[str, ...] = ()     # recognized metadata keys
    addressing: Optional[str] = None    # how the ':address' suffix is interpreted, if any
    expected_ndim: Tuple[int, ...] = ()  # allowed array ndims; empty means unchecked
    expected_dtype: Optional[str] = None  # e.g. 'uint8'; None means unchecked
    shape_validator: Optional[ShapeValidator] = None
    owner: str = "core"                # note: who defines the family
    materializers: Tuple[str, ...] = ()  # note: the objects that produce it
    payload_type: Any = None            # type or tuple of types a payload must be; None means unchecked

    def __post_init__(self):
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"CatalogEntry.direction must be one of {_DIRECTIONS}")
        object.__setattr__(self, "meta_keys", tuple(self.meta_keys))
        object.__setattr__(self, "expected_ndim", tuple(self.expected_ndim))
        object.__setattr__(self, "materializers", tuple(self.materializers))


_CATALOG: Dict[str, CatalogEntry] = {}

# Which channel each stimulus-table column feeds: the one answer for sessions,
# pre-flight, subjects and documents. ``register_channel`` adds columns. Within
# one channel the order is the order a prompt looks for a part's column.
_STIMULUS_COLUMNS: Dict[str, str] = {
    'image_file_name': 'vision:image', 'image_path': 'vision:image',
    'filename': 'vision:image', 'image': 'vision:image', 'vision': 'vision:image',
    'video_path': 'vision:video', 'video_file_name': 'vision:video', 'video': 'vision:video',
    'sentence': 'text', 'text': 'text',
    'audio_path': 'audio', 'audio_file_name': 'audio', 'audio_file': 'audio', 'audio': 'audio',
}


def stimulus_columns() -> Dict[str, str]:
    """Every stimulus column and the channel it feeds, as a copy."""
    return dict(_STIMULUS_COLUMNS)


def column_channel(column: str) -> Optional[str]:
    """The channel a stimulus-table column feeds, or None for a metadata column."""
    return _STIMULUS_COLUMNS.get(column)


def register_channel(entry: CatalogEntry, *, columns: Iterable[str] = (),
                     replace: bool = False) -> CatalogEntry:
    """Add a channel family to the catalog and bind stimulus columns to it.

    Each column in ``columns`` then feeds the channel ``entry.name``: a table
    session sends it on that channel, pre-flight and the subject's projection
    of a table see it. Only an input family takes columns, and a family that
    names its form (``vision``) does not: give a new family.

    Registering a family or a column that already exists is an error unless
    ``replace=True`` (Contract Rules 3.5). Re-registering the same entry
    object is a no-op.
    """
    _, address = parse_channel(entry.name)
    if address is not None:
        raise ValueError("Register a channel family, not an addressed instance")
    if isinstance(columns, str):
        raise TypeError(f"columns is a sequence of column names, not the string {columns!r}")
    columns = tuple(columns)
    existing = _CATALOG.get(entry.name)
    if existing is not None and existing is not entry and not replace:
        raise ValueError(
            f"Channel family {entry.name!r} is already registered "
            f"(owner {existing.owner!r}); pass replace=True to replace it.")
    if columns:
        if entry.direction not in (INPUT, BOTH):
            raise ValueError('Only input channels can own stimulus columns')
        problem = form_problem(entry.name)
        if problem:
            raise ValueError(f"Columns cannot feed {problem}")
    for column in columns:
        if not isinstance(column, str) or not column:
            raise ValueError('Stimulus columns must be nonempty strings')
        owner = _STIMULUS_COLUMNS.get(column)
        if owner is not None and owner != entry.name and not replace:
            raise ValueError(f'Column {column!r} already belongs to {owner!r}')
    _CATALOG[entry.name] = entry
    _STIMULUS_COLUMNS.update({column: entry.name for column in columns})
    return entry


def _family(label: str) -> str:
    return label.split(":", 1)[0]


def _check_direction(direction: str) -> None:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")


def _matches_direction(entry: CatalogEntry, direction: str) -> bool:
    return entry.direction in (direction, BOTH)


# A family listed here has no bare channel: each channel names one of these forms.
CHANNEL_FORMS: Mapping[str, tuple] = MappingProxyType({"vision": ("image", "video")})


def form_problem(channel: str) -> Optional[str]:
    """Why ``channel`` does not name a form its family needs, or None."""
    family, address = parse_channel(channel)
    forms = CHANNEL_FORMS.get(family)
    if forms is None or address in forms:
        return None
    names = " or ".join(f"'{family}:{form}'" for form in forms)
    return f"{channel}: a '{family}' channel names its form; use {names}"


def modalities_to_input_channels(modalities) -> set[str]:
    """Map v1.5 modality names onto v2 input channel families."""
    return {
        MODALITY_TO_INPUT_CHANNEL.get(modality, modality)
        for modality in set(modalities)
    }


def get(label: str) -> CatalogEntry:
    """Look up the catalog entry for a label.

    A plain label (``'text'``) resolves directly. An addressed label
    (``'neural:IT'``) resolves to its family entry (``'neural'``), which holds
    the payload contract; the address is interpreted by the handler.
    """
    if label in _CATALOG:
        return _CATALOG[label]
    family = _family(label)
    if family in _CATALOG:
        return _CATALOG[family]
    raise KeyError(
        f"No Input/Output Catalog entry for '{label}' "
        f"(family '{family}'). Known families: {sorted(_CATALOG)}"
    )


def has(label: str) -> bool:
    """Whether a label resolves to a catalog entry."""
    return label in _CATALOG or _family(label) in _CATALOG


def all_entries() -> List[CatalogEntry]:
    """Every catalog entry, sorted by name."""
    return [_CATALOG[k] for k in sorted(_CATALOG)]


def check_payload(label: str, payload) -> List[str]:
    """Best-effort, documentation-backed shape check.

    Returns a list of human-readable warnings; an empty list means no obvious
    mismatch was found. This is loose by design: it only checks ``.ndim`` /
    ``.dtype`` when the entry declares an expectation and the payload exposes
    them. It never raises on a normal mismatch and never replaces the type
    system; it exists so a benchmark can surface an obvious wrong-shape payload
    at session start instead of deep inside extraction. The ndim and dtype
    hints describe raw arrays: a table (``StimulusSet``) or an assembly has its
    own shape and is not held to them.
    """
    e = get(label)
    warnings: List[str] = []
    if e.expected_ndim and isinstance(payload, np.ndarray):
        if payload.ndim not in e.expected_ndim:
            warnings.append(
                f"{label}: payload ndim {payload.ndim} not in expected "
                f"{e.expected_ndim} ({e.payload_contract})"
            )
    if e.expected_dtype and isinstance(payload, np.ndarray):
        if str(payload.dtype) != e.expected_dtype:
            warnings.append(
                f"{label}: payload dtype {payload.dtype} is not the documented "
                f"{e.expected_dtype} ({e.payload_contract})"
            )
    return warnings


def _normalise_shape_validator_result(label: str, result: object) -> List[str]:
    if result is None or result is True:
        return []
    if result is False:
        return [f"{label}: payload failed shape_validator"]
    if isinstance(result, str):
        return [f"{label}: {result}"]
    if isinstance(result, Iterable):
        return [f"{label}: {warning}" for warning in result]
    return [f"{label}: shape_validator returned unsupported result {result!r}"]


def validate(
    channel: str,
    payload,
    direction: Optional[str] = None,
) -> List[str]:
    """Validate a channel payload against the runtime-queryable registry.

    Returns human-readable validation failures. An empty list means the channel
    is known, its address is syntactically valid, the optional direction matches,
    and no documented payload-shape mismatch was found.
    """
    warnings: List[str] = []
    try:
        family, address = parse_channel(channel)
    except (TypeError, ValueError) as error:
        return [f"{channel!r}: invalid channel name ({error})"]

    try:
        entry = get(family)
    except KeyError as error:
        return [str(error)]

    if address is not None and entry.addressing is None:
        warnings.append(f"{channel}: channel family '{family}' is not addressable")
    problem = form_problem(channel)
    if problem:
        warnings.append(problem)

    if direction is not None:
        _check_direction(direction)
        if not _matches_direction(entry, direction):
            warnings.append(
                f"{channel}: registry direction is {entry.direction!r}, "
                f"not requested {direction!r}"
            )

    warnings.extend(check_payload(channel, payload))
    warnings.extend(payload_type_notes(channel, payload))

    if entry.shape_validator is not None:
        warnings.extend(
            _normalise_shape_validator_result(
                channel, entry.shape_validator(payload)
            )
        )

    return warnings


# ----------------------------------------------------------------------------
# Seed catalog: the allowable inputs and outputs the initial release documents.
# Device-specific payloads (an environment observation, a motor action) are
# harness-defined; the catalog records that they are harness-defined rather than
# pinning a device schema in core.
# ----------------------------------------------------------------------------

# A policy that answers in words keeps, in the extra data of its motor event, what
# it said (answer), whether it was understood (valid, parse_error) and where it
# came from (provenance). fallback_action is the action applied when the answer
# could not be parsed.
RESPONSE_META_KEYS = ("answer", "valid", "parse_error", "fallback_action", "provenance")

# A stimulus table sends one event per input column of each row. Every such
# event names its row and column (stimulus_id, stimulus_index, column). When a
# row has several input columns, its events also carry table_row: the integer
# position of the row in the session's stimulus_set. Consecutive events with a
# table_row that matches the table (stimulus_index equals the index label at
# that position, a null label matching a null label; stimulus_id equals the
# row's id; column is an input column of the table whose channel is the
# event's channel) are one presentation, and the row is shown once (rule 4.5).
# Only consecutive events group: interleaved rows are separate presentations.
# A repeated column or a different t_ms starts a new presentation. A row whose
# remaining columns never arrive is presented when the input moves on or ends.
# A complete, matching table_row is honored on any event, including an
# external one; an event without it, or with one that does not match, is
# presented alone. Rows of a one-column table carry no table_row.
TABLE_ROW_META_KEYS = ("stimulus_id", "stimulus_index", "column", "table_row")

_SEED = [
    # --- inputs: sensory (carried by StimulusSet columns) ---
    # `vision` is a channel family with no bare channel: `vision:image` takes
    # (H, W, 3) payloads and `vision:video` takes (T, H, W, 3) payloads.
    CatalogEntry("vision", INPUT, "StimulusSet column",
                 "(H, W, 3) uint8 image or (T, H, W, 3) uint8 clip, a table of frames, or a path to one",
                 "image/video preprocessor + a vision model harness (wrapper)",
                 meta_keys=TABLE_ROW_META_KEYS, expected_ndim=(3, 4), expected_dtype="uint8",
                 addressing=("address is the form a subject takes natively: 'image' "
                             "(images) or 'video' (video); a subject declares each form it "
                             "has, and a benchmark adapts a stimulus only when a form is missing"),
                 materializers=("StimulusSet",)),
    CatalogEntry("text", INPUT, "StimulusSet column",
                 "a Python str",
                 "tokenizer + a text model harness (TextWrapper)",
                 meta_keys=TABLE_ROW_META_KEYS, materializers=("StimulusSet",)),
    CatalogEntry("audio", INPUT, "StimulusSet column",
                 "(samples,) float32 waveform, or a path; sample rate read from the file header",
                 "an audio model harness (AudioWrapper)",
                 meta_keys=TABLE_ROW_META_KEYS + ("sample_rate_hz",), expected_ndim=(1,),
                 materializers=("StimulusSet",)),
    CatalogEntry("instruction", INPUT, "TaskContext.instruction or a closed-loop row entry",
                 "a Python str framing the task",
                 "the model's generation or policy path",
                 materializers=("TaskContext", "EnvironmentSource")),
    # Episode control, not a sensory stream: a source marks where an episode
    # begins, where the subject is asked to respond, and where it ends (the
    # end event says why in meta['reason']). Every subject receives it.
    CatalogEntry("control", INPUT, "StreamEvent",
                 "one of 'begin', 'respond', 'end'; 'end' carries meta['reason'] "
                 "in ('terminal', 'truncated', 'failure')",
                 "the session; a subject never has to declare it",
                 meta_keys=("reason",), payload_type=str,
                 shape_validator=lambda payload: (
                     [] if payload in ('begin', 'respond', 'end')
                     else [f"control payload must be 'begin', 'respond' or 'end'; got {payload!r}"])),
    # --- inputs: direct neural (carried by StateChange) ---
    # Not addressable: the target is in the StateChange payload, because a
    # benchmark can find it while it runs.
    CatalogEntry("stimulation", INPUT, "StateChange",
                 "a perturbation spec (kind, scale, replacement) applied to selected units",
                 "the model's state_change_fn",
                 materializers=("StateChange",)),
    CatalogEntry("lesion", INPUT, "StateChange",
                 "an ablation spec applied to selected units",
                 "the model's state_change_fn",
                 materializers=("StateChange",)),
    CatalogEntry("pharmacological", INPUT, "StateChange",
                 "a pharmacological perturbation spec applied to an agent or selected units",
                 "the model's state_change_fn",
                 owner="core",
                 materializers=("StateChange",)),
    # --- inputs: closed-loop (carried by a row of an environment, harness-defined payload) ---
    # Retained legacy v1.5 channel; not a canonical v2 seed family.
    CatalogEntry("observation", INPUT, "a closed-loop row entry",
                 "harness-defined; the environment harness documents its own structure",
                 "the environment harness (e.g. robotics, Atari, browser)",
                 owner="environment harness",
                 materializers=("EnvironmentSource",)),
    CatalogEntry("proprioception", INPUT, "a closed-loop row entry (robot body state)",
                 "harness-defined robot state (joint position, end-effector pose, gripper)",
                 "the robotics environment harness",
                 owner="environment harness",
                 materializers=("EnvironmentSource",)),
    CatalogEntry("reward", INPUT, "a closed-loop row entry",
                 "a Python float, the environment's reward for the last action",
                 "the environment harness",
                 meta_keys=("discount",), payload_type=(int, float),
                 owner="environment harness",
                 materializers=("EnvironmentSource",)),
    # --- outputs: neural measurement (a subject answer on neural:<region>) ---
    CatalogEntry("neural", OUTPUT, "a subject answer on neural:<region>",
                 "(units,) or (units, time) activations packaged into a NeuroidAssembly",
                 "the wrapper at region_layer_map[region]",
                 meta_keys=("signal_type", "layer"), expected_ndim=(1, 2),
                 addressing="address is a brain region present in region_layer_map",
                 materializers=("NeuroidAssembly",)),
    # --- outputs: behavior (a subject answer on behavior:<kind>) ---
    CatalogEntry("behavior", OUTPUT, "a subject answer on behavior:<kind>, as the task asks",
                 "a label, a probability vector, or generated text",
                 "a fitted readout or the model's generation path",
                 meta_keys=("label_set",),
                 addressing=("address is the kind of answer a TaskContext response asks for: "
                             "choice, probabilities, confidence, rating, rt or text"),
                 materializers=("BehavioralAssembly",)),
    # --- outputs: motor (answered by a policy, harness-defined) ---
    CatalogEntry("motor", OUTPUT, "a policy's returned mapping entry",
                 "harness-defined continuous action / control vector",
                 "the model's policy, decoded by the environment harness",
                 meta_keys=RESPONSE_META_KEYS,
                 owner="environment harness",
                 materializers=("policy",)),
    # --- outputs: perturbation record (carried by PerturbationApplied) ---
    CatalogEntry("perturbation", OUTPUT, "PerturbationApplied",
                 "PerturbationApplied record with a handle_id for undoing one change",
                 "the model's intervention (state_change_fn)",
                 owner="core",
                 materializers=("PerturbationApplied",)),
    # --- outputs: physiological readouts (reserved for human harnesses) ---
    CatalogEntry("pupil", OUTPUT, "human harness recording",
                 "(t,) float pupil diameter time series, or scalar diameter",
                 "the human harness (eye tracker)",
                 meta_keys=("signal_type",),
                 owner="human harness"),
    CatalogEntry("skin_conductance", OUTPUT, "human harness recording",
                 "(t,) float skin conductance time series, or scalar value",
                 "the human harness (physiological sensor)",
                 meta_keys=("signal_type",),
                 owner="human harness"),
]

for _entry in _SEED:
    register_channel(_entry)


def payload_type_notes(label: str, payload) -> List[str]:
    """Report a payload whose type does not match its family's ``payload_type``.

    :func:`validate` includes this check, so a session treats a mismatch as an error.
    """
    entry = get(label)
    if entry.payload_type is None or isinstance(payload, entry.payload_type):
        return []
    expected = entry.payload_type
    names = (expected.__name__ if isinstance(expected, type)
             else ', '.join(t.__name__ for t in expected))
    return [f"{label}: payload is {type(payload).__name__}, not {names} "
            f"({entry.payload_contract})"]


def _declare_payload_types() -> None:
    """Attach the payload type of each seeded family.

    Done after seeding, with local imports, so the catalog module itself keeps
    no import-time dependency on the event and assembly modules.
    """
    from dataclasses import replace as _replace
    from pathlib import Path
    from .events import PerturbationApplied, StateChange
    from .supported_data_standards.brainio.stimuli import StimulusSet
    from .supported_data_standards.brainio.assemblies import DataAssembly

    sensory = (str, Path, np.ndarray, StimulusSet)  # a StimulusSet is a window of frames
    payload_types = {
        'vision': sensory, 'audio': sensory,
        'text': str, 'instruction': str,
        'lesion': StateChange, 'stimulation': StateChange, 'pharmacological': StateChange,
        'neural': (DataAssembly, np.ndarray),
        'behavior': (DataAssembly, str, np.ndarray),
        'perturbation': (PerturbationApplied, type(None)),
    }
    for family, payload_type in payload_types.items():
        register_channel(_replace(get(family), payload_type=payload_type), replace=True)


_declare_payload_types()
