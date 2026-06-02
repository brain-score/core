"""The Input/Output Catalog: documented, queryable contracts for the allowable
inputs and outputs of a Brain-Score subject.

v1.5 keeps the typed events (``StimulusSet``, ``StateChange``,
``EnvironmentStep``) and the lightweight methods (``process``,
``start_recording``, ``start_task``). The catalog is documentation layered over
them, not a dispatch mechanism. It records, for every allowable input and
output: what carries it, what its payload contains, which harness interprets it,
and what metadata it recognizes.

The channel-style names (e.g. ``'neural:IT'``, ``'motor'``) are labels for
readability and alignment with how the field talks. They are NOT a dispatch
contract: dispatch is still by typed event and measurement method. A name with a
colon has a family and an address (``family:address``); the family carries the
payload contract and the address is interpreted by the relevant harness.

The catalog also offers an optional, best-effort ``check_payload`` for a
documentation-backed pre-flight check. It is deliberately loose (duck-typed on
``.ndim`` / ``.dtype``) so ``core`` stays free of heavy dependencies; it never
replaces the type system, it only flags obvious payload mismatches early.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

INPUT = "input"
OUTPUT = "output"


@dataclass(frozen=True)
class CatalogEntry:
    """One allowable input or output and its payload contract."""
    name: str                          # channel-style label, e.g. 'vision', 'neural:IT'
    kind: str                          # INPUT or OUTPUT
    carried_by: str                    # the typed envelope or method that carries it
    payload_contract: str              # human-readable dtype / shape / units
    handled_by: str                    # which harness or wrapper interprets the payload
    meta_keys: Tuple[str, ...] = ()     # recognized metadata keys
    addressing: Optional[str] = None    # how the ':address' suffix is interpreted, if any
    expected_ndim: Tuple[int, ...] = ()  # allowed array ndims; empty means unchecked
    expected_dtype: Optional[str] = None  # e.g. 'uint8'; None means unchecked


_CATALOG: Dict[str, CatalogEntry] = {}


def register(entry: CatalogEntry) -> None:
    """Add or replace a catalog entry, keyed by its family name."""
    _CATALOG[entry.name] = entry


def _family(label: str) -> str:
    return label.split(":", 1)[0]


def get(label: str) -> CatalogEntry:
    """Look up the catalog entry for a label.

    A plain label (``'vision'``) resolves directly. An addressed label
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


def inputs() -> List[CatalogEntry]:
    """Every input entry."""
    return [e for e in all_entries() if e.kind == INPUT]


def outputs() -> List[CatalogEntry]:
    """Every output entry."""
    return [e for e in all_entries() if e.kind == OUTPUT]


def describe(label: str) -> str:
    """A human-readable one-block description of a catalog entry."""
    e = get(label)
    lines = [
        f"{e.name} ({e.kind})",
        f"  carried by:  {e.carried_by}",
        f"  payload:     {e.payload_contract}",
        f"  handled by:  {e.handled_by}",
    ]
    if e.addressing:
        lines.append(f"  addressing:  {e.addressing}")
    if e.meta_keys:
        lines.append(f"  meta keys:   {', '.join(e.meta_keys)}")
    return "\n".join(lines)


def check_payload(label: str, payload) -> List[str]:
    """Best-effort, documentation-backed shape check.

    Returns a list of human-readable warnings; an empty list means no obvious
    mismatch was found. This is loose by design: it only checks ``.ndim`` /
    ``.dtype`` when the entry declares an expectation and the payload exposes
    them. It never raises on a normal mismatch and never replaces the type
    system; it exists so a benchmark can surface an obvious wrong-shape payload
    at session start instead of deep inside extraction.
    """
    e = get(label)
    warnings: List[str] = []
    if e.expected_ndim and hasattr(payload, "ndim"):
        if payload.ndim not in e.expected_ndim:
            warnings.append(
                f"{label}: payload ndim {payload.ndim} not in expected "
                f"{e.expected_ndim} ({e.payload_contract})"
            )
    if e.expected_dtype and hasattr(payload, "dtype"):
        if str(payload.dtype) != e.expected_dtype:
            warnings.append(
                f"{label}: payload dtype {payload.dtype} is not the documented "
                f"{e.expected_dtype} ({e.payload_contract})"
            )
    return warnings


# ----------------------------------------------------------------------------
# Seed catalog: the allowable inputs and outputs the initial release documents.
# Device-specific payloads (an environment observation, a motor action) are
# harness-defined; the catalog records that they are harness-defined rather than
# pinning a device schema in core.
# ----------------------------------------------------------------------------

_SEED = [
    # --- inputs: sensory (carried by StimulusSet columns) ---
    CatalogEntry("vision", INPUT, "StimulusSet column",
                 "(H, W, 3) uint8 image, or a path to one",
                 "image preprocessor + a vision model harness (wrapper)",
                 meta_keys=("stimulus_id",), expected_ndim=(3,), expected_dtype="uint8"),
    CatalogEntry("text", INPUT, "StimulusSet column",
                 "a Python str",
                 "tokenizer + a text model harness (TextWrapper)",
                 meta_keys=("stimulus_id",)),
    CatalogEntry("audio", INPUT, "StimulusSet column",
                 "(samples,) float32 waveform, or a path; sample rate read from the file header",
                 "an audio model harness (AudioWrapper)",
                 meta_keys=("stimulus_id", "sample_rate_hz"), expected_ndim=(1,)),
    CatalogEntry("video", INPUT, "StimulusSet column",
                 "(T, H, W, 3) uint8 clip, or a path",
                 "a video model harness (VideoWrapper)",
                 meta_keys=("stimulus_id",), expected_ndim=(4,)),
    CatalogEntry("instruction", INPUT, "TaskContext.instruction or EnvironmentStep.instruction",
                 "a Python str framing the task",
                 "the model's generation or policy path"),
    # --- inputs: direct neural (carried by StateChange) ---
    CatalogEntry("stimulation", INPUT, "StateChange",
                 "a perturbation spec (kind, scale, replacement) applied to selected units",
                 "the model's state_change_fn",
                 addressing="address is a unit selector (layer path, optional indices)"),
    CatalogEntry("lesion", INPUT, "StateChange",
                 "an ablation spec applied to selected units",
                 "the model's state_change_fn",
                 addressing="address is a unit selector (layer path, optional indices)"),
    # --- inputs: embodied (carried by EnvironmentStep, harness-defined payload) ---
    CatalogEntry("observation", INPUT, "EnvironmentStep.observation",
                 "harness-defined; the environment harness documents its own structure",
                 "the environment harness (e.g. robotics, Atari, browser)"),
    CatalogEntry("proprioception", INPUT, "EnvironmentStep (robot body state)",
                 "harness-defined robot state (joint position, end-effector pose, gripper)",
                 "the robotics environment harness"),
    # --- outputs: neural measurement (carried by start_recording + process) ---
    CatalogEntry("neural", OUTPUT, "start_recording(target) + process(stimuli)",
                 "(units,) or (units, time) activations packaged into a NeuroidAssembly",
                 "the wrapper at region_layer_map[region]",
                 meta_keys=("signal_type", "layer"), expected_ndim=(1, 2),
                 addressing="address is a brain region present in region_layer_map"),
    # --- outputs: behavior (carried by start_task + process) ---
    CatalogEntry("behavior", OUTPUT, "start_task(task_context) + process(stimuli)",
                 "a label, a probability vector, or generated text",
                 "a fitted readout or the model's generation path",
                 meta_keys=("label_set",)),
    # --- outputs: motor (carried by EnvironmentResponse, harness-defined) ---
    CatalogEntry("motor", OUTPUT, "EnvironmentResponse.action",
                 "harness-defined continuous action / control vector",
                 "the model's action_fn, decoded by the environment harness"),
]

for _entry in _SEED:
    register(_entry)
