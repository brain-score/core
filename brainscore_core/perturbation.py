"""State-change and perturbation lifecycle support for BrainScoreModel."""

from dataclasses import replace
from typing import Callable, Dict

from .events import PerturbationApplied
from .selection import UnitSelection


class PerturbationManager:
    """Owns active perturbation state for a BrainScoreModel instance."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self.active_perturbations: Dict[str, Callable[[], None]] = {}

    def reset(self) -> None:
        for handle_id, cleanup in list(self.active_perturbations.items()):
            del handle_id
            try:
                cleanup()
            except Exception:
                pass
        self.active_perturbations.clear()

    def resolve_selection(self, state_change):
        """Resolve a ``UnitSelection`` target to a concrete ``Selection``."""
        target = state_change.target
        if not isinstance(target, UnitSelection):
            return state_change
        owner = self.owner
        saved_regions = list(getattr(owner, '_recording_regions', []) or [])
        saved_time_bins = owner._time_bins
        try:
            resolved = target.resolve(owner)
        finally:
            if saved_regions:
                tgt = saved_regions if len(saved_regions) > 1 else saved_regions[0]
                owner.start_recording(tgt, time_bins=saved_time_bins)
        return replace(state_change, target=resolved)

    def dispatch_state_change(self, state_change):
        """Route a StateChange event to the registered ``state_change_fn``."""
        owner = self.owner
        if state_change.kind == 'reset':
            handle_id = state_change.handle_id
            if handle_id is None:
                raise ValueError(
                    f"StateChange(kind='reset') requires handle_id to identify "
                    f"which perturbation to undo. Use model.reset() to clear all."
                )
            cleanup = self.active_perturbations.pop(handle_id, None)
            if cleanup is None:
                raise KeyError(
                    f"No active perturbation with handle_id={handle_id!r}. "
                    f"Active: {list(self.active_perturbations.keys())}."
                )
            cleanup()
            return None

        if owner._state_change_fn is None:
            raise NotImplementedError(
                f"Model '{owner.identifier}' has no state_change_fn registered. "
                f"Perturbation evaluation requires the model to declare a "
                f"state_change_fn(state_change) -> (PerturbationApplied, "
                f"cleanup) callable at BrainScoreModel construction time. "
                f"Received: StateChange(kind={state_change.kind!r})."
            )

        state_change = self.resolve_selection(state_change)
        result = owner._state_change_fn(state_change)
        if (not isinstance(result, tuple) or len(result) != 2
                or not isinstance(result[0], PerturbationApplied)
                or not callable(result[1])):
            raise TypeError(
                f"Model '{owner.identifier}' state_change_fn must return a "
                f"(PerturbationApplied, cleanup_callable) tuple. Got "
                f"{type(result).__name__}."
            )
        applied, cleanup = result
        if applied.handle_id in self.active_perturbations:
            raise ValueError(
                f"Duplicate handle_id {applied.handle_id!r} from state_change_fn. "
                f"state_change_fn must produce unique handle_ids per call."
            )
        self.active_perturbations[applied.handle_id] = cleanup
        return applied
