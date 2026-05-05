"""Tests for `process(StateChange)` dispatch on BrainScoreModel.

Covers:
- Dataclass shape (Selection, Perturbation, StateChange, PerturbationApplied)
- state_change_fn dispatch routes through process()
- Missing state_change_fn raises NotImplementedError
- Bad return type raises TypeError
- handle_id collision raises ValueError
- model.reset() runs all cleanups and clears the registry
- StateChange(kind='reset', handle_id=...) selectively undoes
- A simple synthetic ablation flow: baseline → apply → confirm side-effect →
  reset → side-effect gone
"""

import pytest

from brainscore_core.model_interface import (
    BrainScoreModel,
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
)


# ── Dataclass shape ──────────────────────────────────────────────────

class TestStateChangeShape:

    def test_selection_minimal(self):
        sel = Selection(layer='blocks.10')
        assert sel.layer == 'blocks.10'
        assert sel.indices is None
        assert sel.metadata == {}

    def test_selection_with_indices(self):
        sel = Selection(layer='blocks.10', indices=[1, 2, 3])
        assert sel.indices == [1, 2, 3]

    def test_perturbation_zero(self):
        p = Perturbation(kind='zero')
        assert p.kind == 'zero'
        assert p.scale == 0.0
        assert p.replacement is None

    def test_perturbation_scale(self):
        p = Perturbation(kind='scale', scale=0.5)
        assert p.scale == 0.5

    def test_state_change_apply_form(self):
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        assert sc.kind == 'ablation'
        assert sc.handle_id is None  # apply form has no handle_id

    def test_state_change_reset_form(self):
        sc = StateChange(kind='reset', handle_id='abc-123')
        assert sc.kind == 'reset'
        assert sc.handle_id == 'abc-123'
        assert sc.target is None

    def test_perturbation_applied_round_trip(self):
        sel = Selection(layer='blocks.10', indices=[0, 1])
        p = Perturbation(kind='zero')
        applied = PerturbationApplied(
            handle_id='abc-123', target=sel, perturbation=p, applied_at=42.0,
        )
        assert applied.handle_id == 'abc-123'
        assert applied.target.indices == [0, 1]
        assert applied.applied_at == 42.0


# ── Helpers for dispatch tests ───────────────────────────────────────

def _make_model(state_change_fn=None):
    """Minimal BrainScoreModel for state-change dispatch testing."""
    return BrainScoreModel(
        identifier='test-state-change',
        model=None,
        region_layer_map={},
        preprocessors={'vision': lambda x: x},
        state_change_fn=state_change_fn,
    )


def _counting_state_change_fn():
    """Returns a state_change_fn that records every apply/cleanup call.

    The closure exposes `state['installed']` (set of handle_ids currently
    installed) so tests can verify cleanup runs.
    """
    state = {'next_id': 0, 'installed': set()}

    def fn(state_change: StateChange):
        handle_id = f"handle-{state['next_id']}"
        state['next_id'] += 1
        state['installed'].add(handle_id)

        applied = PerturbationApplied(
            handle_id=handle_id,
            target=state_change.target,
            perturbation=state_change.perturbation,
            applied_at=float(state['next_id']),
        )

        def cleanup():
            state['installed'].discard(handle_id)

        return applied, cleanup

    return fn, state


# ── Dispatch ─────────────────────────────────────────────────────────

class TestProcessStateChangeDispatch:

    def test_no_state_change_fn_raises(self):
        model = _make_model(state_change_fn=None)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        with pytest.raises(NotImplementedError, match='no state_change_fn'):
            model.process(sc)

    def test_apply_returns_perturbation_applied(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(state_change_fn=fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10', indices=[1, 2, 3]),
            perturbation=Perturbation(kind='zero'),
        )
        applied = model.process(sc)
        assert isinstance(applied, PerturbationApplied)
        assert applied.target.indices == [1, 2, 3]
        assert applied.handle_id in state['installed']

    def test_apply_tracks_handle_in_active_perturbations(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(state_change_fn=fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        applied = model.process(sc)
        assert applied.handle_id in model._active_perturbations

    def test_state_change_fn_must_return_tuple(self):
        def bad_fn(state_change):
            return PerturbationApplied(
                handle_id='x',
                target=state_change.target,
                perturbation=state_change.perturbation,
            )  # forgot the cleanup callable

        model = _make_model(state_change_fn=bad_fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        with pytest.raises(TypeError, match='must return a'):
            model.process(sc)

    def test_state_change_fn_must_return_callable_cleanup(self):
        def bad_fn(state_change):
            applied = PerturbationApplied(
                handle_id='x',
                target=state_change.target,
                perturbation=state_change.perturbation,
            )
            return applied, 'not callable'  # second arg not callable

        model = _make_model(state_change_fn=bad_fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        with pytest.raises(TypeError, match='must return a'):
            model.process(sc)

    def test_duplicate_handle_id_raises(self):
        # state_change_fn that always returns the same handle_id
        def colliding_fn(state_change):
            applied = PerturbationApplied(
                handle_id='same-id',
                target=state_change.target,
                perturbation=state_change.perturbation,
            )
            return applied, lambda: None

        model = _make_model(state_change_fn=colliding_fn)
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        )
        model.process(sc)  # first one installs fine
        with pytest.raises(ValueError, match='Duplicate handle_id'):
            model.process(sc)


# ── Reset semantics ──────────────────────────────────────────────────

class TestResetClearsPerturbations:

    def test_reset_runs_all_cleanups(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(state_change_fn=fn)
        for _ in range(3):
            model.process(StateChange(
                kind='ablation',
                target=Selection(layer='blocks.10'),
                perturbation=Perturbation(kind='zero'),
            ))
        assert len(state['installed']) == 3
        model.reset()
        assert state['installed'] == set()
        assert model._active_perturbations == {}

    def test_reset_tolerates_failing_cleanup(self):
        # If one cleanup raises, the others should still run.
        cleanup_log = []
        counter = [0]

        def fn_raising(state_change):
            handle_id = f"h-{counter[0]}"
            counter[0] += 1

            def cleanup():
                if handle_id == 'h-1':
                    raise RuntimeError("cleanup failed")
                cleanup_log.append(handle_id)

            applied = PerturbationApplied(
                handle_id=handle_id,
                target=state_change.target,
                perturbation=state_change.perturbation,
            )
            return applied, cleanup

        model = _make_model(state_change_fn=fn_raising)
        for _ in range(3):
            model.process(StateChange(
                kind='ablation',
                target=Selection(layer='blocks.10'),
                perturbation=Perturbation(kind='zero'),
            ))
        model.reset()  # should NOT raise
        # h-0 and h-2 ran cleanly; h-1 raised but was still tolerated
        assert 'h-0' in cleanup_log and 'h-2' in cleanup_log
        assert model._active_perturbations == {}


class TestSelectiveReset:
    """StateChange(kind='reset', handle_id=X) undoes just one perturbation."""

    def test_kind_reset_removes_one(self):
        fn, state = _counting_state_change_fn()
        model = _make_model(state_change_fn=fn)
        a = model.process(StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        ))
        b = model.process(StateChange(
            kind='ablation',
            target=Selection(layer='blocks.20'),
            perturbation=Perturbation(kind='zero'),
        ))
        assert state['installed'] == {a.handle_id, b.handle_id}

        model.process(StateChange(kind='reset', handle_id=a.handle_id))
        assert state['installed'] == {b.handle_id}
        assert a.handle_id not in model._active_perturbations
        assert b.handle_id in model._active_perturbations

    def test_kind_reset_unknown_handle_raises(self):
        fn, _ = _counting_state_change_fn()
        model = _make_model(state_change_fn=fn)
        with pytest.raises(KeyError, match='No active perturbation'):
            model.process(StateChange(kind='reset', handle_id='nonexistent'))

    def test_kind_reset_missing_handle_id_raises(self):
        fn, _ = _counting_state_change_fn()
        model = _make_model(state_change_fn=fn)
        with pytest.raises(ValueError, match='requires handle_id'):
            model.process(StateChange(kind='reset', handle_id=None))


# ── Synthetic ablation lifecycle ─────────────────────────────────────

class TestSyntheticAblationLifecycle:
    """End-to-end: install an ablation, observe a side effect on a fake
    'forward pass', reset, observe the side effect is gone. No torch — uses
    a plain dict to stand in for a model's hidden state."""

    def test_baseline_apply_observe_reset_observe(self):
        # Fake "model state" that the perturbation will mutate
        model_state = {'output': 1.0}

        def state_change_fn(state_change):
            # Save current value so cleanup can restore
            saved = model_state['output']
            # Apply the ablation: zero out
            if state_change.perturbation.kind == 'zero':
                model_state['output'] = 0.0
            elif state_change.perturbation.kind == 'scale':
                model_state['output'] *= state_change.perturbation.scale

            applied = PerturbationApplied(
                handle_id=f"ablation-{id(state_change)}",
                target=state_change.target,
                perturbation=state_change.perturbation,
            )

            def cleanup():
                model_state['output'] = saved

            return applied, cleanup

        model = _make_model(state_change_fn=state_change_fn)
        # Baseline
        assert model_state['output'] == 1.0
        # Apply ablation
        applied = model.process(StateChange(
            kind='ablation',
            target=Selection(layer='blocks.10'),
            perturbation=Perturbation(kind='zero'),
        ))
        # Observe perturbed state
        assert model_state['output'] == 0.0
        # Reset
        model.reset()
        # Side effect undone
        assert model_state['output'] == 1.0
        assert applied.handle_id not in model._active_perturbations
