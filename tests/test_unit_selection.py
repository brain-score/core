"""UnitSelection family: portable, resolve(model)-based perturbation selection.

Lifts the functional localizer (word-vs-non-word, Cohen's d ranking) and its
random-unit control out of the experiment sweep scripts into reusable core
classes. Unit tests run against a fake Subject (a synthetic assembly — no model
weights); integration tests run through BrainScoreModel.process(StateChange) to
prove a UnitSelection target is resolved to a Selection before state_change_fn
sees it, and that the localizer's recording side-effect is restored.
"""
import numpy as np
import pytest
import xarray as xr

from brainscore_core.model_interface import (
    BrainScoreModel,
    Selection,
    Perturbation,
    PerturbationApplied,
    StateChange,
    UnitSelection,
    FunctionalSelection,
    RandomSelection,
    IndexSelection,
)


# ── Fake Subject that returns a synthetic localizer assembly ─────────────

def _localizer_assembly(layer='enc.5', n_neuro=10, discriminating=5):
    """8 presentations (4 'word', 4 'nonword'); the first `discriminating`
    units respond strongly to 'word' (large Cohen's d), the rest have no group
    difference (d == 0, deterministically — not noise that might spike)."""
    labels = ['word'] * 4 + ['nonword'] * 4
    word = np.array([l == 'word' for l in labels])
    rng = np.random.RandomState(0)
    data = np.zeros((8, n_neuro))
    # discriminating units: small within-group noise (so pooled std > 0) + a
    # strong +3 word bump -> d is large and finite
    data[:, :discriminating] = rng.normal(0, 0.1, (8, discriminating))
    data[np.ix_(word, np.arange(discriminating))] += 3.0
    # non-discriminating units: constant per unit -> identical across groups -> d == 0
    for j in range(discriminating, n_neuro):
        data[:, j] = float(j)
    return xr.DataArray(
        data, dims=('presentation', 'neuroid'),
        coords={'label': ('presentation', labels),
                'layer': ('neuroid', [layer] * n_neuro)})


class _FakeSubject:
    def __init__(self, assembly):
        self._asm = assembly
        self.recorded = None

    def start_recording(self, target, time_bins=None, recording_type=None):
        self.recorded = target

    def process(self, stimuli, **kw):
        return self._asm


# ── FunctionalSelection ──────────────────────────────────────────────

class TestFunctionalSelection:

    def test_positive_picks_discriminating_units(self):
        model = _FakeSubject(_localizer_assembly())
        sel = FunctionalSelection(
            recording_target='VWFA', localizer_stimuli='stim',
            contrast=(['word'], ['nonword']), n_units=5, sign='positive',
        ).resolve(model)
        assert isinstance(sel, Selection)
        assert sel.indices == [0, 1, 2, 3, 4]
        assert sel.layer == 'enc.5'                  # read from the assembly's layer coord
        assert sel.metadata['selector'] == 'functional'
        assert sel.metadata['n_recorded'] == 10
        assert model.recorded == 'VWFA'              # recorded the requested region

    def test_negative_picks_the_complement(self):
        model = _FakeSubject(_localizer_assembly())
        sel = FunctionalSelection(
            recording_target='VWFA', localizer_stimuli='stim',
            contrast=(['word'], ['nonword']), n_units=5, sign='negative',
        ).resolve(model)
        assert sel.indices == [5, 6, 7, 8, 9]

    def test_abs_matches_positive_here(self):
        model = _FakeSubject(_localizer_assembly())
        sel = FunctionalSelection(
            recording_target='VWFA', localizer_stimuli='stim',
            contrast=(['word'], ['nonword']), n_units=5, sign='abs',
        ).resolve(model)
        assert sel.indices == [0, 1, 2, 3, 4]

    def test_threshold_mode_selects_above_cutoff(self):
        model = _FakeSubject(_localizer_assembly())
        sel = FunctionalSelection(
            recording_target='VWFA', localizer_stimuli='stim',
            contrast=(['word'], ['nonword']), n_units=None, threshold=2.0,
        ).resolve(model)
        assert sel.indices == [0, 1, 2, 3, 4]         # only the strong units clear d>=2

    def test_empty_contrast_group_raises(self):
        model = _FakeSubject(_localizer_assembly())
        with pytest.raises(ValueError, match='non-empty'):
            FunctionalSelection(
                recording_target='VWFA', localizer_stimuli='stim',
                contrast=(['absent_label'], ['nonword']),
            ).resolve(model)

    def test_bad_sign_raises(self):
        model = _FakeSubject(_localizer_assembly())
        with pytest.raises(ValueError, match='sign must be'):
            FunctionalSelection(
                recording_target='VWFA', localizer_stimuli='stim',
                contrast=(['word'], ['nonword']), sign='sideways',
            ).resolve(model)

    def test_multi_layer_recording_raises(self):
        asm = _localizer_assembly()
        asm = asm.assign_coords(layer=('neuroid', ['a'] * 5 + ['b'] * 5))
        with pytest.raises(ValueError, match='spans 2 layers'):
            FunctionalSelection(
                recording_target='VWFA', localizer_stimuli='stim',
                contrast=(['word'], ['nonword']), n_units=3,
            ).resolve(_FakeSubject(asm))

    def test_collapses_extra_time_bin_dim(self):
        asm = _localizer_assembly().expand_dims({'time_bin': 3}).copy()
        model = _FakeSubject(asm)
        sel = FunctionalSelection(
            recording_target='VWFA', localizer_stimuli='stim',
            contrast=(['word'], ['nonword']), n_units=5,
        ).resolve(model)
        assert sel.indices == [0, 1, 2, 3, 4]


# ── RandomSelection (the causal control) ──────────────────────────────

class TestRandomSelection:

    def test_deterministic_and_in_range(self):
        sel = RandomSelection(layer='enc.5', n_units=3, n_total=10, seed=0).resolve(None)
        assert isinstance(sel, Selection)
        assert len(sel.indices) == 3
        assert all(0 <= i < 10 for i in sel.indices)
        again = RandomSelection(layer='enc.5', n_units=3, n_total=10, seed=0).resolve(None)
        assert sel.indices == again.indices            # deterministic by seed

    def test_different_seed_differs(self):
        a = RandomSelection(layer='enc.5', n_units=5, n_total=20, seed=0).resolve(None).indices
        b = RandomSelection(layer='enc.5', n_units=5, n_total=20, seed=1).resolve(None).indices
        assert a != b

    def test_clamps_to_n_total(self):
        sel = RandomSelection(layer='enc.5', n_units=99, n_total=10, seed=0).resolve(None)
        assert len(sel.indices) == 10


# ── IndexSelection ────────────────────────────────────────────────────

def test_index_selection_passthrough():
    sel = IndexSelection(layer='enc.5', indices=[2, 7]).resolve(None)
    assert sel.layer == 'enc.5'
    assert sel.indices == [2, 7]
    assert sel.metadata['selector'] == 'index'


# ── Integration through BrainScoreModel.process(StateChange) ──────────

def _make_model(state_change_fn=None, region_layer_map=None):
    return BrainScoreModel(
        identifier='test-unit-selection', model=None,
        region_layer_map=region_layer_map or {},
        preprocessors={'vision': lambda x: x},
        state_change_fn=state_change_fn,
    )


def _capturing_state_change_fn(captured):
    def fn(sc):
        captured['target'] = sc.target
        applied = PerturbationApplied(handle_id='h0', target=sc.target,
                                      perturbation=sc.perturbation)
        return applied, lambda: None
    return fn


def test_dispatch_resolves_unit_selection_before_state_change_fn():
    captured = {}
    model = _make_model(state_change_fn=_capturing_state_change_fn(captured))
    sc = StateChange(kind='ablation',
                     target=IndexSelection(layer='blocks.10', indices=[1, 2, 3]),
                     perturbation=Perturbation(kind='zero'))
    model.process(sc)
    # state_change_fn saw a concrete Selection, not the UnitSelection
    assert isinstance(captured['target'], Selection)
    assert captured['target'].indices == [1, 2, 3]
    assert captured['target'].layer == 'blocks.10'
    # the caller's StateChange was not mutated
    assert isinstance(sc.target, IndexSelection)


def test_dispatch_passes_plain_selection_through_unchanged():
    captured = {}
    model = _make_model(state_change_fn=_capturing_state_change_fn(captured))
    sc = StateChange(kind='ablation',
                     target=Selection(layer='b', indices=[0]),
                     perturbation=Perturbation(kind='zero'))
    model.process(sc)
    assert captured['target'].indices == [0]


def test_dispatch_restores_recording_after_localizer():
    """A UnitSelection that switches recording during resolve must not leave the
    model's recording target changed."""
    class _SwitchSelection(UnitSelection):
        def resolve(self, model):
            model.start_recording('OTHER')          # localizer-style side effect
            return Selection(layer='enc.2', indices=[0])

    captured = {}
    model = _make_model(state_change_fn=_capturing_state_change_fn(captured),
                        region_layer_map={'IT': 'enc.1', 'OTHER': 'enc.2'})
    model.start_recording('IT')
    sc = StateChange(kind='ablation', target=_SwitchSelection(),
                     perturbation=Perturbation(kind='zero'))
    model.process(sc)
    assert captured['target'].indices == [0]
    assert model._recording_regions == ['IT']        # restored to the pre-localizer target
