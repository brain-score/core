"""Null controls for Brain-Score experiments.

A score is only trustworthy relative to a floor. Every capability has a
matched null that runs through the *same* pipeline but with the signal
destroyed in one specific, interpretable way:

==========================  ==================================================
capability                  matched null
==========================  ==================================================
encoding (neural)           ``shuffle_rows`` (break stimulus<->response pairing),
                            ``shift_features`` (mis-time), random-init model
behavioral                  ``shuffle_labels``, chance baseline
temporal / multimodal       ``shift_features`` along time (mis-align modality
                            vs brain), ``temporal_shift_curve`` (degradation)
state-change / ablation     ``random_unit_subset`` (ablate random units),
                            opposite-sign perturbation
unit / composite select     ``random_unit_subset`` (vs the selected population)
embodied (policy)           ``random_action`` (vs the learned policy)
==========================  ==================================================

These transforms are deliberately mechanical so the floor is unambiguous: if
the real score does not clear the matched null, the apparent signal is an
artifact of the pipeline (leakage, an over-expressive readout, an alignment
coincidence) rather than the model. The single most informative null for a
temporal/multimodal encoding benchmark is ``shift_features``: rolling the
model features by a few TRs relative to the brain should collapse the score
toward the ``shuffle_rows`` floor; if it does *not*, the "alignment" was never
carrying stimulus-locked information in the first place.

All functions are deterministic given ``seed`` and return copies — they never
mutate their inputs. They operate on plain arrays (the design matrix ``X`` of
shape ``(n_samples, n_features)`` and target ``Y`` of shape
``(n_samples, n_targets)``) so they compose with any scoring routine.
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Capability -> applicable nulls. A lookup so an experiment driver can assert
# it ran every null that its capability demands.
# ---------------------------------------------------------------------------
NULLS_BY_CAPABILITY: Dict[str, List[str]] = {
    'neural_encoding': ['shuffle_rows', 'shift_features', 'random_init_model'],
    'behavioral': ['shuffle_labels', 'chance_baseline', 'random_init_model'],
    'temporal_multimodal': ['shift_features', 'shuffle_rows', 'random_init_model'],
    'state_change': ['random_unit_subset', 'opposite_sign'],
    'unit_selection': ['random_unit_subset'],
    'composite_selection': ['random_unit_subset'],
    'embodied': ['random_action', 'random_init_model'],
}


def _rng(seed: Optional[int]) -> np.random.RandomState:
    # RandomState (not default_rng) for reproducibility parity with the rest
    # of the codebase, which pins numpy<2 and uses RandomState everywhere.
    return np.random.RandomState(seed)


def shuffle_rows(X: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Permute the sample axis, breaking the stimulus<->response pairing.

    The canonical encoding null: a model regressed against shuffled features
    can only fit whatever structure survives a random pairing, which is
    nothing in expectation. A real encoding score must clear this floor.

    Returns a copy with rows in a random order (same shape, same marginal
    feature distribution per column).
    """
    X = np.asarray(X)
    perm = _rng(seed).permutation(X.shape[0])
    return X[perm].copy()


def shift_features(X: np.ndarray, shift: int) -> np.ndarray:
    """Temporal-misalignment null: roll the sample axis by ``shift``.

    For a feature matrix whose rows are consecutive TRs, shifting by a few
    rows mis-times the features relative to the brain response while
    preserving each column's marginal distribution and temporal
    autocorrelation. The score should degrade toward the shuffle floor; the
    *rate* of degradation as ``|shift|`` grows (see
    :func:`temporal_shift_curve`) quantifies how stimulus-locked the signal
    is.

    ``shift`` may be positive or negative. A circular roll (no NaN edges) is
    used so the null reuses every sample exactly once.
    """
    X = np.asarray(X)
    return np.roll(X, shift, axis=0).copy()


def shuffle_labels(y: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Permute behavioral labels — the readout floor.

    A logistic / linear readout trained on shuffled labels reports the
    label-balance chance level plus whatever a finite training set lets it
    overfit. A real behavioral score must clear this.
    """
    y = np.asarray(y)
    perm = _rng(seed).permutation(y.shape[0])
    return y[perm].copy()


def random_unit_subset(n_total: int, k: int, *, seed: int = 0) -> np.ndarray:
    """Random ``k`` of ``n_total`` unit indices (sorted, no replacement).

    The matched null for unit-selection / composite-selection and for
    targeted ablation: if a *curated* population (e.g. top-K Cohen's-d units,
    or a CompositeSelector across layers) is load-bearing, ablating or reading
    out a *random* same-size population should produce a measurably weaker
    effect. When random == curated, the selection carried no information.
    """
    if k > n_total:
        raise ValueError(f"k={k} exceeds n_total={n_total}.")
    return np.sort(_rng(seed).choice(n_total, size=k, replace=False))


def random_action(action_shape: Sequence[int], *, seed: int = 0,
                  low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Uniform random action in ``[low, high]`` — the embodied policy floor.

    A learned policy stepping a game/robot must outperform an agent that
    emits random actions of the same shape. Without this, a high task-success
    number could come from an environment that is trivially solvable.
    """
    return _rng(seed).uniform(low, high, size=tuple(action_shape))


def temporal_shift_curve(score_fn: Callable[[np.ndarray], float],
                         X: np.ndarray,
                         shifts: Sequence[int]) -> Dict[int, float]:
    """Score the same model at a range of temporal mis-alignments.

    ``score_fn`` takes a (possibly shifted) feature matrix and returns a
    scalar score (it closes over the targets + CV). Returns ``{shift: score}``.
    The expected shape is a peak at ``shift == 0`` falling off symmetrically;
    a flat curve means the alignment was never load-bearing.
    """
    return {int(s): float(score_fn(shift_features(X, s))) for s in shifts}


def describe(capability: str) -> List[str]:
    """Return the nulls that a given capability must be validated against."""
    if capability not in NULLS_BY_CAPABILITY:
        raise ValueError(
            f"unknown capability {capability!r}; known: "
            f"{sorted(NULLS_BY_CAPABILITY)}.")
    return list(NULLS_BY_CAPABILITY[capability])
