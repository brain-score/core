"""Small numeric primitives shared across metrics + benchmark scorers.

Pure numpy, no heavy deps — the canonical home for per-unit correlation so the
centered-Pearson formula isn't re-derived in every metric and benchmark scorer
(it had drifted into several near-identical copies across the unified package).
"""
import numpy as np


def per_unit_pearson(A, B):
    """Per-column Pearson r between two ``(n_samples, n_units)`` matrices.

    Each column is centered over samples; a zero-variance column yields NaN.
    Callers that must drop unpredicted (NaN) rows should mask before calling.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    den = np.sqrt((Ac ** 2).sum(axis=0) * (Bc ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return (Ac * Bc).sum(axis=0) / np.where(den == 0, np.nan, den)
