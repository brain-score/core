"""Helpers for explicit score result states.

UMI v2.0 distinguishes four score states:

- compatible and scored: return the benchmark's actual score
- compatible but failed: report failure metadata; never silently convert to 0
- incompatible by contract: return an explicit N/A score when a caller asks for
  N/A conversion
- poor but compatible: return the actual score, including a genuine 0.0

The domain ``score()`` entry points keep their existing behavior and raise
``CompatibilityError`` before compute. Orchestration layers that need a result
object can catch that typed error directly, or call ``score_or_na``.
"""

from typing import Callable, TypeVar

import numpy as np

from .compatibility import CompatibilityError
from .metrics import Score


STATUS_NA = "N/A"
_T = TypeVar("_T")


def na_score(reason: str, *, error_type: str | None = None) -> Score:
    """Return a distinguishable N/A score for an incompatible pair."""
    score = Score(np.nan)
    score.attrs["status"] = STATUS_NA
    score.attrs["reason"] = str(reason)
    if error_type is not None:
        score.attrs["error_type"] = error_type
    return score


def score_or_na(score_function: Callable[..., _T], *args, **kwargs) -> _T | Score:
    """Run ``score_function`` and convert only compatibility failures to N/A.

    Other exceptions propagate so compatible-but-failed pairs remain failures,
    not N/A and not zero.
    """
    try:
        return score_function(*args, **kwargs)
    except CompatibilityError as error:
        return na_score(str(error), error_type=type(error).__name__)
