"""The N/A result of ``score()`` for a pair that is not applicable.

A score has four states:

- compatible and scored: the benchmark's actual score;
- compatible but failed: an error; never silently converted to 0;
- not applicable (N/A): the subject cannot take part, decided from the
  declarations in pre-flight or by the subject at the start of a session; the
  error is a :class:`~brainscore_core.compatibility.CompatibilityError`;
- poor but compatible: the actual score, including a genuine 0.0.

Every ``score()`` entry point (``brainscore``, ``brainscore_vision``,
``brainscore_language``) runs pre-flight and the benchmark in one ``try`` and
returns :func:`na_score` for a ``CompatibilityError``. A memory-check failure is
an error, not N/A.
"""

import numpy as np

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.metrics import Score


STATUS_NA = "N/A"


def na_score(error: CompatibilityError) -> Score:
    """The N/A result for an incompatible pair: NaN, never zero.

    ``attrs`` carry ``status`` (``'N/A'``), ``reason`` (the error message) and
    ``missing_channels`` (the channels that make the pair N/A, sorted).
    """
    score = Score(np.nan)
    score.attrs["status"] = STATUS_NA
    score.attrs["reason"] = str(error)
    score.attrs["missing_channels"] = sorted(error.channels)
    return score
