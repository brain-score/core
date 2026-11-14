"""The N/A result of ``score()`` for a pair that is not applicable.

A score has four states:

- compatible and scored: the benchmark's actual score;
- compatible but failed: an error; never silently converted to 0;
- not applicable (N/A): the subject cannot take part, and this is known before
  any compute of the run: pre-flight refuses the declarations, or the subject
  refuses at the start of the run's first session. The error is a
  :class:`~brainscore_core.compatibility.CompatibilityError`;
- poor but compatible: the actual score, including a genuine 0.0.

Every ``score()`` entry point (``brainscore``, ``brainscore_vision``,
``brainscore_language``) runs pre-flight and the benchmark in one ``try`` and
passes a ``CompatibilityError`` to :func:`na_score`, which decides. A refusal
after the run computed is an error, not N/A. A memory-check failure is an error.
"""

import numpy as np

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.metrics import Score


STATUS_NA = "N/A"


def na_score(error: CompatibilityError, measures=None) -> Score:
    """The N/A result for an incompatible pair: NaN, never zero.

    ``attrs`` carry ``status`` (``'N/A'``), ``reason`` (the error message) and
    ``missing_channels`` (the channels that make the pair N/A, sorted).

    ``measures`` is the run's ``record_measures()`` record, or None when the
    refusal came from pre-flight, before the benchmark started. When a session of
    the run already delivered an input to the subject or accepted a response, or
    an intervention of the run is still active, the refusal came after compute:
    this raises ``RuntimeError`` (from ``error``) instead of returning N/A. The
    memory probe runs before the record opens, so its compute is not counted.
    """
    if measures is not None and (measures._sessions or measures.unreleased):
        sessions = list(measures._sessions.values())
        raise RuntimeError(
            f"The subject refused after the run computed, so the pair is not N/A: "
            f"{len(sessions)} session(s) of this run had already delivered inputs or "
            f"accepted responses (requested outputs of the last: {list(sessions[-1]) if sessions else []})"
            + (f"; intervention handle(s) {sorted(measures.unreleased)} are still active"
               if measures.unreleased else '')
            + f". Refusal: {error}") from error
    score = Score(np.nan)
    score.attrs["status"] = STATUS_NA
    score.attrs["reason"] = str(error)
    score.attrs["missing_channels"] = sorted(error.channels)
    return score
