"""Run typed subjects on legacy vision and language benchmarks.

Legacy benchmarks call ``look_at`` / ``visual_degrees`` / ``start_task`` /
``start_recording`` (vision) or ``digest_text`` / ``start_neural_recording`` /
``start_behavioral_task`` (language). Those benchmarks are never removed, so
this facade is permanent (Contract Rules v3, rule 11.2). It translates each
legacy call onto the typed subject's ``process`` / ``start_*`` paths and is the
only place that legacy surface is implemented. ``BrainScoreModel`` exposes the
same methods as one-line forwards to a facade around itself.
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple


class LegacyBenchmarkFacade:
    """Present a typed subject as a legacy ``BrainModel`` / ``ArtificialSubject``.

    :param subject: a typed subject (``process``, ``start_task``,
        ``start_recording``, ``reset``), e.g. a ``BrainScoreModel``.
    :param visual_degrees: field of view reported to vision benchmarks.
    """

    def __init__(self, subject, visual_degrees: int = 8) -> None:
        self._subject = subject
        self._visual_degrees = visual_degrees

    @property
    def identifier(self) -> str:
        return self._subject.identifier

    @property
    def subject(self):
        return self._subject

    # --- vision: BrainModel ---------------------------------------------

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task, fitting_stimuli=None, **kwargs) -> None:
        self._subject.start_task(task, fitting_stimuli, **kwargs)

    def start_recording(self, recording_target, time_bins=None,
                        recording_type=None) -> None:
        self._subject.start_recording(recording_target, time_bins=time_bins,
                                      recording_type=recording_type)

    def look_at(self, stimuli, number_of_trials: int = 1,
                require_variance: bool = False, **kwargs):
        """Legacy ``BrainModel.look_at``.

        ``number_of_trials`` only changes legacy responses together with
        ``require_variance`` (repeated noisy presentations); a deterministic
        subject returns the same response either way. A request for trial
        variance the subject cannot produce fails instead of returning
        identical trials.
        """
        del kwargs
        if require_variance:
            raise NotImplementedError(
                f"Subject {self.identifier!r} cannot produce trial-to-trial "
                f"variance (look_at(require_variance=True, number_of_trials="
                f"{number_of_trials})). Legacy models add it with microsaccades; "
                f"this subject has no equivalent.")
        return self._subject.process(stimuli)

    # --- language: ArtificialSubject ------------------------------------

    def start_neural_recording(self, recording_target, recording_type='fMRI') -> None:
        self._subject.start_recording(recording_target, recording_type=recording_type)

    def start_behavioral_task(self, task) -> None:
        raise NotImplementedError(
            f"Subject {self.identifier!r} has no path for the legacy language "
            f"behavioral task {getattr(task, 'value', task)!r} (reading times / "
            f"next-word prediction). Legacy language behavioral benchmarks need "
            f"a subject that implements that task.")

    def digest_text(self, text) -> Dict[str, Any]:
        """Legacy ``ArtificialSubject.digest_text``: neural part only."""
        return {'neural': self._subject.process(text_stimulus_set(text))}

    def reset(self) -> None:
        self._subject.reset()


def text_stimulus_set(text):
    """The stimulus set legacy ``digest_text`` callers mean by ``text``.

    A string or list of strings is one passage, in order. A stimulus set is
    passed through unchanged. The identifier hashes the content so different
    passages never share a cache entry.
    """
    import numpy as np
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    if isinstance(text, np.ndarray):
        text = text.tolist()
    if not isinstance(text, (str, list)):
        return text
    if isinstance(text, str):
        text = [text]
    stimuli = StimulusSet(pd.DataFrame({
        'sentence': text,
        'stimulus_id': list(range(len(text))),
        'context_id': ['passage'] * len(text),
    }))
    content_hash = hashlib.md5('|'.join(text).encode('utf-8')).hexdigest()[:12]
    stimuli.identifier = f'text_stimuli_{len(text)}_{content_hash}'
    return stimuli


def repeat_over_time_bins(assembly, time_bins: Optional[Sequence[Tuple[int, int]]]):
    """Repeat a static response across several requested time bins.

    Matches the legacy vision behavior for models without a time axis: one
    requested bin leaves the response unchanged; several bins repeat it along a
    ``time_bin`` dimension indexed by ``(time_bin_start, time_bin_end)``.
    """
    bins = _time_bin_pairs(time_bins)
    if len(bins) < 2 or 'time_bin' in getattr(assembly, 'dims', ()):
        return assembly
    import pandas as pd
    import xarray as xr

    plain = xr.DataArray(assembly)
    stacked = xr.concat([plain] * len(bins), dim='time_bin')
    stacked = stacked.assign_coords(time_bin=pd.MultiIndex.from_tuples(
        bins, names=['time_bin_start', 'time_bin_end']))
    return type(assembly)(stacked)


def _time_bin_pairs(time_bins) -> List[Tuple[Any, Any]]:
    """Normalize benchmark time bins (lists, arrays, MultiIndex values) to pairs."""
    import numpy as np
    if time_bins is None:
        return []
    items = time_bins.tolist() if isinstance(time_bins, np.ndarray) else list(time_bins)
    if len(items) == 2 and all(np.isscalar(value) for value in items):
        return [(items[0], items[1])]  # one flat (start, end) pair
    pairs = []
    for item in items:
        item = item.tolist() if isinstance(item, np.ndarray) else item
        start, end = item
        pairs.append((start, end))
    return pairs
