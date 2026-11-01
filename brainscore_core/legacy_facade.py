"""Run any subject on legacy vision and language benchmarks.

Legacy benchmarks call ``look_at`` / ``visual_degrees`` / ``start_task`` /
``start_recording`` (vision) or ``digest_text`` / ``start_neural_recording`` /
``start_behavioral_task`` (language). Those benchmarks are never removed, so
this facade is permanent (Contract Rules v3, rule 11.2). It records each
legacy setting, presents the stimuli as a ``BufferedSession`` that carries
those settings (time bins, trial count, task, recording type) and runs the
session through the subject's ``interact``. Nothing here depends on what kind
of subject is wrapped: a ``BrainScoreModel``, a domain adapter and an
interact-only ``Subject`` all take part the same way.

``subject_for_benchmark`` applies the facade at the scoring boundary to
benchmarks a domain loader marked as legacy. ``BrainScoreModel`` forwards its
own legacy methods to a facade seeded with its current recording and task
state, so direct calls keep working too.
"""

import hashlib
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contract import TaskContext
from .streaming import BufferedSession

PASSIVE = 'passive'

LEGACY_API = 'legacy'     # calls look_at / digest_text and friends
TYPED_API = 'typed'       # calls start_recording + process on the candidate
SESSION_API = 'session'   # declares channels and drives interact itself


class LegacyBenchmarkFacade:
    """Present a subject as a legacy ``BrainModel`` / ``ArtificialSubject``.

    :param subject: any subject with ``interact``, ``reset``, ``identifier``
        and ``out_channels``.
    :param visual_degrees: field of view reported to vision benchmarks, or
        ``None`` when the subject declares none.
    """

    def __init__(self, subject, visual_degrees: Optional[float] = None) -> None:
        self._subject = subject
        self._visual_degrees = visual_degrees
        self._record: Any = None          # region name, list of names or layer path
        self._time_bins: Any = None
        self._recording_type: Optional[str] = None
        self._task: Optional[TaskContext] = None

    @property
    def identifier(self) -> str:
        return self._subject.identifier

    @property
    def subject(self):
        return self._subject

    # --- vision: BrainModel ---------------------------------------------

    def visual_degrees(self) -> float:
        if self._visual_degrees is None:
            raise NotImplementedError(
                f"Subject {self.identifier!r} declares no visual field, and "
                "legacy vision benchmarks place stimuli by visual degrees. Pass "
                "visual_degrees= to LegacyBenchmarkFacade or define "
                "visual_degrees() on the subject.")
        return self._visual_degrees

    def start_task(self, task, fitting_stimuli=None, **settings) -> None:
        """Record the task; it reaches the subject with the next measurement."""
        self._task = (task if isinstance(task, TaskContext) else TaskContext(
            task_type=task, fitting_stimuli=fitting_stimuli, metadata=dict(settings)))

    def start_recording(self, recording_target, time_bins=None,
                        recording_type=None) -> None:
        self._record = recording_target
        self._time_bins = time_bins
        self._recording_type = recording_type

    def look_at(self, stimuli, number_of_trials: int = 1,
                require_variance: bool = False, **settings):
        """Legacy ``BrainModel.look_at``: one session requesting the configured output.

        ``number_of_trials`` and ``require_variance`` travel in the session's
        task settings. A subject that cannot produce trial-to-trial variance
        fails instead of returning identical trials.
        """
        if settings:
            raise TypeError(f'look_at got unsupported settings {sorted(settings)}')
        channel = 'behavior' if self._behavior_requested() else self._neural_channel('look_at')
        measured = self._measure(stimuli, [channel], {
            'number_of_trials': number_of_trials,
            'require_variance': require_variance,
        })
        return measured[channel]

    # --- language: ArtificialSubject ------------------------------------

    def start_neural_recording(self, recording_target, recording_type='fMRI') -> None:
        self.start_recording(recording_target, recording_type=recording_type)

    def start_behavioral_task(self, task) -> None:
        if 'behavior' not in set(self._subject.out_channels):
            raise NotImplementedError(
                f"Subject {self.identifier!r} has no path for the legacy language "
                f"behavioral task {getattr(task, 'value', task)!r} (reading times / "
                f"next-word prediction): it declares no 'behavior' output. Legacy "
                f"language behavioral benchmarks need a subject that implements "
                f"that task.")
        self.start_task(task)

    def digest_text(self, text) -> Dict[str, Any]:
        """Legacy ``ArtificialSubject.digest_text``: every configured output in one call.

        Returns ``{'neural': ...}`` and/or ``{'behavior': ...}`` according to
        what ``start_neural_recording`` / ``start_behavioral_task`` configured.
        Plain text is handed to the subject unchanged as ``legacy_text`` in the
        task settings, so a legacy adapter can replay the exact original call.
        """
        stimuli = text_stimulus_set(text)
        channels: Dict[str, str] = {}
        if self._record is not None:
            channels['neural'] = self._neural_channel('digest_text')
        if self._behavior_requested():
            channels['behavior'] = 'behavior'
        if not channels:
            raise ValueError('Call start_neural_recording or start_behavioral_task '
                             'before digest_text')
        settings = {} if stimuli is text else {'legacy_text': text}
        measured = self._measure(stimuli, list(channels.values()), settings)
        return {name: measured[channel] for name, channel in channels.items()}

    def reset(self) -> None:
        self._subject.reset()
        self._record = self._time_bins = self._recording_type = self._task = None

    # --- internals --------------------------------------------------------

    def _behavior_requested(self) -> bool:
        return self._task is not None and self._task.task_type != PASSIVE

    def _neural_channel(self, method: str) -> str:
        if self._record is None:
            raise ValueError(f'Call start_recording or start_task before {method}')
        if not isinstance(self._record, str):
            raise ValueError(
                f'{method} returns one region; several were requested '
                f'({list(self._record)}). Use neural_response(..., record=[...]) '
                f'or process() for several regions.')
        return f'neural:{self._record}'

    def _measure(self, stimuli, channels: List[str], settings: Dict[str, Any]):
        if self._recording_type is not None:
            settings['recording_type'] = self._recording_type
        context = self._task or TaskContext(task_type=PASSIVE)
        context = replace(context, metadata={**context.metadata, **settings})
        session = BufferedSession(stimulus_set=stimuli,
                                  requested_output_channels=channels,
                                  task_context=context, time_bins=self._time_bins)
        self._subject.interact(session)
        return {channel: session.collect(channel) for channel in channels}


def mark_legacy_benchmark(benchmark):
    """Record a benchmark's calling convention at a domain loader.

    A benchmark that declares its own input channels drives sessions itself;
    one that declares none calls the legacy methods. A benchmark class may
    set ``subject_api`` itself (``'typed'`` for the retained
    ``start_recording`` + ``process`` convention).
    """
    if getattr(benchmark, 'subject_api', None) is None:
        declares_channels = getattr(benchmark, 'required_input_channels', None) is not None
        benchmark.subject_api = SESSION_API if declares_channels else LEGACY_API
    return benchmark


def subject_for_benchmark(subject, benchmark, *, visual_degrees: Optional[float] = None):
    """The candidate a benchmark receives: the facade for legacy benchmarks, else the subject."""
    if getattr(benchmark, 'subject_api', None) != LEGACY_API:
        return subject
    return LegacyBenchmarkFacade(subject, visual_degrees=visual_degrees)


def declared_visual_degrees(subject) -> Optional[float]:
    """The field of view a vision candidate reports, or ``None`` when it has none.

    ``visual_degrees()`` is part of the legacy vision ``BrainModel`` surface, a
    measurement setting of the vision harness rather than an ability; a
    subject without it gets no invented value.
    """
    degrees = getattr(subject, 'visual_degrees', None)
    if degrees is None:
        return None
    return degrees() if callable(degrees) else degrees


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
