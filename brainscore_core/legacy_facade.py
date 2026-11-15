"""Run any subject on legacy vision and language benchmarks.

Legacy benchmarks call ``look_at`` / ``visual_degrees`` / ``start_task`` /
``start_recording`` (vision) or ``digest_text`` / ``start_neural_recording`` /
``start_behavioral_task`` (language). Those benchmarks are never removed, so
this facade is permanent (Contract Rules v3, rule 11.2). It records each
legacy setting, presents the stimuli as a ``Session`` that carries
those settings (time bins, trial count, task, recording type) and runs the
session through the subject's ``participate``. Nothing here depends on what kind
of subject is wrapped: a ``BrainScoreModel``, a domain adapter and an
participate-only ``Subject`` all take part the same way.

``subject_for_benchmark`` applies the facade at the scoring boundary to
benchmarks a domain loader marked as legacy. A subject has no legacy methods
of its own: every legacy call goes through this facade.
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core._sessions.buffered import stimulus_session

PASSIVE = 'passive'

LEGACY_API = 'legacy'     # calls look_at / digest_text and friends
TYPED_API = 'typed'       # calls start_recording + process on the subject
SESSION_API = 'session'   # declares channels and drives participate itself


class LegacyBenchmarkFacade:
    """Present a subject as a legacy ``BrainModel`` / ``ArtificialSubject``.

    :param subject: any subject with ``participate``, ``reset``, ``identifier``
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
        """Record the task; it reaches the subject with the next measurement.

        Keyword settings a benchmark passes to ``start_task`` (Malania2007
        sends ``number_of_trials`` and ``require_variance``) are fitting
        settings. They travel as ``metadata['task_settings']``, apart from the
        measurement settings of each ``look_at``, which may use the same names.
        The one TaskContext object is sent with every measurement, so a subject
        can tell a repeated task from a new one and keep its fitted state.
        """
        if isinstance(task, TaskContext):
            self._task = task
            return
        metadata = {'task_settings': dict(settings)} if settings else {}
        labels = _label_list(task, fitting_stimuli)
        if labels is not None:
            # The label task with a list of labels is a choice among them
            # (behavior:choice); the benchmark still receives the label of each stimulus.
            from brainscore_core.responses import Choice
            self._task = TaskContext(response=Choice(labels), metadata=metadata)
            return
        self._task = TaskContext(task_type=task, fitting_stimuli=fitting_stimuli,
                                 metadata=metadata)

    def start_recording(self, recording_target, time_bins=None,
                        recording_type=None) -> None:
        self._record = recording_target
        self._time_bins = time_bins
        self._recording_type = recording_type

    def look_at(self, stimuli, number_of_trials: int = 1,
                require_variance: bool = False, **settings):
        """Legacy ``BrainModel.look_at``: one session requesting the configured output.

        ``number_of_trials`` and ``require_variance`` travel as the session's
        measurement ``settings``. A subject that cannot produce trial-to-trial
        variance fails instead of returning identical trials.
        """
        if settings:
            raise TypeError(f'look_at got unsupported settings {sorted(settings)}')
        if self._choice_requested():
            channel = 'behavior:choice'
            if channel not in set(self._subject.out_channels):
                raise ExecutionNotFeasibleError(
                    f"Subject {self.identifier!r} refused the legacy label task: it is asked "
                    f"as a choice among the labels, and the subject declares no "
                    f"'behavior:choice' output.", channels={channel}, task_type='label')
        else:
            channel = 'behavior' if self._behavior_requested() else self._neural_channel('look_at')
        measured = self._measure(stimuli, [channel], {
            'number_of_trials': number_of_trials,
            'require_variance': require_variance,
        })
        if channel == 'behavior:choice':
            return _labels_from_choice(resolve_invalid_answers(measured[channel]))
        if channel == 'behavior':
            return resolve_invalid_answers(measured[channel])
        return measured[channel]

    # --- language: ArtificialSubject ------------------------------------

    def start_neural_recording(self, recording_target, recording_type='fMRI') -> None:
        self.start_recording(recording_target, recording_type=recording_type)

    def start_behavioral_task(self, task) -> None:
        if 'behavior' not in set(self._subject.out_channels):
            # A refusal when the run starts: an error of score().
            raise ExecutionNotFeasibleError(
                f"Subject {self.identifier!r} refused the legacy language behavioral task "
                f"{getattr(task, 'value', task)!r} (reading times / next-word prediction): "
                f"it declares no 'behavior' output. The pair passed the declared "
                f"compatibility check because this legacy benchmark declares no requested "
                f"output, so there is no result; the benchmark's declarations do not yet "
                f"express that it needs 'behavior'.", channels={'behavior'},
                task_type=getattr(task, 'value', task))
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
        try:
            self._subject.reset()
        finally:
            self._record = self._time_bins = self._recording_type = self._task = None
            self._passive = None

    # --- internals --------------------------------------------------------

    _passive: Optional[TaskContext] = None

    def _behavior_requested(self) -> bool:
        return self._task is not None and self._task.task_type != PASSIVE

    def _choice_requested(self) -> bool:
        from brainscore_core.responses import Choice
        return self._task is not None and isinstance(self._task.response, Choice)

    def _session_task(self) -> TaskContext:
        """The task sent with each measurement: the recorded one, else one passive task."""
        if self._task is not None:
            return self._task
        if self._passive is None:
            self._passive = TaskContext(task_type=PASSIVE)
        return self._passive

    def _neural_channel(self, method: str) -> str:
        if self._record is None:
            raise ValueError(f'Call start_recording or start_task before {method}')
        if not isinstance(self._record, str):
            raise ValueError(
                f'{method} returns one region; several were requested '
                f'({list(self._record)}). Use a session with one neural channel per region, '
                "for example stimulus_session(stimuli, ['neural:V4', 'neural:IT']).")
        return f'neural:{self._record}'

    def _measure(self, stimuli, channels: List[str], settings: Dict[str, Any]):
        """One session: the recorded task object, the time bins and the measurement settings."""
        if self._recording_type is not None:
            settings['recording_type'] = self._recording_type
        session = stimulus_session(stimuli, channels, self._session_task(),
                                   time_bins=self._time_bins, settings=dict(settings))
        self._subject.participate(session)
        return {channel: session.collect(channel) for channel in channels}


def _label_list(task, fitting_stimuli):
    """The labels of a legacy label task with a list of labels, else ``None``.

    ``start_task('label', [...])`` names the options of a forced choice. The label
    task with ``'imagenet'`` (1000 synsets, which core does not list) and every other
    task stay on bare ``behavior``.
    """
    if task != 'label' or isinstance(fitting_stimuli, str) or \
            not isinstance(fitting_stimuli, (list, tuple)):
        return None
    labels = list(fitting_stimuli)
    if len(labels) < 2 or len(set(labels)) != len(labels) or \
            not all(isinstance(label, str) for label in labels):
        return None
    return labels


# Coordinates core adds to a stored answer; the legacy label answer has none of them.
_STORED_ONLY = ('raw_answer', 'response_state', 'response_reason', 'answer_valid', 'parse_error')


def _labels_from_choice(answer):
    """A stored ``behavior:choice`` answer as the legacy label answer.

    One label per stimulus, dims ``('choice', 'presentation')``, with every
    presentation coordinate of the stored answer in its order, except the ones core
    adds to a stored answer (``_STORED_ONLY``), and every attribute except the
    format's response record (``n_invalid_answers`` stays). Each row is one-hot (an invalid row
    was made a valid label by ``resolve_invalid_answers`` before).
    """
    import numpy as np
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    labels = [str(label) for label in answer['choice'].values]
    rows = np.asarray(answer.transpose('presentation', 'choice').values)
    chosen = [labels[int(np.argmax(row))] for row in rows]
    index = answer.indexes.get('presentation')
    names = [name for name in (getattr(index, 'names', None) or ()) if name and name != 'presentation']
    names += [name for name, coord in answer.coords.items()
              if coord.dims == ('presentation',) and name != 'presentation' and name not in names]
    coords = {name: ('presentation', list(answer[name].values))
              for name in names if name not in _STORED_ONLY}
    # Every attribute that is not the format's own response record stays, among them
    # the count of invalid answers that resolve_invalid_answers set.
    attrs = {key: value for key, value in answer.attrs.items() if key != 'response'}
    return BehavioralAssembly([chosen], coords=coords, dims=['choice', 'presentation'], attrs=attrs)


def mark_legacy_benchmark(benchmark):
    """Record a benchmark's calling convention at a domain loader.

    A benchmark that declares its own input channels drives sessions itself;
    one that declares none calls the legacy methods. A benchmark class may
    set ``subject_api`` itself (``'typed'``: the subject receives the
    ``start_recording`` + ``process`` calls itself; no registered benchmark
    sets it since the ``-unified`` MajajHong2015 and Pereira2018 copies were
    deleted).
    """
    if getattr(benchmark, 'subject_api', None) is None:
        declares_channels = getattr(benchmark, 'required_input_channels', None) is not None
        benchmark.subject_api = SESSION_API if declares_channels else LEGACY_API
    return benchmark


def subject_for_benchmark(subject, benchmark, *, visual_degrees: Optional[float] = None):
    """What a benchmark receives: the facade for legacy benchmarks, else the subject."""
    if getattr(benchmark, 'subject_api', None) != LEGACY_API:
        return subject
    return LegacyBenchmarkFacade(subject, visual_degrees=visual_degrees)


def declared_visual_degrees(subject) -> Optional[float]:
    """The field of view a vision subject reports, or ``None`` when it has none.

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
    # Length prefixes keep different segmentations of the same characters
    # (['a|b', 'c'] and ['a', 'b|c']) from sharing one cache entry.
    content = ''.join(f'{len(part)}:{part}' for part in text).encode('utf-8')
    content_hash = hashlib.md5(content).hexdigest()[:12]
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




# Above this share of invalid answers the model is not doing the task; scoring
# its guesses would report noise as alignment.
MAX_INVALID_ANSWER_FRACTION = 0.1


def resolve_invalid_answers(assembly, *, seed: int = 0):
    """Replace invalid generated answers with a seeded random valid label.

    Legacy behavioral benchmarks cannot express "invalid": probability metrics
    need a distribution and choice metrics take an argmax, which would credit
    the first label. A random valid label scores an invalid answer at chance
    for both. The count is kept in ``attrs['n_invalid_answers']``, and more
    than ``MAX_INVALID_ANSWER_FRACTION`` invalid answers raises instead.
    """
    import warnings
    import numpy as np

    if 'choice' not in assembly.dims or assembly.ndim != 2:
        return assembly
    if np.asarray(assembly.values).dtype.kind in ('U', 'S', 'O'):
        # A legacy answer of text labels or stimulus ids (label task, odd_one_out) has
        # no NaN row to resolve; it reaches the benchmark unchanged. A numeric answer of
        # any kind takes the path below.
        return assembly
    # A row of NaN marks an invalid answer: a reply that could not be read, or
    # a conversion of a source row that is not valid (for example a non-finite one).
    choice_axis = assembly.dims.index('choice')
    invalid = np.isnan(assembly.values).all(axis=choice_axis)
    if not invalid.any():
        return assembly
    fraction = invalid.mean()
    if fraction > MAX_INVALID_ANSWER_FRACTION:
        raise ValueError(
            f"{int(invalid.sum())} of {invalid.size} answers ({fraction:.0%}) are "
            f"outside the label set, above the {MAX_INVALID_ANSWER_FRACTION:.0%} "
            f"limit for legacy behavioral benchmarks. Check the answer function's "
            f"parsing; the replies are in the 'raw_answer' coordinate.")
    resolved = assembly.copy()
    values = np.moveaxis(resolved.values, choice_axis, -1)
    rng = np.random.RandomState(seed)
    for row in np.flatnonzero(invalid):
        values[row] = 0
        values[row, rng.randint(values.shape[-1])] = 1
    resolved.values = np.moveaxis(values, -1, choice_axis)
    resolved.attrs['n_invalid_answers'] = int(invalid.sum())
    warnings.warn(f"{int(invalid.sum())} invalid answers were scored as random valid "
                  f"labels (seed {seed}) for a legacy behavioral benchmark.")
    return resolved
