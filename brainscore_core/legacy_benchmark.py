"""Permanent legacy benchmark methods over the session contract.

The facade owns measurement settings. A subject only sees sessions, regardless
of which benchmark API requested the measurement.
"""

from dataclasses import replace
import hashlib
from typing import Any, Optional, List, Tuple
from weakref import ref

from .contract import Subject, TaskContext
from .streaming import BufferedSession
from .supported_data_standards.brainio.stimuli import StimulusSet


_visual_fields = {}


def register_visual_field(subject: Subject, visual_degrees) -> None:
    """Register a field-of-view provider without extending Subject's contract."""
    key = id(subject)
    _visual_fields[key] = (
        ref(subject, lambda unused: _visual_fields.pop(key, None)), visual_degrees)


def mark_legacy_benchmark(benchmark):
    """Declare the calling convention at a domain registry's loading boundary."""
    if getattr(benchmark, 'subject_api', None) is None:
        benchmark.subject_api = (
            'session' if getattr(benchmark, 'required_input_channels', None) is not None
            else 'legacy')
    return benchmark


def subject_for_benchmark(subject: Subject, benchmark):
    """Apply the facade at the legacy registry boundary, never by model class."""
    if getattr(benchmark, 'subject_api', None) == 'legacy':
        registered = _visual_fields.get(id(subject))
        provider = (registered[1] if registered is not None and registered[0]() is subject
                    else lambda: 8)
        return LegacyBenchmarkFacade(subject, visual_degrees=provider())
    return subject


class LegacyBenchmarkFacade:
    """Present a Subject to vision and language benchmarks without changing them."""

    def __init__(self, subject: Subject, *, visual_degrees: float = 8) -> None:
        self.subject = subject
        self._visual_degrees = visual_degrees
        self._record: str | None = None
        self._time_bins = None
        self._task: TaskContext | None = None
        self._recording_type = None

    @property
    def identifier(self) -> str:
        return self.subject.identifier

    def visual_degrees(self) -> float:
        """Return the field of view configured by the model registration."""
        return self._visual_degrees

    def start_recording(self, recording_target: str, time_bins=None) -> None:
        self._record = recording_target
        self._time_bins = time_bins

    def start_neural_recording(self, recording_target: str,
                               recording_type: str = 'fMRI') -> None:
        self.start_recording(recording_target)
        self._recording_type = recording_type

    def start_task(self, task, fitting_stimuli=None, **settings) -> None:
        self._task = (task if isinstance(task, TaskContext) else TaskContext(
            task_type=task, fitting_stimuli=fitting_stimuli, metadata=settings))

    def start_behavioral_task(self, task: str) -> None:
        self.start_task(task)

    def _measure(self, stimuli, channel: str, *, number_of_trials: int = 1):
        return self._measure_channels(stimuli, (channel,), number_of_trials=number_of_trials)[channel]

    def _measure_channels(self, stimuli, channels: tuple[str, ...], *, number_of_trials: int = 1,
                          legacy_text=None):
        if isinstance(number_of_trials, bool) or not isinstance(number_of_trials, int) or number_of_trials < 1:
            raise ValueError('number_of_trials must be a positive integer')
        context = self._task
        settings = {'number_of_trials': number_of_trials}
        if legacy_text is not None:
            settings['legacy_text'] = legacy_text
        if self._recording_type is not None:
            settings['recording_type'] = self._recording_type
        if context is not None:
            context = replace(context, metadata={**context.metadata, **settings,
                                                 'stimulus_set': stimuli})
        else:
            context = TaskContext(task_type='passive', metadata=settings)
        session = BufferedSession(
            stimulus_set=stimuli, requested_output_channels=channels,
            task_context=context, time_bins=self._time_bins,
        )
        self.subject.interact(session)
        return {channel: session.collect(channel) for channel in channels}

    def look_at(self, stimuli, number_of_trials: int = 1, **settings):
        """Present images using the measurement the benchmark configured."""
        if settings:
            raise TypeError(f'Unsupported look_at settings: {sorted(settings)}')
        behavior = self._task is not None and self._task.task_type != 'passive'
        channel = ('behavior' if behavior else f'neural:{self._record}')
        if not behavior and self._record is None:
            raise ValueError('Configure recording or a task before look_at')
        return self._measure(stimuli, channel, number_of_trials=number_of_trials)

    def digest_text(self, text) -> dict[str, Any]:
        """Return the keys requested by the legacy language benchmark."""
        stimuli = text
        if not isinstance(text, StimulusSet):
            parts = [text] if isinstance(text, str) else list(text)
            if not all(isinstance(part, str) for part in parts):
                raise TypeError('Text must contain strings')
            # Length prefixes keep different segmentations from sharing a cache key.
            encoded = ''.join(f'{len(part)}:{part}' for part in parts).encode()
            digest = hashlib.sha256(encoded).hexdigest()[:16]
            stimuli = StimulusSet({
                'sentence': parts, 'stimulus_id': list(range(len(parts))),
                'part_number': list(range(len(parts))),
                'context_id': [digest] * len(parts),
            })
            stimuli.identifier = f'legacy-text-{digest}'
        outputs = {}
        if self._record is not None:
            outputs['neural'] = f'neural:{self._record}'
        if self._task is not None and self._task.task_type != 'passive':
            outputs['behavior'] = 'behavior'
        if not outputs:
            raise ValueError('Configure neural recording or a behavioral task before digest_text')
        measured = self._measure_channels(
            stimuli, tuple(outputs.values()),
            legacy_text=None if isinstance(text, StimulusSet) else text,
        )
        return {name: measured[channel] for name, channel in outputs.items()}

    def reset(self) -> None:
        self.subject.reset()
        self._record = self._task = self._time_bins = self._recording_type = None


class LegacyModelMethods:
    """Forward old BrainScoreModel spellings through the permanent facade.

    This implementation mixin preserves calls on existing model registrations;
    session-native benchmarks never need it.
    """

    def _legacy_benchmark_view(self):
        view = LegacyBenchmarkFacade(self, visual_degrees=self._visual_degrees_val)
        regions = self._recorder.recording_regions
        if len(regions) > 1:
            raise ValueError('Legacy methods request one region; use a session for multiple outputs')
        if regions:
            view.start_recording(regions[0], self._recorder.time_bins)
        view._task = self._behavioral.task_context
        return view

    def look_at(self, stimuli, number_of_trials=1, **settings):
        return self._legacy_benchmark_view().look_at(
            stimuli, number_of_trials=number_of_trials, **settings)

    def digest_text(self, text):
        return self._legacy_benchmark_view().digest_text(text)

    def start_behavioral_task(self, task):
        self.start_task(task)

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        self.start_recording(recording_target, recording_type=recording_type)

    def visual_degrees(self):
        return self._visual_degrees_val

    @property
    def _recording_layer(self) -> Optional[str]:
        return self._recorder.recording_layer

    @_recording_layer.setter
    def _recording_layer(self, value: Optional[str]) -> None:
        self._recorder.recording_layer = value

    @property
    def _recording_regions(self) -> List[str]:
        return self._recorder.recording_regions

    @_recording_regions.setter
    def _recording_regions(self, value: List[str]) -> None:
        self._recorder.recording_regions = value

    @property
    def _recording_layers(self) -> List[str]:
        return self._recorder.recording_layers

    @_recording_layers.setter
    def _recording_layers(self, value: List[str]) -> None:
        self._recorder.recording_layers = value

    @property
    def _is_multi_region(self) -> bool:
        return self._recorder.is_multi_region

    @_is_multi_region.setter
    def _is_multi_region(self, value: bool) -> None:
        self._recorder.is_multi_region = value

    @property
    def _time_bins(self) -> Optional[List[Tuple[int, int]]]:
        return self._recorder.time_bins

    @_time_bins.setter
    def _time_bins(self, value: Optional[List[Tuple[int, int]]]) -> None:
        self._recorder.time_bins = value

    @property
    def _composite_recording(self) -> bool:
        return self._recorder.composite_recording

    @_composite_recording.setter
    def _composite_recording(self, value: bool) -> None:
        self._recorder.composite_recording = value

    @property
    def _task_context(self) -> Optional[TaskContext]:
        return self._behavioral.task_context

    @_task_context.setter
    def _task_context(self, value: Optional[TaskContext]) -> None:
        self._behavioral.task_context = value

    @property
    def _readout_classifier(self):
        return self._behavioral.readout_classifier

    @_readout_classifier.setter
    def _readout_classifier(self, value) -> None:
        self._behavioral.readout_classifier = value

    @property
    def _use_generation_for_task(self) -> bool:
        return self._behavioral.use_generation_for_task

    @_use_generation_for_task.setter
    def _use_generation_for_task(self, value: bool) -> None:
        self._behavioral.use_generation_for_task = value

    @property
    def _active_perturbations(self):
        return self._perturbations.active_perturbations

    @_active_perturbations.setter
    def _active_perturbations(self, value) -> None:
        self._perturbations.active_perturbations = value

