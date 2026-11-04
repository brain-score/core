"""Behavioral readout and generation support for BrainScoreModel."""

from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contract import TaskContext

import numpy as np


class BehavioralReadout:
    """Owns behavioral task state for a BrainScoreModel instance."""

    BEHAVIORAL_TASK_TYPES = frozenset({
        'probabilities', 'classification', 'label',
    })

    def __init__(self, owner) -> None:
        self.owner = owner
        self.task_context: Optional[TaskContext] = None
        self.readout_classifier: Any = None
        self.use_generation_for_task = False
        # The TaskContext object the current configuration came from, so a
        # repeated send of the same task keeps the fitted readout.
        self.configured_from: Optional[TaskContext] = None
        self._configured_as: Optional[set] = None  # fingerprints of the fitted configuration
        self._stale = False
        # Response tasks (TaskContext.response): how the answer and probabilities are produced.
        self.response_path: Optional[str] = None
        self.probability_source: Optional[str] = None

    @classmethod
    def requires_behavioral_readout(cls, task_type: str) -> bool:
        return task_type in cls.BEHAVIORAL_TASK_TYPES

    @staticmethod
    def _fingerprint(task_context: TaskContext) -> tuple:
        """Everything the fitted readout depends on: task, fitting rows and labels, generation settings."""
        fitting = task_context.fitting_stimuli
        rows = labels = None
        columns = getattr(fitting, 'columns', None)
        if columns is not None:
            try:
                if 'stimulus_id' in columns:
                    rows = tuple(fitting['stimulus_id'])
                for column in ('image_label', 'label'):
                    if column in columns:
                        labels = tuple(fitting[column])
                        break
            except (TypeError, KeyError, AttributeError):
                rows = labels = None  # not a table; identity alone must do
        label_set = task_context.label_set
        return (task_context.task_type, id(fitting), rows, labels, task_context.instruction,
                tuple(label_set) if label_set is not None else None,
                task_context.prefer_path or 'auto')

    def is_configured_for(self, task_context: TaskContext) -> bool:
        """Whether ``task_context`` is the task already configured, unchanged since.

        True for the object the configuration came from and for the context
        this component holds (the same task after input projection), as long
        as its task, fitting table, labels and generation settings are the
        ones fitted. TaskContext is mutable: a caller that swaps the fitting
        table or relabels it and calls start_task again gets a new fit.
        """
        if self._stale or self.task_context is None or not self._configured_as:
            return False
        if task_context is not self.configured_from and task_context is not self.task_context:
            return False
        return self._fingerprint(task_context) in self._configured_as

    def mark_configured(self, task_context: Optional[TaskContext]) -> None:
        """Remember the configuration just fitted, as the caller's context and as the one held."""
        self.configured_from = task_context
        self._configured_as = None
        if task_context is not None:
            self._configured_as = {self._fingerprint(task_context)}
            if self.task_context is not None and self.task_context is not task_context:
                self._configured_as.add(self._fingerprint(self.task_context))
        self._stale = False

    def invalidate(self) -> None:
        """The model changed (a state change was applied or undone): fit again next time."""
        self._stale = True

    def start_task(self, task_context_or_task, fitting_stimuli=None,
                   **kwargs) -> None:
        del kwargs
        owner = self.owner
        if isinstance(task_context_or_task, TaskContext):
            task_context = task_context_or_task
        else:
            task_context = TaskContext(
                task_type=task_context_or_task,
                fitting_stimuli=fitting_stimuli,
            )
        self.task_context = task_context

        task_type = task_context.task_type
        if task_context.readout_fit not in ('current', 'intact'):
            raise ValueError(f"readout_fit must be 'current' or 'intact', not {task_context.readout_fit!r}.")
        if task_context.response is not None:
            self._start_response_task(task_context)
            return
        if task_type != 'passive' and not owner._requires_behavioral_readout(task_type):
            raise ValueError(
                f"Unknown task_type {task_type!r}. BrainScoreModel accepts "
                f"'passive' or one of {sorted(self.BEHAVIORAL_TASK_TYPES)}; "
                f"answer and readout models both use 'probabilities'."
            )
        if not owner._requires_behavioral_readout(task_type):
            self.use_generation_for_task = False
            self.readout_classifier = None
            return

        instruction = task_context.instruction
        label_set = task_context.label_set
        fitting = task_context.fitting_stimuli
        prefer = task_context.prefer_path or 'auto'
        if prefer not in ('auto', 'generation', 'readout'):
            raise ValueError(
                f"TaskContext.prefer_path must be one of "
                f"('auto', 'generation', 'readout'); got {prefer!r}."
            )

        gen_viable = (owner._generation_fn is not None
                      and instruction and label_set)
        readout_viable = fitting is not None

        if prefer == 'generation':
            if not gen_viable:
                raise ValueError(
                    f"Model '{owner.identifier}' cannot run task "
                    f"'{task_type}' via generation path "
                    f"(prefer_path='generation'): "
                    f"requires an answer function registered with "
                    f"BrainScoreModel(answer=...) AND TaskContext.instruction "
                    f"AND TaskContext.label_set to be set."
                )
            self.use_generation_for_task = True
            self.readout_classifier = None
            return

        if prefer == 'readout':
            if not readout_viable:
                raise ValueError(
                    f"Model '{owner.identifier}' cannot run task "
                    f"'{task_type}' via readout path "
                    f"(prefer_path='readout'): "
                    f"requires TaskContext.fitting_stimuli to be set."
                )
            self.use_generation_for_task = False
            owner._fit_behavioral_readout(fitting)
            return

        if gen_viable:
            # A readout exists only with a layer to read and an extractor to run.
            readout_wired = (owner._behavioral_readout_layer is not None
                             and bool(owner._extractors or owner._layer_extractors))
            if readout_viable and readout_wired:
                import warnings
                warnings.warn(
                    'Both generation and fitted readout are available; '
                    "prefer_path='auto' selects generation and does not fit on "
                    'fitting_stimuli. Set TaskContext.prefer_path to '
                    "'generation' or 'readout' to choose explicitly.",
                    UserWarning, stacklevel=3)
            self.use_generation_for_task = True
            self.readout_classifier = None
            return

        self.use_generation_for_task = False
        if readout_viable:
            owner._fit_behavioral_readout(fitting)
        else:
            raise ValueError(
                f"Model '{owner.identifier}' cannot run behavioral task "
                f"'{task_type}': TaskContext provides neither fitting_stimuli "
                f"(needed for readout path) nor (instruction + label_set, "
                f"with BrainScoreModel(answer=...)) (needed for generation path)."
            )

    def reset(self) -> None:
        self.response_path = self.probability_source = None
        self.task_context = None
        self.readout_classifier = None
        self.use_generation_for_task = False
        self.configured_from = None
        self._configured_as = None
        self._stale = False

    def fit_behavioral_readout(self, fitting_stimuli) -> None:
        """Extract readout-layer features and fit a probabilities classifier."""
        from brainscore_core._model.behavior import ProbabilitiesClassifier

        owner = self.owner
        task_type = (
            self.task_context.task_type
            if self.task_context is not None else None
        )
        if owner._behavioral_readout_layer is None:
            raise ValueError(
                f"Model '{owner.identifier}' was asked to perform a behavioral "
                f"task (task_type={task_type!r}) but no "
                f"behavioral_readout_layer is set. Register the model with "
                f"BrainScoreModel(..., behavioral_readout_layer='<layer name>')."
            )

        labels = self.extract_labels(fitting_stimuli)
        intact = (self.task_context is not None and self.task_context.readout_fit == 'intact')
        with (owner._perturbations.intact() if intact else nullcontext()):
            features = owner._extract_behavioral_features(fitting_stimuli)

        from ..responses import Scale
        response = self.task_context.response if self.task_context is not None else None
        if isinstance(response, Scale):
            from brainscore_core._model.behavior import RatingRegressor
            self.readout_classifier = RatingRegressor().fit(features, labels)
        else:
            self.readout_classifier = ProbabilitiesClassifier()
            self.readout_classifier.fit(features, labels)

    @staticmethod
    def extract_labels(stimuli):
        for col in ('image_label', 'label'):
            if col in stimuli.columns:
                return list(stimuli[col].values)
        raise ValueError(
            f"Cannot find labels in fitting_stimuli. Expected one of "
            f"'image_label' or 'label' columns; got: {list(stimuli.columns)}."
        )

    def extract_behavioral_features(self, stimuli):
        """Run the model once at the behavioral readout layer."""
        owner = self.owner
        saved_layer = owner._recorder.recording_layer
        saved_layers = owner._recorder.recording_layers
        saved_regions = owner._recorder.recording_regions
        saved_multi = owner._recorder.is_multi_region
        saved_composite = owner._recorder.composite_recording
        saved_classifier = self.readout_classifier
        owner._recorder.recording_layer = owner._behavioral_readout_layer
        owner._recorder.recording_layers = (
            [owner._behavioral_readout_layer]
            if owner._behavioral_readout_layer else []
        )
        owner._recorder.recording_regions = []
        owner._recorder.is_multi_region = False
        owner._recorder.composite_recording = False
        self.readout_classifier = None
        try:
            features = owner.process(stimuli)
        finally:
            owner._recorder.recording_layer = saved_layer
            owner._recorder.recording_layers = saved_layers
            owner._recorder.recording_regions = saved_regions
            owner._recorder.is_multi_region = saved_multi
            owner._recorder.composite_recording = saved_composite
            self.readout_classifier = saved_classifier

        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        return features

    def predict_probabilities(self, stimuli):
        """Return a BehavioralAssembly of per-label probabilities."""
        features = self.owner._extract_behavioral_features(stimuli)
        return self.readout_classifier.predict_proba(features)

    def _start_response_task(self, task_context) -> None:
        """Choose how a Choice, Scale or Text answer is produced, before any compute."""
        from ..responses import Choice, Scale, Text
        owner, response = self.owner, task_context.response
        prefer = task_context.prefer_path or 'auto'
        if prefer not in ('auto', 'generation', 'likelihood', 'readout'):
            raise ValueError("prefer_path must be 'auto', 'generation', 'likelihood' or 'readout'; "
                             f'got {prefer!r}.')
        readout_wired = (owner._behavioral_readout_layer is not None
                         and bool(owner._extractors or owner._layer_extractors)
                         and task_context.fitting_stimuli is not None)
        reported_confidence = isinstance(response, Choice) and response.confidence is not None
        can = {
            'generation': owner._reply_fn is not None or (
                owner._generation_fn is not None and isinstance(response, Choice)
                and not reported_confidence),
            'likelihood': (owner._likelihood_fn is not None and isinstance(response, Choice)
                           and not reported_confidence),
            'readout': readout_wired and isinstance(response, (Choice, Scale)) and not reported_confidence,
        }
        order = ['generation', 'likelihood', 'readout'] if prefer == 'auto' else [prefer]
        path = next((p for p in order if can[p]), None)
        if path is None:
            needs = {'generation': 'reply= (or answer= for a plain Choice)',
                     'likelihood': 'likelihood= and a Choice without reported confidence',
                     'readout': 'behavioral_readout_layer=, an extractor and fitting_stimuli'}
            raise ValueError(
                f"Model '{owner.identifier}' cannot give a {type(response).__name__} answer"
                + (' with reported confidence (needs reply=)' if reported_confidence else '')
                + '. Paths tried: ' + '; '.join(f'{p} needs {needs[p]}' for p in order))
        source = None
        if isinstance(response, Choice) and response.probabilities:
            if path != 'readout' and owner._likelihood_fn is not None:
                source = 'likelihood'
            elif readout_wired:
                source = 'readout'
            else:
                raise ValueError(f"Model '{owner.identifier}' cannot give implied probabilities: "
                                 'it needs likelihood= or a readout (behavioral_readout_layer=, '
                                 'fitting_stimuli).')
        self.response_path, self.probability_source = path, source
        self.use_generation_for_task = False   # response tasks run through sessions only
        self.readout_classifier = None
        if 'readout' in (path, source):
            owner._fit_behavioral_readout(task_context.fitting_stimuli)

    def respond(self, stimuli, channels, regions=()):
        """Answer a response task; return ``({channel: assembly}, {region: NeuroidAssembly})``.

        Each stimulus is presented once on the answer's path, and regions are
        recorded during that presentation: the prompt pass of a reply or of
        likelihood scoring, or the readout's own extraction. Implied
        probabilities next to a reply are scored first, from the same prompt,
        so the recording is still the prompt pass.
        """
        from ..responses import Choice, Scale, Text, labels_of, trial_prompt
        owner, task = self.owner, self.task_context
        response, path = task.response, self.response_path
        layers = {region: self._region_layers(region) for region in regions}
        labels = labels_of(response)
        # Every trial's prompt is built first, so a bad layout fails before any compute.
        prompts = [trial_prompt(task, row) for _, row in stimuli.iterrows()]
        neural, outputs = {}, {}

        if path == 'readout':
            values, neural = self._read_out_and_record(stimuli, layers, raw=True)
            if isinstance(response, Scale):
                outputs['behavior:rating'] = _numbers(values, stimuli, 'readout')
            else:
                outputs['behavior:choice'] = _one_hot(
                    [str(values['choice'].values[i]) for i in np.asarray(values.values).argmax(1)],
                    [True] * len(stimuli), stimuli, labels, 'readout', raw=None)
                if self.probability_source == 'readout':
                    outputs['behavior:probabilities'] = _implied(values, 'readout')
        else:
            score_first = path == 'generation' and self.probability_source == 'likelihood'
            if path == 'likelihood':
                def call(row, prompt):
                    return owner._likelihood_fn(prompt, list(labels))
            elif owner._reply_fn is not None:
                def call(row, prompt):
                    # Scoring first keeps the recorded first pass the prompt alone.
                    scores = owner._likelihood_fn(prompt, list(labels)) if score_first else None
                    return owner._reply_fn(prompt), scores
            else:   # a registration's answer: it phrases and parses a Choice itself
                def call(row, prompt):
                    return owner._generation_fn(stimulus_row=row, instruction=task.instruction,
                                                label_set=list(labels))
            replies, neural = self._per_stimulus(stimuli, layers, call, prompts)
            scores = None
            if path == 'generation' and owner._reply_fn is not None:
                scores = [s for _, s in replies]
                replies = [r for r, _ in replies]
            if path == 'likelihood':
                probabilities = _softmax_rows(replies, labels)
                chosen = [labels[i] for i in probabilities.argmax(1)]
                outputs['behavior:choice'] = _one_hot(chosen, [True] * len(stimuli), stimuli,
                                                      labels, 'likelihood', raw=None)
                if self.probability_source == 'likelihood':
                    outputs['behavior:probabilities'] = _implied_array(
                        probabilities, stimuli, labels, 'likelihood')
            elif isinstance(response, Text):
                parsed = [response.parse(reply) for reply in replies]
                outputs['behavior:text'] = _texts(parsed, replies, stimuli)
            elif isinstance(response, Scale):
                parsed = [response.parse(reply) for reply in replies]
                outputs['behavior:rating'] = _numbers(parsed, stimuli, 'reply', raw=replies)
            else:
                parsed = [response.parse(str(reply)) for reply in replies]
                source = 'reply' if owner._reply_fn is not None else 'answer'
                outputs['behavior:choice'] = _one_hot(
                    [label for label, _ in parsed], [label is not None for label, _ in parsed],
                    stimuli, labels, source, raw=[str(r) for r in replies],
                    errors=[error for _, error in parsed])
                if response.confidence is not None:
                    outputs['behavior:confidence'] = _numbers(
                        [response.parse_confidence(reply) for reply in replies], stimuli,
                        'reply', raw=replies)
                if score_first:
                    outputs['behavior:probabilities'] = _implied_array(
                        _softmax_rows(scores, labels), stimuli, labels, 'likelihood')
                elif self.probability_source == 'readout':
                    outputs['behavior:probabilities'] = _implied(self.readout_classifier.predict_proba(
                        owner._extract_behavioral_features(stimuli)), 'readout')
        outputs['behavior'] = outputs[response.channels[0]]
        if prompts:
            # A multi-image trial records one step per image: name each by its part.
            image_names = [name for (kind, _), name in zip(prompts[0].parts, prompts[0].names)
                           if kind == 'image']
            neural = {region: (assembly.assign_coords(part_name=('part', image_names))
                               if 'part' in assembly.dims and assembly.sizes['part'] == len(image_names)
                               else assembly)
                      for region, assembly in neural.items()}
        stamp = self._stamp(path, 'readout extraction' if path == 'readout' else 'prompt')
        return ({c: outputs[c] for c in channels},
                {r: a.assign_attrs(recording=stamp) for r, a in neural.items()})

    def _open_recordings(self, stimuli, layers):
        """One live recording per input type, before any compute.

        An extractor's own ``record(layers)`` when it has one; otherwise forward
        hooks on the model passed as ``BrainScoreModel(model=...)``, with layer
        paths read from that model.
        """
        owner = self.owner
        from .dispatch import extractor_for
        by_modality: Dict[str, list] = {}
        for region, region_layers in layers.items():
            by_modality.setdefault(self._region_modality(region), []).extend(region_layers)
        recordings = {}
        for modality, modality_layers in by_modality.items():
            modality_layers = list(dict.fromkeys(modality_layers))
            extractor = extractor_for(owner, modality, stimuli)
            if extractor is None:
                raise NotImplementedError(f'No {modality!r} extractor to record from.')
            try:
                record = extractor.record
            except AttributeError:
                record = None
            if record is not None:
                recordings[modality] = record(modality_layers)
            elif owner._model is not None:
                recordings[modality] = _hook_recording(owner._model, modality_layers, owner.identifier)
            else:
                raise NotImplementedError(
                    f'The {modality!r} extractor cannot record while the model answers: give it '
                    'a record(layers) method, or pass the network as BrainScoreModel(model=...).')
        return recordings

    def _per_stimulus(self, stimuli, layers, call, prompts=None):
        """Call ``call(row, prompt)`` once per stimulus, recording ``layers`` on each first pass."""
        recordings = self._open_recordings(stimuli, layers)
        replies, trials = [], []
        with ExitStack() as stack:
            live = {m: stack.enter_context(r) for m, r in recordings.items()}
            for index, (_, row) in enumerate(stimuli.iterrows()):
                replies.append(call(row, None if prompts is None else prompts[index]))
                trials.append({m: recording.take() for m, recording in live.items()})
        return replies, self._trial_regions(trials, stimuli, layers)

    def _trial_regions(self, trials, stimuli, layers):
        """Per-trial recordings -> one NeuroidAssembly per region (composite regions gathered)."""
        import xarray as xr
        neural = {}
        for region, region_layers in layers.items():
            modality = self._region_modality(region)
            per_layer = [_trials_to_assembly([trial[modality] for trial in trials], stimuli, layer, region)
                         for layer in region_layers]
            neural[region] = self._gather(region, per_layer)
        return neural

    def _gather(self, region, per_layer):
        """A region's units: one layer as is; a composite region's units gathered from its layers."""
        import xarray as xr
        from ..selection import CompositeSelector
        selector = self.owner._region_layer_selectors.get(region)
        if not isinstance(selector, CompositeSelector):
            return per_layer[0]
        recorder = self.owner._recorder
        combined = xr.concat([recorder.ensure_unit_indices(a) for a in per_layer], dim='neuroid')
        return recorder.gather_composite_units(combined, selector)

    @contextmanager
    def suspended(self):
        """Keep the configured task, but let it answer nothing for a while.

        A session that asks only for ``neural:<region>`` must get activity, even
        when an earlier session configured a task on this model (rule 4.1).
        """
        saved = self.task_context
        self.task_context = None
        try:
            yield
        finally:
            self.task_context = saved

    def respond_and_record(self, stimuli, regions: Sequence[str]):
        """Answer every stimulus once and record ``regions`` from that same pass.

        Returns ``(BehavioralAssembly, {region: NeuroidAssembly})``. With
        ``answer``, the recording is the first forward pass of each answer (the
        prompt, before any generated token). With a readout, the region layers
        are extracted in the same call as the readout layer.
        """
        layers = {region: self._region_layers(region) for region in regions}
        if self.use_generation_for_task:
            return self._generate_and_record(stimuli, layers)
        return self._read_out_and_record(stimuli, layers)

    def _region_layers(self, region: str) -> List[str]:
        from ..selection import CompositeSelector
        selector = self.owner._region_layer_selectors.get(region)
        if selector is None:
            raise ValueError(f'Region {region!r} is not in region_layer_map.')
        if isinstance(selector, CompositeSelector):
            return list(selector.layer_paths)
        return [selector.layer_path]

    def _region_modality(self, region: str) -> str:
        from ..io_catalog import canonical_modality
        modality = self.owner._region_modality_map.get(region)
        if modality is not None:
            return canonical_modality(modality)
        modalities = {canonical_modality(key) for key in self.owner._extractors}
        if len(modalities) != 1:
            raise ValueError(
                f'Region {region!r} does not say which input it reads. Map it with '
                "LayerSelector(name=..., modality=...) to record it during a task.")
        return next(iter(modalities))

    def _generate_and_record(self, stimuli, layers):
        recordings = self._open_recordings(stimuli, layers)
        trials = []
        with ExitStack() as stack:
            live = {m: stack.enter_context(r) for m, r in recordings.items()}
            behavior = self.generate_predictions(
                stimuli, after_each=lambda: trials.append(
                    {m: recording.take() for m, recording in live.items()}))
        stamp = self._stamp('answer', 'prompt')
        return behavior, {region: assembly.assign_attrs(recording=stamp)
                          for region, assembly in self._trial_regions(trials, stimuli, layers).items()}

    def _stamp(self, path, forward_pass):
        """How a recording was made, so task and passive recordings stay distinct."""
        return {'task_type': self.task_context.task_type, 'path': path,
                'response': type(self.task_context.response).__name__
                if self.task_context.response is not None else None,
                'forward_pass': forward_pass, 'readout_fit': self.task_context.readout_fit}

    def _read_out_and_record(self, stimuli, layers, raw: bool = False):
        from ..responses import Scale
        owner = self.owner
        readout_layer = owner._behavioral_readout_layer
        modality = owner._pick_modality(owner._detect_modalities(stimuli))
        for region in layers:
            if owner._region_modality_map.get(region) and self._region_modality(region) != modality:
                raise ValueError(
                    f'Region {region!r} reads {self._region_modality(region)!r} input, but '
                    f'the readout presents {modality!r} input; one presentation cannot '
                    'record it.')
        all_layers = list(dict.fromkeys([readout_layer, *(l for ls in layers.values() for l in ls)]))
        extracted = owner._recorder.ensure_layer_provenance(
            owner._extract_for_modality(stimuli, modality, all_layers), all_layers,
            description='Readout extraction')
        layer_of = extracted['layer'].values
        features = extracted.isel(neuroid=layer_of == readout_layer)
        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        if raw and isinstance(self.task_context.response, Scale):
            behavior = [(float(v), None) for v in self.readout_classifier.predict(features)]
        else:
            behavior = self.readout_classifier.predict_proba(features)
        neural = {}
        for region, region_layers in layers.items():
            per_layer = [extracted.isel(neuroid=layer_of == layer) for layer in region_layers]
            selected = self._gather(region, per_layer)
            neural[region] = selected.assign_coords(
                region=('neuroid', [region] * selected.sizes['neuroid']))
            if not raw:
                neural[region] = neural[region].assign_attrs(
                    recording=self._stamp('readout', 'readout extraction'))
        return behavior, neural

    def generate_predictions(self, stimuli, after_each: Optional[Callable[[], None]] = None):
        """Call the registered answer function per stimulus, return one-hot BehavioralAssembly.

        ``after_each`` runs after every answer, before the next stimulus; the
        joint behavior-and-neural session uses it to close each recording.
        """
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly,
        )

        owner = self.owner
        task_context = self.task_context
        if task_context is None:
            raise RuntimeError(
                "generate_predictions requires an active TaskContext."
            )
        label_set = list(task_context.label_set)
        instruction = task_context.instruction
        label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}

        n_stimuli = len(stimuli)
        n_labels = len(label_set)
        proba = np.zeros((n_stimuli, n_labels), dtype=np.float32)

        raw_answers, valid = [], []
        for i, (_, row) in enumerate(stimuli.iterrows()):
            predicted = owner._generation_fn(
                stimulus_row=row,
                instruction=instruction,
                label_set=label_set,
            )
            if after_each is not None:
                after_each()
            raw_answers.append(str(predicted))
            valid.append(predicted in label_to_idx)
            if valid[-1]:
                proba[i, label_to_idx[predicted]] = 1.0
            else:
                # No label is credited: a substituted label would bias accuracy toward it.
                proba[i, :] = np.nan
                import warnings
                warnings.warn(
                    f"answer function returned {predicted!r}, not in "
                    f"label_set={label_set}. The response is recorded as invalid "
                    f"(NaN probabilities, answer_valid=False)."
                )

        stimulus_ids = list(stimuli['stimulus_id'].values)
        presentation_coords = {
            'stimulus_id': ('presentation', stimulus_ids),
        }
        for column in stimuli.columns:
            if column == 'stimulus_id':
                continue
            presentation_coords[column] = (
                'presentation', list(stimuli[column].values))
        presentation_coords['raw_answer'] = ('presentation', raw_answers)
        presentation_coords['answer_valid'] = ('presentation', valid)

        return BehavioralAssembly(
            proba,
            coords={
                **presentation_coords,
                'choice': label_set,
            },
            dims=['presentation', 'choice'],
        )


@contextmanager
def _hook_recording(model, layers, identifier):
    """Forward hooks on ``model`` for extractors without ``record()``.

    ``take()`` returns the first pass since the previous take, each layer's
    output flattened per presentation, as a still-image extractor packages it.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    def module_at(path):
        module = model
        for part in path.split('.'):
            module = module[int(part)] if part.isdigit() else getattr(module, part, None)
            if module is None:
                raise NotImplementedError(f'Layer {path!r} is not a module of the model passed as model=; '
                                          'give the extractor a record(layers) method.')
        return module

    modules = {layer: module_at(layer) for layer in layers}   # unknown paths fail before compute
    captured = {}

    def keep(layer):
        def hook(_module, _inputs, output):
            if layer not in captured:
                output = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer] = np.asarray(output.detach().float().cpu().numpy())
        return hook

    handles = [module.register_forward_hook(keep(layer)) for layer, module in modules.items()]

    class Recording:
        @staticmethod
        def take():
            missing = [layer for layer in layers if layer not in captured]
            if missing:
                raise ValueError(f'No forward pass reached {missing} since the last presentation.')
            rows = {layer: captured[layer].reshape(1, -1) for layer in layers}
            captured.clear()
            values = np.concatenate([rows[layer] for layer in layers], axis=1)
            neuroid_layers = [layer for layer in layers for _ in range(rows[layer].shape[1])]
            numbers = [i for layer in layers for i in range(rows[layer].shape[1])]
            return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', ['?']),
                'neuroid_id': ('neuroid', [f'{identifier}.{l}.{i}' for l, i in zip(neuroid_layers, numbers)]),
                'neuroid_num': ('neuroid', numbers), 'layer': ('neuroid', neuroid_layers)})
    try:
        yield Recording()
    finally:
        for handle in handles:
            handle.remove()


def _presentation(stimuli):
    return {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns}


def _one_hot(chosen, valid, stimuli, labels, source, raw=None, errors=None):
    """One row per stimulus, 1 for the chosen label; invalid rows are NaN (rule 6.2)."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.zeros((len(stimuli), len(labels)), dtype=np.float32)
    for i, (label, ok) in enumerate(zip(chosen, valid)):
        if ok:
            values[i, list(labels).index(label)] = 1.0
        else:
            values[i] = np.nan
    coords = {**_presentation(stimuli), 'choice': list(labels),
              'answer_valid': ('presentation', list(valid)),
              'raw_answer': ('presentation', [str(r) for r in (raw or chosen)])}
    if errors is not None:
        coords['parse_error'] = ('presentation', [e or '' for e in errors])
    return BehavioralAssembly(values, coords=coords, dims=['presentation', 'choice'],
                              attrs={'measure': {'kind': 'reported' if source in ('reply', 'answer')
                                                 else 'implied', 'source': source}})


def _numbers(parsed, stimuli, source, raw=None):
    """One number per stimulus (a rating or a confidence); invalid ones are NaN."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([np.nan if v is None else v for v, _ in parsed], dtype=float)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed]),
              'parse_error': ('presentation', [e or '' for _, e in parsed])}
    if raw is not None:
        coords['raw_answer'] = ('presentation', [str(r) for r in raw])
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'measure': {'kind': 'reported' if source == 'reply'
                                                 else 'implied', 'source': source}})


def _texts(parsed, replies, stimuli):
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.array([v if v is not None else '' for v, _ in parsed], dtype=object)
    coords = {**_presentation(stimuli),
              'answer_valid': ('presentation', [v is not None for v, _ in parsed])}
    return BehavioralAssembly(values, coords=coords, dims=['presentation'],
                              attrs={'measure': {'kind': 'reported', 'source': 'reply'}})


def _softmax_rows(scores, labels):
    logp = np.array([[float(row[label]) for label in labels] for row in scores], dtype=float)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def _implied_array(probabilities, stimuli, labels, source):
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    return BehavioralAssembly(probabilities.astype(np.float32),
                              coords={**_presentation(stimuli), 'choice': list(labels)},
                              dims=['presentation', 'choice'],
                              attrs={'measure': {'kind': 'implied', 'source': source}})


def _implied(assembly, source):
    return assembly.assign_attrs(measure={'kind': 'implied', 'source': source})


def _trials_to_assembly(trials, stimuli, layer: str, region: str):
    """Stack one-presentation recordings into a NeuroidAssembly for ``region``."""
    import numpy as np
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    rows = []
    # A clip keeps its time steps and a multi-image trial its parts: one extra dimension.
    extra = [dim for dim in trials[0].dims if dim not in ('presentation', 'neuroid')]
    if len(extra) > 1:
        raise ValueError(f'A recording during a task may have one extra dimension; got {extra}.')
    timed = bool(extra)
    extra = extra[0] if extra else None
    for index, trial in enumerate(trials):
        selected = trial.isel(neuroid=trial['layer'].values == layer)
        if selected.sizes['neuroid'] == 0:
            raise ValueError(
                f'No activity reached layer {layer!r} while answering stimulus '
                f'{stimuli["stimulus_id"].iloc[index]!r}; that input may not use it.')
        if timed:
            if selected.sizes[extra] != trials[0].sizes[extra]:
                raise ValueError(f'Trials gave different numbers of {extra} steps; present trials of one '
                                 'length (clips of one duration, sequences of one length) together.')
            rows.append(np.asarray(selected.transpose('presentation', extra, 'neuroid').values)[0])
        else:
            rows.append(np.asarray(selected.values).reshape(-1))
    first = trials[0].isel(neuroid=trials[0]['layer'].values == layer)
    index = first.indexes.get('neuroid')
    names = list(index.names) if isinstance(index, pd.MultiIndex) else []
    names += [name for name, coord in first.coords.items()
              if coord.dims == ('neuroid',) and name != 'neuroid' and name not in names]
    coords = {name: ('neuroid', np.asarray(first[name].values)) for name in names}
    coords['region'] = ('neuroid', [region] * first.sizes['neuroid'])
    for column in stimuli.columns:
        coords[column] = ('presentation', list(stimuli[column].values))
    if not timed:
        return NeuroidAssembly(np.stack(rows), coords=coords, dims=['presentation', 'neuroid'])
    extra_index = first.indexes.get(extra)
    extra_names = list(extra_index.names) if isinstance(extra_index, pd.MultiIndex) else []
    extra_names += [name for name, coord in first.coords.items()
                    if coord.dims == (extra,) and name != extra and name not in extra_names]
    for name in extra_names:
        coords[name] = (extra, np.asarray(first[name].values))
    return NeuroidAssembly(np.stack(rows), coords=coords,
                           dims=['presentation', extra, 'neuroid'])


# Keep existing pickle references and public type names stable.
for _definition in (
    BehavioralReadout,
):
    _definition.__module__ = 'brainscore_core.behavioral'
del _definition
