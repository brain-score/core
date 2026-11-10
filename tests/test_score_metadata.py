import numpy as np
import pytest

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.metrics import Score
from brainscore_core.score_metadata import (
    infer_score_protocol,
    record_measures,
    requested_output_channels_for_score,
    stamp_score_metadata,
)
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import behavioral_response
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from tests.test_behavioral_readout import (
    _make_image_stimulus_set,
)


class _Subject:
    identifier = "subject"
    in_channels = {"vision", "text"}


class _Benchmark:
    def __init__(self, *, region=None, parent=None, requested_output_channels=None):
        if region is not None:
            self.region = region
        if parent is not None:
            self.parent = parent
        if requested_output_channels is not None:
            self.requested_output_channels = requested_output_channels


class _AbstractParentBenchmark:
    @property
    def parent(self):
        raise NotImplementedError


def test_stamp_score_metadata_adds_provenance_and_preserves_existing_attrs():
    score = Score(1)
    score.attrs["runtime_sec"] = 0.01

    returned = stamp_score_metadata(
        score,
        _Subject(),
        requested_channels={"neural:IT"},
        protocol="neural",
        harness_id="test-harness",
    )

    assert returned is score
    assert score.attrs["runtime_sec"] == 0.01
    assert score.attrs["in_channels"] == ("text", "vision")
    assert score.attrs["out_channels"] == ("neural:IT",)
    assert score.attrs["protocol"] == "neural"
    assert score.attrs["subject_id"] == "subject"
    assert score.attrs["harness_id"] == "test-harness"


def test_requested_output_channels_use_explicit_or_region_fallback():
    assert requested_output_channels_for_score(
        _Benchmark(requested_output_channels={"behavior"})
    ) == {"behavior"}
    assert requested_output_channels_for_score(_Benchmark(region="IT")) == {
        "neural:IT"
    }


def test_requested_output_channels_infer_behavioral_parent():
    assert requested_output_channels_for_score(
        _Benchmark(parent="behavioral")
    ) == {"behavior"}


def test_parent_lookup_is_best_effort_for_legacy_benchmarks():
    benchmark = _AbstractParentBenchmark()

    assert requested_output_channels_for_score(benchmark) == set()
    assert infer_score_protocol(_Subject(), benchmark, set()) is None


def test_infer_score_protocol_from_requested_channels():
    assert infer_score_protocol(_Subject(), _Benchmark(), {"neural:V4"}) == "neural"
    assert infer_score_protocol(_Subject(), _Benchmark(), {"motor"}) == "motor"
    assert infer_score_protocol(
        _Subject(), _Benchmark(), {"perturbation"}
    ) == "perturbation"


# --- behavioral protocol from the measures the responses state ------------

class _Answerer(Subject):
    """A participate-only subject (not a BrainScoreModel) with a fixed measure."""

    identifier = 'answerer'
    in_channels = {'vision'}
    out_channels = {'behavior'}
    required_channels = set()

    def __init__(self, measure):
        self.measure = measure

    def participate(self, session):
        while session.next_input() is not None:
            pass
        stimuli = (session.stimulus_set if getattr(session, 'stimulus_set', None) is not None
                   else session.task_context.metadata['stimulus_set'])
        n = len(stimuli)
        payload = BehavioralAssembly(
            np.tile([1.0, 0.0], (n, 1)),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs={'measure': self.measure})
        for channel in session.requested_output_channels:
            session.respond(StreamEvent(channel, payload, None))

    def reset(self):
        pass


def _protocol(subject, measures):
    return infer_score_protocol(subject, _Benchmark(), {'behavior'}, measures=measures)


def _stimuli():
    return _make_image_stimulus_set(['cat'] * 3 + ['dog'] * 3, identifier='measure-test')


def test_generation_subject_without_brainscore_model_is_stamped_generation():
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       instruction='Which animal?', metadata={'stimulus_set': _stimuli()})
    with record_measures() as measures:
        behavioral_response(subject, task)

    score = stamp_score_metadata(Score(0.5), subject, {'behavior'},
                                 protocol=_protocol(subject, measures),
                                 harness_id='test', measures=measures)

    assert score.attrs['protocol'] == 'behavior:generation'
    assert score.attrs['measure'] == 'reported'


def test_likelihood_subject_is_not_stamped_readout():
    subject = _Answerer({'kind': 'implied', 'source': 'likelihood'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       metadata={'stimulus_set': _stimuli()})
    with record_measures() as measures:
        behavioral_response(subject, task)

    assert _protocol(subject, measures) == 'behavior:likelihood'


def test_implied_measure_without_a_known_source_does_not_claim_readout():
    subject = _Answerer({'kind': 'implied'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       metadata={'stimulus_set': _stimuli()})
    with record_measures() as measures:
        behavioral_response(subject, task)

    assert _protocol(subject, measures) == 'behavior:implied'


def test_legacy_facade_behavioral_benchmark_gets_measure_attr():
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})

    def benchmark(candidate):          # a legacy behavioral benchmark
        candidate.start_task('probabilities', fitting_stimuli=_stimuli())
        candidate.look_at(_stimuli())
        return Score(0.5)

    with record_measures() as measures:
        score = benchmark(LegacyBenchmarkFacade(subject, visual_degrees=8))
    stamp_score_metadata(score, subject, {'behavior'},
                         protocol=_protocol(subject, measures),
                         harness_id='test', measures=measures)

    assert score.attrs['measure'] == 'reported'
    assert score.attrs['protocol'] == 'behavior:generation'


def test_reported_and_implied_measures_stamp_a_sorted_tuple():
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})
    with record_measures() as measures:
        for measure in ({'kind': 'reported', 'source': 'reply'},
                        {'kind': 'implied', 'source': 'likelihood'}):
            subject.measure = measure
            behavioral_response(subject, TaskContext(
                task_type='probabilities', label_set=['cat', 'dog'],
                metadata={'stimulus_set': _stimuli()}))
    score = stamp_score_metadata(Score(0.5), subject, {'behavior'},
                                 protocol=_protocol(subject, measures),
                                 harness_id='test', measures=measures)

    assert score.attrs['measure'] == ('implied', 'reported')
    assert score.attrs['protocol'] == 'behavior:generation'


def test_nothing_recorded_keeps_the_old_default_stamp():
    score = stamp_score_metadata(Score(0.5), _Subject(), {'behavior'},
                                 protocol=infer_score_protocol(_Subject(), _Benchmark(),
                                                               {'behavior'}),
                                 harness_id='test')

    assert score.attrs['protocol'] == 'behavior:readout'
    assert 'measure' not in score.attrs


def test_measures_outside_a_run_are_not_recorded():
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       metadata={'stimulus_set': _stimuli()})
    behavioral_response(subject, task)            # no run open: nothing kept
    with record_measures() as measures:
        pass

    assert measures.measures == set()


_GENERATION_TASK = dict(task_type='probabilities', label_set=['cat', 'dog'],
                        instruction='What animal is this?')


# --- direct sessions, nesting, mixed collection --------------------------


@pytest.mark.parametrize('kind, expected', [('reply', 'behavior:generation'),
                                            ('likelihood', 'behavior:likelihood')])
def test_direct_response_task_sessions_are_recorded(kind, expected):
    from tests.test_task_responses import _ask, _model
    from brainscore_core.responses import Choice

    if kind == 'reply':
        model, _ = _model(replies={s: 'cat' for s in 'abcd'})
        task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
        channels = ['behavior:choice']
    else:
        model, _ = _model(scores={s: {'cat': -1.0, 'dog': -2.0} for s in 'abcd'})
        task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True))
        channels = ['behavior:choice', 'behavior:probabilities']
    with record_measures() as measures:
        session = _ask(model, task, channels)
        for channel in channels:
            session.collect(channel)

    assert _protocol(model, measures) == expected


def test_mixed_measures_in_one_collection_are_all_recorded():
    """An implied event then a reported event on one channel: concat keeps
    only the first assembly's attrs, so each event is noted on its own."""
    from brainscore_core.streaming import BufferedSession

    def event(measure, stimulus):
        return StreamEvent('behavior', BehavioralAssembly(
            [[1.0, 0.0]], coords={'stimulus_id': ('presentation', [stimulus]),
                                  'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs={'measure': measure}), None)

    with record_measures() as measures:
        session = BufferedSession(requested_output_channels=['behavior'])
        session.respond(event({'kind': 'implied', 'source': 'readout'}, 's0'))
        session.respond(event({'kind': 'reported', 'source': 'reply'}, 's1'))
        session.collect('behavior')
    score = stamp_score_metadata(Score(0.5), _Subject(), {'behavior'},
                                 protocol=_protocol(_Subject(), measures),
                                 harness_id='test', measures=measures)

    assert score.attrs['measure'] == ('implied', 'reported')
    assert score.attrs['protocol'] == 'behavior:generation'


def test_an_inner_score_passes_its_measures_to_the_outer_score():
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       metadata={'stimulus_set': _stimuli()})
    with record_measures() as outer:
        with record_measures() as inner:      # an inner score() call
            behavioral_response(subject, task)
        assert inner.measures == {('reported', 'reply')}

    assert _protocol(subject, outer) == 'behavior:generation'


def test_a_failed_inner_run_passes_nothing_to_the_outer_record():
    """Only a successful inner score reaches the outer result, so only its
    measures are passed on; the outer record is restored either way."""
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       metadata={'stimulus_set': _stimuli()})
    with record_measures() as outer:
        with pytest.raises(RuntimeError):
            with record_measures():
                behavioral_response(subject, task)
                raise RuntimeError('benchmark failed')
        subject.measure = {'kind': 'implied', 'source': 'likelihood'}
        behavioral_response(subject, task)     # lands in the outer record

    assert outer.measures == {('implied', 'likelihood')}


def test_an_event_made_before_the_run_and_sent_during_it_is_recorded():
    from brainscore_core.streaming import BufferedSession

    made_earlier = StreamEvent('behavior', BehavioralAssembly(
        [[1.0, 0.0]], coords={'stimulus_id': ('presentation', ['s0']),
                              'choice': ['cat', 'dog']},
        dims=['presentation', 'choice'],
        attrs={'measure': {'kind': 'reported', 'source': 'reply'}}), None)

    class _Replayer(_Answerer):
        def participate(self, session):
            while session.next_input() is not None:
                pass
            session.respond(made_earlier)

    subject = _Replayer(measure=None)
    with record_measures() as measures:
        session = BufferedSession(requested_output_channels=['behavior'])
        subject.participate(session)
        session.collect('behavior')

    assert _protocol(subject, measures) == 'behavior:generation'
