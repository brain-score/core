import numpy as np
import pandas as pd
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
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _make_image_stimulus_set(labels, identifier='test_stim'):
    n = len(labels)
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_file_name': [f'/tmp/s{i}.png' for i in range(n)],
        'image_label': labels,
    }))
    stimuli.identifier = identifier
    return stimuli


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
        harness_id="test-harness")

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
        stimuli = session.stimuli
        n = len(stimuli)
        payload = BehavioralAssembly(
            np.tile([1.0, 0.0], (n, 1)),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs={'measure': self.measure})
        for channel in session.outputs:
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
                       instruction='Which animal?')
    with record_measures() as measures:
        behavioral_response(subject, _stimuli(), task)

    score = stamp_score_metadata(Score(0.5), subject, {'behavior'},
                                 protocol=_protocol(subject, measures),
                                 harness_id='test', measures=measures)

    assert score.attrs['protocol'] == 'behavior:generation'
    assert score.attrs['measure'] == 'reported'


def test_likelihood_subject_is_not_stamped_readout():
    subject = _Answerer({'kind': 'implied', 'source': 'likelihood'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    with record_measures() as measures:
        behavioral_response(subject, _stimuli(), task)

    assert _protocol(subject, measures) == 'behavior:likelihood'


def test_implied_measure_without_a_known_source_does_not_claim_readout():
    subject = _Answerer({'kind': 'implied'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    with record_measures() as measures:
        behavioral_response(subject, _stimuli(), task)

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
            behavioral_response(subject, _stimuli(), TaskContext(
                task_type='probabilities', label_set=['cat', 'dog']))
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
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    behavioral_response(subject, _stimuli(), task)            # no run open: nothing kept
    with record_measures() as measures:
        pass

    assert measures.measures == set()


_GENERATION_TASK = dict(task_type='probabilities', label_set=['cat', 'dog'],
                        instruction='What animal is this?')


# --- direct sessions, nesting, mixed collection --------------------------


def test_mixed_measures_in_one_collection_are_all_recorded():
    """An implied event then a reported event on one channel: concat keeps
    only the first assembly's attrs, so each event is noted on its own."""
    from brainscore_core.streaming import Session

    def event(measure, stimulus):
        return StreamEvent('behavior', BehavioralAssembly(
            [[1.0, 0.0]], coords={'stimulus_id': ('presentation', [stimulus]),
                                  'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs={'measure': measure}), None)

    with record_measures() as measures:
        session = Session((), outputs=['behavior'])
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
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    with record_measures() as outer:
        with record_measures() as inner:      # an inner score() call
            behavioral_response(subject, _stimuli(), task)
        assert inner.measures == {('reported', 'reply')}

    assert _protocol(subject, outer) == 'behavior:generation'


def test_a_failed_inner_run_passes_nothing_to_the_outer_record():
    """Only a successful inner score reaches the outer result, so only its
    measures are passed on; the outer record is restored either way."""
    subject = _Answerer({'kind': 'reported', 'source': 'reply'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    with record_measures() as outer:
        with pytest.raises(RuntimeError):
            with record_measures():
                behavioral_response(subject, _stimuli(), task)
                raise RuntimeError('benchmark failed')
        subject.measure = {'kind': 'implied', 'source': 'likelihood'}
        behavioral_response(subject, _stimuli(), task)     # lands in the outer record

    assert outer.measures == {('implied', 'likelihood')}


def test_an_event_made_before_the_run_and_sent_during_it_is_recorded():
    from brainscore_core.streaming import Session

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
        session = Session((), outputs=['behavior'])
        subject.participate(session)
        session.collect('behavior')

    assert _protocol(subject, measures) == 'behavior:generation'


# ── Execution configuration and software versions in the provenance ─────────

def _answer_session(session):
    """Answer every input of ``session`` on its one requested output."""
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    while (event := session.next_input()) is not None:
        session.respond(StreamEvent(session.outputs[0], NeuroidAssembly(
            np.ones((1, 1)), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', ['s0']), 'neuroid_id': ('neuroid', ['n0'])}),
            event.t_ms))


def _stamped(events, monkeypatch, plan=None):
    """Run one answered session; ``plan`` is what a subject that owns the setting passes."""
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', 'one')   # core must not read it
    with record_measures() as measures:
        session = Session(events, ['neural:IT'])
        if plan is not None:
            note_session(session, plan)
        _answer_session(session)
    return stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                protocol='neural', harness_id='test', measures=measures)


def _events():
    return [StreamEvent('text', 'a cat', 0.0)]


def test_a_listed_run_records_its_delivery_and_plan(monkeypatch):
    score = _stamped(_events(), monkeypatch, plan='all')
    assert score.attrs['delivery'] == 'listed'
    assert score.attrs['regions_per_pass'] == 'all'
    assert float(score) == 0.5


def test_a_streamed_run_records_its_delivery_and_plan(monkeypatch):
    score = _stamped(iter(_events()), monkeypatch, plan='one')
    assert score.attrs['delivery'] == 'stream'
    assert score.attrs['regions_per_pass'] == 'one'


def test_a_run_with_both_deliveries_lists_both(monkeypatch):
    from brainscore_core.streaming import Session
    with record_measures() as measures:
        _answer_session(Session(_events(), ['neural:IT']))
        _answer_session(Session(iter(_events()), ['neural:IT']))
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                 protocol='neural', harness_id='test', measures=measures)
    assert score.attrs['delivery'] == ('listed', 'stream')


def test_a_run_without_sessions_records_no_delivery(monkeypatch):
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', 'one')
    with record_measures() as measures:
        pass
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels=set(),
                                 protocol=None, harness_id='test', measures=measures)
    assert score.attrs['delivery'] is None
    assert score.attrs['regions_per_pass'] is None


def test_a_subject_that_passes_no_plan_is_stamped_none(monkeypatch):
    """A legacy-style subject does not read the setting: core must not stamp it."""
    score = _stamped(_events(), monkeypatch)       # the variable says 'one'; nobody passed it
    assert score.attrs['delivery'] == 'listed'
    assert score.attrs['regions_per_pass'] is None



def test_the_score_records_software_versions_and_commits(monkeypatch):
    import platform
    from importlib.metadata import version
    score = _stamped(_events(), monkeypatch)
    software = score.attrs['software']
    assert software['python'] == platform.python_version()
    assert software['numpy'] == version('numpy')
    assert software['xarray'] == version('xarray')
    assert set(software) >= {'torch', 'transformers', 'scikit-learn', 'brainscore-core',
                             'brainscore', 'brainscore_vision', 'brainscore_language'}
    commits = score.attrs['code_commits']
    assert set(commits) == {'brainscore_core', 'brainscore', 'brainscore_vision',
                            'brainscore_language'}
    commit = commits['brainscore_core']
    assert commit is None or (len(commit) == 40 and all(c in '0123456789abcdef' for c in commit))


def _fake_package(monkeypatch, folder):
    """Make a stand-in package look imported from ``folder/stand_in_package``.

    The stand-in has its own name: a module registered under a real Brain-Score
    name from a temporary folder would count as an import outside the installed
    package in the installed-package checks.
    """
    import sys
    import types
    from brainscore_core import environment_check
    (folder / "stand_in_package").mkdir(parents=True)
    (folder / "pyproject.toml").write_text("[project]\nname='stand-in'\n")
    module = types.ModuleType("stand_in_package")
    module.__file__ = str(folder / "stand_in_package" / "__init__.py")
    monkeypatch.setitem(sys.modules, "stand_in_package", module)
    monkeypatch.setattr(environment_check, "_BRAINSCORE_PACKAGES",
                        (("stand_in_package", "stand-in"),))


def test_a_git_checkout_stamps_its_real_forty_digit_commit(monkeypatch, tmp_path):
    import subprocess
    from brainscore_core.environment_check import _score_environment
    _fake_package(monkeypatch, tmp_path)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
    expected = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    commit = _score_environment()["code_commits"]["stand_in_package"]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    assert commit == expected


def test_no_checkout_stamps_none(monkeypatch, tmp_path):
    from brainscore_core.environment_check import _score_environment
    _fake_package(monkeypatch, tmp_path)
    assert _score_environment()["code_commits"]["stand_in_package"] is None


def test_software_keys_are_installed_distribution_names():
    from brainscore_core.environment_check import _score_environment
    assert set(_score_environment()["software"]) == {
        "python", "numpy", "torch", "transformers", "scikit-learn", "xarray",
        "brainscore-core", "brainscore", "brainscore_vision", "brainscore_language"}


def test_nested_score_records_merge_their_deliveries():
    from brainscore_core._evaluation.measures import execution_attrs, note_session
    from brainscore_core.streaming import Session
    with record_measures() as outer:
        note_session(Session(_events(), ["neural:IT"]), "all")
        with record_measures():                         # an inner score() call
            note_session(Session(iter(_events()), ["neural:IT"]), "one")
    assert execution_attrs(outer) == {"delivery": ("listed", "stream"),
                                      "regions_per_pass": ("all", "one")}


def test_reading_versions_does_not_import_torch():
    import subprocess
    import sys
    code = ('import sys; from brainscore_core.environment_check import _score_environment; '
            '_score_environment(); '
            'from brainscore_core.score_metadata import stamp_score_metadata; '
            'from brainscore_core._evaluation.measures import MeasureRecord; '
            'from brainscore_core.metrics import Score; '
            'stamp_score_metadata(Score(1), object(), set(), protocol=None, harness_id="t", '
            'measures=MeasureRecord()); '
            'assert "torch" not in sys.modules; print("torch" in sys.modules)')
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == 'False'
