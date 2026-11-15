import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.metrics import Score
from brainscore_core.response_formats import response_record
from brainscore_core.score_metadata import (
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
        entry_point="test-harness")

    assert returned is score
    assert score.attrs["runtime_sec"] == 0.01
    assert score.attrs["in_channels"] == ("text", "vision")
    assert score.attrs["requested_outputs"] == ("neural:IT",)
    assert score.attrs["response"] == ()
    assert "protocol" not in score.attrs and "measure" not in score.attrs
    assert score.attrs["subject_identifier"] == "subject"
    assert score.attrs["entry_point"] == "test-harness"


class _NamedBenchmark:
    def __init__(self, identifier, trial_memory=None):
        self.identifier = identifier
        if trial_memory is not None:
            self.trial_memory = trial_memory


class _MethodNamedBenchmark:
    def identifier(self):
        return 'Method.identifier'


def _stamp_identity(score, benchmark, requested):
    return stamp_score_metadata(score, _Subject(), requested_channels=set(), entry_point='h',
                                benchmark=benchmark, benchmark_identifier=requested)


def test_the_stamp_writes_the_requested_identifier_and_the_trial_memory():
    score = _stamp_identity(Score(1), _NamedBenchmark('Bench.v1'), 'Bench-alias')
    assert score.attrs['benchmark_identifier'] == 'Bench-alias'
    assert score.attrs['trial_memory'] == 'none'          # declared by no attribute
    score = _stamp_identity(Score(1), _NamedBenchmark('Bench.v1', trial_memory='history'), 'Bench.v1')
    assert score.attrs['trial_memory'] == 'history'


@pytest.mark.parametrize('benchmark, requested, written', [
    (_NamedBenchmark('Bench.v1'), 'Bench-alias', 'Bench.v1'),       # the benchmark's own identifier
    (_NamedBenchmark('Bench.v1'), 'Bench-alias', 'Bench-alias'),    # the requested alias
    (_MethodNamedBenchmark(), 'alias', 'Method.identifier'),        # identifier() as a method
])
def test_a_benchmark_may_write_its_own_identifier_or_the_requested_one(benchmark, requested, written):
    score = Score(1)
    score.attrs['benchmark_identifier'] = written
    assert _stamp_identity(score, benchmark, requested).attrs['benchmark_identifier'] == written


@pytest.mark.parametrize('written', ['Other.benchmark', 7])
def test_another_written_identifier_is_refused(written):
    score = Score(1)
    score.attrs['benchmark_identifier'] = written
    with pytest.raises(ValueError, match='must match the requested identifier'):
        _stamp_identity(score, _NamedBenchmark('Bench.v1'), 'Bench-alias')


def test_without_the_benchmark_arguments_the_stamp_writes_neither_key():
    score = stamp_score_metadata(Score(1), _Subject(), requested_channels=set(), entry_point='h')
    assert 'benchmark_identifier' not in score.attrs and 'trial_memory' not in score.attrs


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


# --- the response records of the run's answers ---------------------------

REPLY = response_record('behavior', 'generation', parsed_by='model')
LIKELIHOOD = response_record('behavior', 'likelihood', conversion='softmax',
                             source_format='behavior:log_probabilities')


class _Answerer(Subject):
    """A participate-only subject (not a BrainScoreModel) with a fixed record."""

    identifier = 'answerer'
    in_channels = {'vision'}
    out_channels = {'behavior'}
    required_channels = set()

    def __init__(self, record):
        self.record = record

    def participate(self, session):
        while session.next_input() is not None:
            pass
        stimuli = session.stimuli
        n = len(stimuli)
        payload = BehavioralAssembly(
            np.tile([1.0, 0.0], (n, 1)),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'],
            attrs={} if self.record is None else {'response': self.record})
        for channel in session.outputs:
            session.respond(StreamEvent(channel, payload, None))

    def reset(self):
        pass


def _stimuli():
    return _make_image_stimulus_set(['cat'] * 3 + ['dog'] * 3, identifier='measure-test')


def _stamp(subject, measures):
    return stamp_score_metadata(Score(0.5), subject, {'behavior'}, entry_point='test',
                                measures=measures)


def test_a_subject_without_brainscore_model_gets_its_record_on_the_score():
    subject = _Answerer(REPLY)
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'],
                       instruction='Which animal?')
    with record_measures() as measures:
        behavioral_response(subject, _stimuli(), task)

    score = _stamp(subject, measures)

    assert score.attrs['response'] == (REPLY,)
    assert 'protocol' not in score.attrs and 'measure' not in score.attrs


def test_legacy_facade_behavioral_benchmark_gets_the_record():
    subject = _Answerer(REPLY)

    def benchmark(candidate):          # a legacy behavioral benchmark
        candidate.start_task('probabilities', fitting_stimuli=_stimuli())
        candidate.look_at(_stimuli())
        return Score(0.5)

    with record_measures() as measures:
        score = benchmark(LegacyBenchmarkFacade(subject, visual_degrees=8))
    stamp_score_metadata(score, subject, {'behavior'}, entry_point='test', measures=measures)

    assert score.attrs['response'] == (REPLY,)


def test_two_records_are_both_on_the_score_once_each():
    subject = _Answerer(REPLY)
    with record_measures() as measures:
        for record in (REPLY, LIKELIHOOD, REPLY):
            subject.record = record
            behavioral_response(subject, _stimuli(), TaskContext(
                task_type='probabilities', label_set=['cat', 'dog']))
    score = _stamp(subject, measures)

    assert sorted(score.attrs['response'], key=repr) == sorted([REPLY, LIKELIHOOD], key=repr)


def test_nothing_recorded_gives_an_empty_record():
    score = stamp_score_metadata(Score(0.5), _Subject(), {'behavior'}, entry_point='test')

    assert score.attrs['response'] == ()
    assert 'protocol' not in score.attrs


def test_an_answer_without_a_record_adds_nothing():
    subject = _Answerer(None)
    with record_measures() as measures:
        behavioral_response(subject, _stimuli(), TaskContext(
            task_type='probabilities', label_set=['cat', 'dog']))

    assert measures.responses == ()


def test_records_outside_a_run_are_not_recorded():
    subject = _Answerer(REPLY)
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    behavioral_response(subject, _stimuli(), task)            # no run open: nothing kept
    with record_measures() as measures:
        pass

    assert measures.responses == ()


# --- direct sessions, nesting, mixed collection --------------------------


def test_mixed_records_in_one_collection_are_all_recorded():
    """Two events with different records on one channel: each event is noted on its
    own, and collecting them as one assembly is refused (one assembly, one record)."""
    from brainscore_core.streaming import Session
    readout = response_record('behavior', 'readout')

    def event(record, stimulus):
        return StreamEvent('behavior', BehavioralAssembly(
            [[1.0, 0.0]], coords={'stimulus_id': ('presentation', [stimulus]),
                                  'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs={'response': record}), None)

    with record_measures() as measures:
        session = Session((), outputs=['behavior'])
        session.respond(event(readout, 's0'))
        session.respond(event(REPLY, 's1'))
        with pytest.raises(ValueError, match='different response records'):
            session.collect('behavior')

    assert sorted(_stamp(_Subject(), measures).attrs['response'], key=repr) == sorted(
        [readout, REPLY], key=repr)


def test_an_addressed_channel_refuses_answers_with_different_records():
    from brainscore_core.response_formats import _collect
    readout = response_record('behavior', 'readout')

    def answer(record, stimulus):
        return BehavioralAssembly(
            [[1.0, 0.0]], coords={'stimulus_id': ('presentation', [stimulus]),
                                  'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'], attrs={'response': record})

    with pytest.raises(ValueError, match='different response records'):
        _collect('behavior:probabilities', [answer(readout, 's0'), answer(REPLY, 's1')])
    same = _collect('behavior:probabilities', [answer(REPLY, 's0'), answer(REPLY, 's1')])
    assert same.attrs['response'] == REPLY and same.sizes['presentation'] == 2


def test_an_inner_score_passes_its_records_to_the_outer_score():
    subject = _Answerer(REPLY)
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    with record_measures() as outer:
        with record_measures() as inner:      # an inner score() call
            behavioral_response(subject, _stimuli(), task)
        assert inner.responses == (REPLY,)

    assert outer.responses == (REPLY,)


def test_a_failed_inner_run_passes_nothing_to_the_outer_record():
    """Only a successful inner score reaches the outer result, so only its
    records are passed on; the outer record is restored either way."""
    subject = _Answerer(REPLY)
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'])
    with record_measures() as outer:
        with pytest.raises(RuntimeError):
            with record_measures():
                behavioral_response(subject, _stimuli(), task)
                raise RuntimeError('benchmark failed')
        subject.record = LIKELIHOOD
        behavioral_response(subject, _stimuli(), task)     # lands in the outer record

    assert outer.responses == (LIKELIHOOD,)


def test_an_event_made_before_the_run_and_sent_during_it_is_recorded():
    from brainscore_core.streaming import Session

    made_earlier = StreamEvent('behavior', BehavioralAssembly(
        [[1.0, 0.0]], coords={'stimulus_id': ('presentation', ['s0']),
                              'choice': ['cat', 'dog']},
        dims=['presentation', 'choice'], attrs={'response': REPLY}), None)

    class _Replayer(_Answerer):
        def participate(self, session):
            while session.next_input() is not None:
                pass
            session.respond(made_earlier)

    subject = _Replayer(record=None)
    with record_measures() as measures:
        session = Session((), outputs=['behavior'])
        subject.participate(session)
        session.collect('behavior')

    assert measures.responses == (REPLY,)


def test_a_score_with_records_survives_pickling():
    import pickle
    subject = _Answerer(LIKELIHOOD)
    with record_measures() as measures:
        behavioral_response(subject, _stimuli(), TaskContext(
            task_type='probabilities', label_set=['cat', 'dog']))
    score = pickle.loads(pickle.dumps(_stamp(subject, measures)))

    assert score.attrs['response'] == (LIKELIHOOD,)


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
                                entry_point='test', measures=measures)


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
                                 entry_point='test', measures=measures)
    assert score.attrs['delivery'] == ('listed', 'stream')


def test_a_run_without_sessions_records_no_delivery(monkeypatch):
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', 'one')
    with record_measures() as measures:
        pass
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels=set(),
                                 entry_point='test', measures=measures)
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
        "brainscore-core", "brainscore", "brainscore_vision", "brainscore_language",
        "environment_fingerprint", "accelerator"}


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
            'stamp_score_metadata(Score(1), object(), set(), entry_point="t", '
            'measures=MeasureRecord()); '
            'assert "torch" not in sys.modules; print("torch" in sys.modules)')
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == 'False'


# Provenance of a session, noted once (rule 6.3)

def _noted(entries, monkeypatch, disable='1', answer_with=None):
    """One answered session per entry; each entry is noted with note_session."""
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    monkeypatch.setenv('RESULTCACHING_DISABLE', disable)
    with record_measures() as measures:
        for entry in entries:
            session = Session(_events(), ['neural:IT'])
            note_session(session, provenance=entry)
            _answer_session(session)
    return stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                entry_point='test', measures=measures)


def test_a_noted_entry_reaches_the_score_with_the_store_state(monkeypatch):
    entry = {'subject': 'subject', 'parts': [{'serves': 'neural:IT', 'kind': 'pkg.Wrapper'}]}
    score = _noted([entry], monkeypatch)
    assert score.attrs['provenance'] == ({**entry, 'activation_store': 'off'},)
    assert float(score) == 0.5


@pytest.mark.parametrize('disable,state', [('1', 'off'), ('0', 'on'), ('', 'on'),
                                           ('pkg.Wrapper', 'off for: pkg.Wrapper')])
def test_the_store_state_follows_the_switch(monkeypatch, disable, state):
    score = _noted([{'subject': 'subject', 'parts': []}], monkeypatch, disable=disable)
    assert score.attrs['provenance'] == ({'subject': 'subject', 'parts': [], 'activation_store': state},)


def test_the_same_entry_is_kept_once_and_two_entries_both(monkeypatch):
    one, two = {'subject': 'subject', 'parts': []}, {'subject': 'subject', 'registration': {'dtype': 'bf16'}}
    score = _noted([one, one, two], monkeypatch)
    assert score.attrs['provenance'] == ({**one, 'activation_store': 'off'},
                                         {**two, 'activation_store': 'off'})


def test_a_subject_that_notes_nothing_gets_facts_not_stated(monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    score = _stamped(_events(), monkeypatch)
    assert score.attrs['provenance'] == (
        {'subject': 'subject', 'parts': 'not stated', 'activation_store': 'off'},)


def test_a_stamp_without_a_run_states_no_store():
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels=set(), entry_point='test')
    assert score.attrs['provenance'] == (
        {'subject': 'subject', 'parts': 'not stated', 'activation_store': 'not stated'},)


def test_the_score_keeps_a_copy_of_the_entry(monkeypatch):
    entry = {'subject': 'subject', 'parts': [], 'registration': {'shard': 'x'}}
    score = _noted([entry], monkeypatch)
    entry['registration']['shard'] = 'edited'
    assert score.attrs['provenance'][0]['registration'] == {'shard': 'x'}


def test_an_inner_score_passes_its_entries_to_the_outer_record(monkeypatch):
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    with record_measures() as outer:
        with record_measures():
            note_session(Session(_events(), ['neural:IT']), provenance={'subject': 'inner'})
    assert outer.provenance == ({'subject': 'inner', 'activation_store': 'off'},)


def test_responses_and_their_attrs_are_not_read_for_provenance(monkeypatch):
    """A payload or meta mapping named provenance is the subject's data; the score ignores it."""
    from brainscore_core.streaming import Session
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    with record_measures() as measures:
        session = Session(_events(), ['neural:IT'])
        while (event := session.next_input()) is not None:
            payload = NeuroidAssembly(np.ones((1, 1)), dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', ['s0']), 'neuroid_id': ('neuroid', ['n0'])})
            payload.attrs['provenance'] = {'dtype': 'x'}
            session.respond(StreamEvent('neural:IT', payload, event.t_ms, {'provenance': {'dtype': 'y'}}))
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                 entry_point='test', measures=measures)
    assert score.attrs['provenance'] == (
        {'subject': 'subject', 'parts': 'not stated', 'activation_store': 'off'},)
    assert session.emitted[0].payload.attrs == {'provenance': {'dtype': 'x'}}


def test_a_registration_entry_alone_does_not_state_parts(monkeypatch):
    registration = {'subject': 'subject', 'registration': {'dtype': 'bf16'}}
    score = _noted([registration], monkeypatch)
    assert score.attrs['provenance'] == (
        {'subject': 'subject', 'parts': 'not stated', 'activation_store': 'off'},
        {**registration, 'activation_store': 'off'})


def _two_subjects(monkeypatch, nested=False):
    """A model that states its parts and a custom subject, each in its own session of one score."""
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')

    def run():
        model_session = Session(_events(), ['neural:IT'])
        _answer_session(model_session)
        note_session(model_session, provenance={'subject': 'model', 'parts': [{'serves': 'neural:IT'}]})
        custom_session = Session(_events(), ['neural:IT'])
        note_session(custom_session, subject='custom')        # the caller that runs it names it
        _answer_session(custom_session)
    with record_measures() as measures:
        if nested:
            with record_measures():
                run()
        else:
            run()
    return stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                entry_point='test', measures=measures)


@pytest.mark.parametrize('nested', [False, True], ids=['one record', 'nested record'])
def test_a_silent_subject_beside_a_noted_one_gets_its_own_entry(monkeypatch, nested):
    assert _two_subjects(monkeypatch, nested).attrs['provenance'] == (
        {'subject': 'custom', 'parts': 'not stated', 'activation_store': 'off'},
        {'subject': 'model', 'parts': [{'serves': 'neural:IT'}], 'activation_store': 'off'})


def test_an_unnamed_session_belongs_to_the_scored_subject(monkeypatch):
    """A noted model and a session nobody names: the scored subject took part and stated nothing."""
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    with record_measures() as measures:
        noted = Session(_events(), ['neural:IT'])
        note_session(noted, provenance={'subject': 'model', 'parts': []})
        _answer_session(Session(_events(), ['neural:IT']))
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                 entry_point='test', measures=measures)
    assert score.attrs['provenance'] == (     # sorted by content
        {'subject': 'subject', 'parts': 'not stated', 'activation_store': 'off'},
        {'subject': 'model', 'parts': [], 'activation_store': 'off'})


def test_the_record_keeps_no_session_alive():
    import gc
    import weakref
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    with record_measures() as measures:
        session = Session(_events(), ['neural:IT'])
        note_session(session, subject='custom')
        reference = weakref.ref(session)
        del session
        gc.collect()
        assert reference() is None
    assert list(measures._sessions.values()) == ['custom']


def test_an_inner_record_binds_its_unnamed_sessions_to_its_own_subject(monkeypatch):
    """The session of an inner score that no note names belongs to the inner score's subject,
    also after the inner record passes to the outer one."""
    from brainscore_core.streaming import Session
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    with record_measures(subject='subject') as outer:
        with record_measures(subject='inner-subject'):
            _answer_session(Session(_events(), ['neural:IT']))
    score = stamp_score_metadata(Score(0.5), _Subject(), requested_channels={'neural:IT'},
                                 entry_point='test', measures=outer)
    assert score.attrs['provenance'] == (
        {'subject': 'inner-subject', 'parts': 'not stated', 'activation_store': 'off'},)


def test_a_named_session_keeps_its_name_when_the_record_binds_the_others(monkeypatch):
    from brainscore_core._evaluation.measures import note_session
    from brainscore_core.streaming import Session
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    with record_measures(subject='subject') as measures:
        named = Session(_events(), ['neural:IT'])
        note_session(named, subject='partner')
        _answer_session(named)
        _answer_session(Session(_events(), ['neural:IT']))
    assert sorted(measures._sessions.values()) == ['partner', 'subject']
