import logging

import numpy as np

from brainscore_core import Score as ScoreObject
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.submission.database import (connect_db, reference_from_bibtex, benchmarkinstance_from_benchmark,
                                                 submissionentry_from_meta, modelentry_from_model, update_score,
                                                 public_model_identifiers, public_benchmark_identifiers,
                                                 email_from_uid, uid_from_email)
from brainscore_core.submission.database_models import Score, BenchmarkType, BenchmarkInstance, Reference, clear_schema
from tests.test_submission import init_users

logger = logging.getLogger(__name__)

SAMPLE_BIBTEX = """@Article{Freeman2013,
                                author={Freeman, Jeremy and Ziemba, Corey M. and Heeger, David J. 
                                        and Simoncelli, Eero P. and Movshon, J. Anthony},
                                title={A functional and perceptual signature of the second visual area in primates},
                                journal={Nature Neuroscience},
                                year={2013},
                                month={Jul},
                                day={01},
                                volume={16},
                                number={7},
                                pages={974-981},
                                issn={1546-1726},
                                doi={10.1038/nn.3402},
                                url={https://doi.org/10.1038/nn.3402}
                                }"""


class SchemaTest:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(db_secret='sqlite3.db')
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database entries')
        init_users()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()


class TestUser(SchemaTest):
    def test_email_from_uid(self):
        email = email_from_uid(1)
        assert email == 'test@brainscore.com'

    def test_uid_from_email(self):
        uid = uid_from_email('admin@brainscore.com')
        assert uid == 2

    def test_uid_from_email_does_not_exist(self):
        uid = uid_from_email('doesnotexist@brainscore.com')
        assert uid == None


def _mock_submission_entry(jenkins_id=123, user_id=1, model_type='base_model'):
    return submissionentry_from_meta(jenkins_id=jenkins_id, user_id=user_id, model_type=model_type)


class TestModel(SchemaTest):
    def test_submission(self):
        entry = submissionentry_from_meta(jenkins_id=123, user_id=1, model_type='artificial_subject')
        assert entry.status == 'running'

    def test_model_no_bibtex(self):
        submission_entry = _mock_submission_entry()
        entry = modelentry_from_model(model_identifier='dummy', domain='test',
                                      submission=submission_entry, public=False, competition='cosyne2022')
        assert entry.reference is None

    def test_model_with_bibtex(self):
        submission_entry = _mock_submission_entry()
        entry = modelentry_from_model(model_identifier='dummy', domain='test',
                                      submission=submission_entry, public=True, competition=None, bibtex=SAMPLE_BIBTEX)
        assert entry.reference.year == '2013'

    def test_resubmission(self):  # make model entry from user 1, then retrieve model entry from user 2 ("resubmit")
        submission_entry = _mock_submission_entry()
        params = dict(model_identifier='dummy', domain='test', public=True, competition=None, bibtex=SAMPLE_BIBTEX)
        original_entry = modelentry_from_model(**params, submission=submission_entry)
        # resubmit
        resubmission = _mock_submission_entry(user_id=2)
        resubmit_entry = modelentry_from_model(**params, submission=resubmission)
        # even though resubmission had different submitter, model owner should still be original user
        assert original_entry.owner == resubmit_entry.owner


class _MockBenchmark(BenchmarkBase):
    def __init__(self):
        dummy_ceiling = ScoreObject(0.6)
        dummy_ceiling.attrs['error'] = 0.1
        super(_MockBenchmark, self).__init__(
            identifier='dummy', ceiling=dummy_ceiling, version=0, parent='neural')


class TestBenchmark(SchemaTest):
    def test_benchmark_instance_no_parent(self):
        benchmark = _MockBenchmark()
        instance = benchmarkinstance_from_benchmark(benchmark, domain='test')
        type = BenchmarkType.get(identifier=instance.benchmark)
        assert instance.ceiling == 0.6
        assert instance.ceiling_error == 0.1
        assert not type.parent

    def test_benchmark_instance_existing_parent(self):
        # initially create the parent to see if the benchmark properly links to it
        BenchmarkType.create(identifier='neural', order=3, domain='test', owner_id=2)
        benchmark = _MockBenchmark()
        instance = benchmarkinstance_from_benchmark(benchmark, domain='test')
        assert instance.benchmark.parent.identifier == 'neural'

    def test_reference(self):
        ref = reference_from_bibtex(SAMPLE_BIBTEX)
        assert isinstance(ref, Reference)
        assert ref.url == 'https://doi.org/10.1038/nn.3402'
        assert ref.year == '2013'
        assert ref.author is not None
        ref2 = reference_from_bibtex(SAMPLE_BIBTEX)
        assert ref2.id == ref.id


def _create_score_entry():
    submission_entry = submissionentry_from_meta(jenkins_id=123, user_id=1, model_type='brain_model')
    model_entry = modelentry_from_model(model_identifier='dummy', domain='test',
                                        submission=submission_entry, public=True, competition=None)
    benchmark_entry = benchmarkinstance_from_benchmark(_MockBenchmark(), domain='test')
    entry, created = Score.get_or_create(benchmark=benchmark_entry, model=model_entry)
    return entry


class TestScore(SchemaTest):
    def test_score_no_ceiling(self):
        score = ScoreObject([.123, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        entry = _create_score_entry()
        update_score(score, entry)
        assert entry.score_ceiled is None
        assert np.isnan(entry.error)
        assert entry.score_raw == .123

    def test_score_with_ceiling(self):
        score = ScoreObject([.42, .1], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score.attrs['raw'] = ScoreObject([.336, .08], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score.attrs['ceiling'] = ScoreObject(.8)
        entry = _create_score_entry()
        update_score(score, entry)
        assert entry.score_ceiled == .42
        assert entry.error == .1
        assert entry.score_raw == .336

    def test_score_no_aggregation(self):
        score = ScoreObject(.42)
        entry = _create_score_entry()
        update_score(score, entry)
        assert entry.score_raw == .42
        assert entry.error is None

    def test_score_error_attr(self):
        score = ScoreObject(.42)
        score.attrs['error'] = .1
        entry = _create_score_entry()
        update_score(score, entry)
        assert entry.error == .1


class TestPublic(SchemaTest):
    def test_one_public_model(self):
        # create model
        submission_entry = _mock_submission_entry()
        modelentry_from_model(model_identifier='dummy', domain='test',
                              public=True, competition=None, submission=submission_entry)
        # test
        public_models = public_model_identifiers(domain='test')
        assert public_models == ["dummy"]

    def test_one_public_one_private_model(self):
        # create models
        submission = _mock_submission_entry()
        modelentry_from_model(model_identifier='dummy_public', domain='test',
                              public=True, competition=None, submission=submission)
        modelentry_from_model(model_identifier='dummy_private', domain='test',
                              public=False, competition=None, submission=submission)
        # test
        public_models = public_model_identifiers(domain='test')
        assert public_models == ["dummy_public"]

    def test_two_public_one_private_model(self):
        # create models
        submission = _mock_submission_entry()
        modelentry_from_model(model_identifier='dummy_public1', domain='test',
                              public=True, competition=None, submission=submission)
        modelentry_from_model(model_identifier='dummy_public2', domain='test',
                              public=True, competition=None, submission=submission)
        modelentry_from_model(model_identifier='dummy_private', domain='test',
                              public=False, competition=None, submission=submission)
        # test
        public_models = public_model_identifiers(domain='test')
        assert set(public_models) == {"dummy_public1", "dummy_public2"}

    def test_one_public_benchmark(self):
        # create benchmarktypes and benchmarkinstances
        BenchmarkType.create(identifier='dummy', domain='test', visible=True, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy')
        # test
        public_benchmarks = public_benchmark_identifiers(domain='test')
        assert public_benchmarks == ["dummy"]

    def test_one_public_one_private_benchmark(self):
        # create benchmarktypes and benchmarkinstances
        BenchmarkType.create(identifier='dummy_public', domain='test', visible=True, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_public')

        BenchmarkType.create(identifier='dummy_private', domain='test', visible=False, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_private')
        # test
        public_benchmarks = public_benchmark_identifiers(domain='test')
        assert public_benchmarks == ["dummy_public"]

    def test_two_public_two_private_benchmarks(self):
        # create benchmarktypes and benchmarkinstances
        BenchmarkType.create(identifier='dummy_public1', domain='test', visible=True, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_public1')

        BenchmarkType.create(identifier='dummy_public2', domain='test', visible=True, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_public2')

        BenchmarkType.create(identifier='dummy_private1', domain='test', visible=False, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_private1')

        BenchmarkType.create(identifier='dummy_private2', domain='test', visible=False, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_private2')
        # test
        public_benchmarks = public_benchmark_identifiers(domain='test')
        assert set(public_benchmarks) == {"dummy_public1", "dummy_public2"}

    def test_with_parent_benchmark(self):
        # create benchmarktypes and benchmarkinstances
        BenchmarkType.create(identifier='dummy_parent', domain='test', visible=True, order=1, owner_id=2)

        BenchmarkType.create(identifier='dummy_child1', parent="dummy_parent",
                             domain='test', visible=True, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_child1')

        BenchmarkType.create(identifier='dummy_child2', parent="dummy_parent",
                             domain='test', visible=True, order=1, owner_id=2)
        BenchmarkInstance.create(benchmark='dummy_child2')

        # test
        public_benchmarks = public_benchmark_identifiers(domain='test')
        assert public_benchmarks == ["dummy_child1", "dummy_child2"]
