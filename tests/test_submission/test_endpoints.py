from collections import namedtuple

import logging

from brainscore_core import Score, Benchmark
from brainscore_core.submission import database_models
from brainscore_core.submission.database import connect_db
from brainscore_core.submission.database_models import clear_schema
from brainscore_core.submission.endpoints import RunScoringEndpoint, DomainPlugins, shorten_text
from tests.test_submission import init_users

test_database = 'brainscore-ohio-test'

logger = logging.getLogger(__name__)


class TestRunScoring:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(test_database)
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database entries')
        init_users()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_1model_1benchmark(self):
        class DummyDomainPlugins(DomainPlugins):
            def load_model(self, model_identifier: str):
                model_class = namedtuple('DummyModel',
                                         field_names=[])
                return model_class()

            def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
                benchmark_class = namedtuple('DummyBenchmark',
                                             field_names=['identifier', 'parent', 'version', 'bibtex', 'ceiling'])
                return benchmark_class(identifier='dummybenchmark', parent='neural', version=0, bibtex=None,
                                       ceiling=Score(1))

            def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
                return Score([0.8, 0.1], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        endpoint = RunScoringEndpoint(domain_plugins=DummyDomainPlugins(), db_secret=test_database)
        endpoint(domain='test', models=['dummymodel'], benchmarks=['dummybenchmark'],
                 jenkins_id=123, user_id=1, model_type='artificial_subject', public=True, competition=None)
        score_entries = database_models.Score.select()
        score_entries = list(score_entries)
        assert len(score_entries) == 1
        score_entry = score_entries[0]
        assert score_entry.score_raw == 0.8


class TestShortenText:
    def test_text_short_enough(self):
        text = 'lorem ipsum dolor sit amet'
        shortened = shorten_text(text, max_length=30)
        assert shortened == text

    def test_text_exact_size(self):
        text = 'lorem ipsum dolor sit amet'
        shortened = shorten_text(text, max_length=len(text))
        assert shortened == text

    def test_text_too_long(self):
        text = 'lorem ipsum dolor sit amet'
        shortened = shorten_text(text, max_length=20)
        assert len(shortened) == 20
        assert shortened == 'lorem[...]r sit amet'

    def test_text_too_long_by_1(self):
        text = 'lorem ipsum dolor sit amet'
        max_length = len(text) - 1
        shortened = shorten_text(text, max_length=max_length)
        assert len(shortened) == max_length
        assert shortened == 'lorem i[...]olor sit amet'

    def test_text_too_long_by_2(self):
        text = 'lorem ipsum dolor sit amet'
        max_length = len(text) - 2
        shortened = shorten_text(text, max_length=max_length)
        assert len(shortened) == max_length
        assert shortened == 'lorem i[...]lor sit amet'
