import logging
from collections import namedtuple

import botocore.exceptions

from brainscore_core import Score, Benchmark
from brainscore_core.submission import database_models
from brainscore_core.submission.database import connect_db
from brainscore_core.submission.database_models import Model, BenchmarkType, BenchmarkInstance, clear_schema
from brainscore_core.submission.endpoints import RunScoringEndpoint, DomainPlugins, UserManager, shorten_text, \
    resolve_models_benchmarks, resolve_models, resolve_benchmarks, make_argparser
from tests.test_submission import init_users

logger = logging.getLogger(__name__)

POSTGRESQL_TEST_DATABASE = 'brainscore-ohio-test'


class TestUserManager:
    test_database = None

    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        try:
            connect_db(db_secret=POSTGRESQL_TEST_DATABASE)
            cls.test_database = POSTGRESQL_TEST_DATABASE
        except botocore.exceptions.NoCredentialsError:  # we're in an environment where we cannot retrieve AWS secrets
            connect_db(db_secret='sqlite3.db')
            cls.test_database = 'sqlite3.db'  # -> use local sqlite database
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database entries')
        init_users()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_create_new_user(self, requests_mock):
        # mock GET & POST responses
        get_adapter = requests_mock.get('https://www.brain-score.org/signup', cookies={'cookie_name': 'cookie_value'})
        post_adapter = requests_mock.post('https://www.brain-score.org/signup', status_code=200)

        user_manager = UserManager(self.test_database)
        user_manager.create_new_user('test@example.com')

        assert get_adapter.call_count == 1
        assert post_adapter.call_count == 1

    def test_get_uid_for_existing_user(self):
        user_manager = UserManager(self.test_database)
        uid = user_manager.get_uid('admin@brainscore.com')
        assert uid == 2

    def test_send_user_email(self, mocker):
        smtp_mock = mocker.MagicMock(name='smtp_mock')
        mocker.patch('brainscore_core.submission.endpoints.smtplib.SMTP_SSL', new=smtp_mock)
        user_manager = UserManager(self.test_database)
        user_manager.send_user_email(2, 'Subject', 'Test email body', 'sender@gmail.com', 'testpassword')
        smtp_mock.assert_called_once_with('smtp.gmail.com', 465)


class DummyDomainPlugins(DomainPlugins):
    def load_model(self, model_identifier: str):
        model_class1 = namedtuple('DummyModel1', field_names=['identifier'])
        model_class2 = namedtuple('DummyModel2', field_names=['identifier'])
        model_class = {
            "dummymodel1": model_class1,
            "dummymodel2": model_class2,
        }[model_identifier]
        return model_class(identifier=model_identifier)

    def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
        benchmark_class1 = namedtuple('DummyBenchmark1',
                                      field_names=['identifier', 'parent', 'version', 'bibtex', 'ceiling'])
        benchmark_class2 = namedtuple('DummyBenchmark2',
                                      field_names=['identifier', 'parent', 'version', 'bibtex', 'ceiling'])
        benchmark_class = {
            "dummybenchmark1": benchmark_class1,
            "dummybenchmark2": benchmark_class2,
        }[benchmark_identifier]
        return benchmark_class(identifier=benchmark_identifier, parent='neural', version=0, bibtex=None,
                               ceiling=Score(1))

    def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
        return Score([0.8, 0.1], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])


class TestRunScoring:
    test_database = None

    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        try:
            connect_db(db_secret=POSTGRESQL_TEST_DATABASE)
            cls.test_database = POSTGRESQL_TEST_DATABASE
            raise botocore.exceptions.NoCredentialsError
        except botocore.exceptions.NoCredentialsError:  # we're in an environment where we cannot retrieve AWS secrets
            connect_db(db_secret='sqlite3.db')
            cls.test_database = 'sqlite3.db'  # -> use local sqlite database
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database entries')
        init_users()

        for model_id in ["dummymodel1", "dummymodel2"]:
            Model.get_or_create(name=model_id, domain="test", public=True, owner=2, submission=0)
        for benchmark_id in ["dummybenchmark1", "dummybenchmark2"]:
            BenchmarkType.get_or_create(identifier=benchmark_id, domain="test", visible=True, order=999)
            BenchmarkInstance.get_or_create(benchmark=benchmark_id)

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_get_models_list_benchmarks_list(self):
        domain, models, benchmarks = 'test', ['dummymodel1'], ['dummybenchmark1']

        endpoint_models = resolve_models(domain=domain, models=models)
        endpoint_benchmarks = resolve_benchmarks(domain=domain, benchmarks=benchmarks)
        assert endpoint_models == models
        assert endpoint_benchmarks == benchmarks

    def test_get_models_all_benchmarks_list(self):
        domain, models, benchmarks = 'test', RunScoringEndpoint.ALL_PUBLIC, ['dummybenchmark1']

        endpoint_models = resolve_models(domain=domain, models=models)
        endpoint_benchmarks = resolve_benchmarks(domain=domain, benchmarks=benchmarks)
        assert endpoint_models == ["dummymodel1", "dummymodel2"]
        assert endpoint_benchmarks == benchmarks

    def test_get_models_list_benchmarks_all(self):
        domain, models, benchmarks = 'test', ['dummymodel1'], RunScoringEndpoint.ALL_PUBLIC

        endpoint_models = resolve_models(domain=domain, models=models)
        endpoint_benchmarks = resolve_benchmarks(domain=domain, benchmarks=benchmarks)
        assert endpoint_models == models
        assert endpoint_benchmarks == ['dummybenchmark1', 'dummybenchmark2']

    def test_get_models_all_benchmarks_all(self):
        domain, models, benchmarks = 'test', RunScoringEndpoint.ALL_PUBLIC, RunScoringEndpoint.ALL_PUBLIC

        endpoint_models = resolve_models(domain=domain, models=models)
        endpoint_benchmarks = resolve_benchmarks(domain=domain, benchmarks=benchmarks)
        assert endpoint_models == ["dummymodel1", "dummymodel2"]
        assert endpoint_benchmarks == ['dummybenchmark1', 'dummybenchmark2']

    def test_resolve_models_and_benchmarks(self):
        domain, new_models, new_benchmarks = 'test', ['dummymodel1'], ['dummybenchmark1']
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'artificialsubject',
                     'public': True, 'competition': 'None', 'new_models': new_models,
                     'new_benchmarks': new_benchmarks, 'specified_only': True}
        model_ids, benchmark_ids = resolve_models_benchmarks(domain=domain, args_dict=args_dict)

        assert model_ids == new_models
        assert benchmark_ids == new_benchmarks

    def test_score_model_benchmark(self):
        domain, model_id, benchmark_id = 'test', 'dummymodel1', 'dummybenchmark1'

        endpoint = RunScoringEndpoint(domain_plugins=DummyDomainPlugins(), db_secret=self.test_database)
        endpoint(domain=domain, model_identifier=model_id, benchmark_identifier=benchmark_id,
                 jenkins_id=123, user_id=1, model_type='artificial_subject', public=True, competition=None)

        score_entries = database_models.Score.select()
        score_entries = list(score_entries)
        assert len(score_entries) == 1
        score_entry = score_entries[0]
        assert score_entry.score_raw == 0.8

    def test_full_scoring(self):
        domain, models, benchmarks = 'test', ['dummymodel1'], RunScoringEndpoint.ALL_PUBLIC

        models = resolve_models(domain=domain, models=models)
        benchmarks = resolve_benchmarks(domain=domain, benchmarks=benchmarks)
        endpoint = RunScoringEndpoint(domain_plugins=DummyDomainPlugins(), db_secret=self.test_database)

        for model_id in models:
            for benchmark_id in benchmarks:
                endpoint(domain=domain, model_identifier=model_id, benchmark_identifier=benchmark_id, jenkins_id=123,
                         user_id=1, model_type='artificial_subject', public=True, competition=None)
        score_entries = database_models.Score.select()
        score_entries = list(score_entries)
        assert len(score_entries) == 2
        for score_entry in score_entries:
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


class TestArgparser:
    def test_competition_None(self):
        parser = make_argparser()
        args = parser.parse_args([0,  # required jenkins_id
                                  '--competition', 'None'])
        assert args.competition is None

    def test_competition_cosyne2022(self):
        parser = make_argparser()
        args = parser.parse_args([0,  # required jenkins_id
                                  '--competition', 'cosyne2022'])
        assert args.competition == 'cosyne2022'
