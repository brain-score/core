import json
import logging
from datetime import datetime
from typing import List, Union

from peewee import PostgresqlDatabase, SqliteDatabase, DoesNotExist
from pybtex.database.input import bibtex

from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score as ScoreObject
from brainscore_core.submission.database_models import database_proxy, \
    Submission, Model, BenchmarkType, BenchmarkInstance, Reference, Score
from brainscore_core.submission.utils import get_secret

logger = logging.getLogger(__name__)


def connect_db(db_secret):
    if 'sqlite3' not in db_secret:
        print('sqlite not in db_secret')
        secret = get_secret(db_secret)
        db_configs = json.loads(secret)
        postgres = PostgresqlDatabase(db_configs['dbInstanceIdentifier'],
                                      **{'host': db_configs['host'], 'port': 5432,
                                         'user': db_configs['username'], 'password': db_configs['password']})
        database_proxy.initialize(postgres)
        print(f"db_proxy: {database_proxy}")
    else:
        print('sqlite is in db_secret')
        sqlite = SqliteDatabase(db_secret)
        database_proxy.initialize(sqlite)


def submissionentry_from_meta(jenkins_id: int, user_id: int, model_type: str) -> Submission:
    now = datetime.now()
    submission = Submission.create(id=jenkins_id, submitter=user_id, model_type=model_type,
                                   timestamp=now, status='running')
    return submission


def public_model_identifiers() -> List[str]:
    print("in public_model_identifiers")
    entries = Model.select().where(Model.public == True)
    print(f"entries: {entries}")
    identifiers = [entry.name for entry in entries]
    print(f"identifiers: {identifiers}")
    return identifiers


def public_benchmark_identifiers() -> List[str]:
    entries = BenchmarkType.select().where(BenchmarkType.visible == True)
    identifiers = [entry.identifier for entry in entries]
    return identifiers


def modelentry_from_model(model_identifier: str, public: bool, competition: Union[None, str],
                          submission: Submission,
                          bibtex: Union[None, str] = None) -> Model:
    model_entry, created = Model.get_or_create(name=model_identifier, owner=submission.submitter,
                                               defaults={'public': public,
                                                         'submission': submission,
                                                         'competition': competition})
    if bibtex and created:  # model entry was just created and we can add bibtex
        reference = reference_from_bibtex(bibtex)
        model_entry.reference = reference
        model_entry.save()
    return model_entry


def benchmarkinstance_from_benchmark(benchmark: Benchmark) -> BenchmarkInstance:
    benchmark_identifier = benchmark.identifier
    benchmark_type, created = BenchmarkType.get_or_create(identifier=benchmark_identifier, defaults=dict(order=999))
    if created:
        # store parent
        try:
            parent = BenchmarkType.get(identifier=benchmark.parent)
            benchmark_type.parent = parent
            benchmark_type.save()
        except DoesNotExist:
            logger.warning(f'Could not connect benchmark {benchmark_identifier} to parent {benchmark.parent} '
                           f'since parent does not exist')
        # store reference
        if hasattr(benchmark, 'bibtex') and benchmark.bibtex is not None:
            bibtex_string = benchmark.bibtex
            ref = reference_from_bibtex(bibtex_string)
            if ref:
                benchmark_type.reference = ref
                benchmark_type.save()

    # process instance
    bench_inst, created = BenchmarkInstance.get_or_create(benchmark=benchmark_type, version=benchmark.version)
    if created:
        # the version has changed and the benchmark instance was not yet in the database
        ceiling = benchmark.ceiling
        bench_inst.ceiling = ceiling.item()
        bench_inst.ceiling_error = _retrieve_score_error(ceiling)
        bench_inst.save()
    return bench_inst


def reference_from_bibtex(bibtex_string: str) -> Union[Reference, None]:
    def parse_bib(bibtex_str):
        bib_parser = bibtex.Parser()
        entry = bib_parser.parse_string(bibtex_str)
        entry = entry.entries
        assert len(entry) == 1
        entry = list(entry.values())[0]
        return entry

    try:
        entry = parse_bib(bibtex_string)
        ref, created = Reference.get_or_create(url=entry.fields['url'],
                                               defaults={'bibtex': bibtex_string,
                                                         'author': entry.persons["author"][0].last()[0],
                                                         'year': entry.fields['year']})
        return ref
    except Exception:
        logger.exception('Could not load reference from bibtex string')
        return None


def update_score(score: ScoreObject, entry: Score):
    if 'ceiling' not in score.attrs:  # many engineering benchmarks do not have a primate ceiling
        # only store raw (unceiled) value
        entry.score_raw = _retrieve_score_center(score)
    else:  # score has a ceiling. Store ceiled as well as raw value
        score_raw = _retrieve_score_center(score.raw)
        entry.score_raw = score_raw
        entry.score_ceiled = _retrieve_score_center(score)
    entry.error = _retrieve_score_error(score)
    entry.save()


def _retrieve_score_center(score: ScoreObject) -> float:
    """
    Deal with multiple formats of storing the score,
    i.e. a single scalar in the score object versus an aggregation dimension.
    """
    if hasattr(score, 'aggregation'):
        return score.sel(aggregation='center').item()
    return score.item()


def _retrieve_score_error(score: ScoreObject) -> Union[float, None]:
    """
    Deal with multiple formats of storing the score,
    i.e. an error attribute in the score object versus an aggregation dimension.
    Returns None if no error was found
    """
    if hasattr(score, 'aggregation'):
        return score.sel(aggregation='error').item()
    if 'error' in score.attrs:
        return score.attrs['error']
    return None
