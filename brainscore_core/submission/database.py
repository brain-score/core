import json
import logging
from datetime import datetime
from peewee import PostgresqlDatabase, SqliteDatabase, DoesNotExist
from peewee import Proxy
from pybtex.database.input import bibtex
from typing import Union

from brainscore_core.metrics import Score as ScoreObject
from brainscore_core.benchmarks import Benchmark
from brainscore_core.submission.database_models import Submission, Model, BenchmarkType, BenchmarkInstance, Reference, \
    Score
from brainscore_core.submission.utils import get_secret

database_proxy = Proxy()
logger = logging.getLogger(__name__)


def connect_db(db_secret):
    if 'sqlite3' not in db_secret:
        secret = get_secret(db_secret)
        db_configs = json.loads(secret)
        postgres = PostgresqlDatabase(db_configs['dbInstanceIdentifier'],
                                      **{'host': db_configs['host'], 'port': 5432,
                                         'user': db_configs['username'], 'password': db_configs['password']})
        database_proxy.initialize(postgres)
    else:
        sqlite = SqliteDatabase(db_secret)
        database_proxy.initialize(sqlite)


def submissionentry_from_meta(jenkins_id: str, user_id: str) -> Submission:
    now = datetime.now()
    submission = Submission.create(id=jenkins_id, submitter=user_id, timestamp=now, status='running')
    return submission


def modelentry_from_model(model, model_identifier: str, submission: Submission) -> Model:
    model_entry, created = Model.get_or_create(name=model_identifier, owner=submission.submitter,
                                               defaults={'public': submission.public,
                                                         'submission': submission,
                                                         'competition': submission.competition_submission})
    if hasattr(model, 'bibtex') and created:  # model entry was just created and we can add bibtex
        bibtex_string = model.bibtex
        reference = reference_from_bibtex(bibtex_string)
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
        # probe the ceiling for an estimate of the error
        ceiling_error = None
        error_retrievors = [lambda ceiling: ceiling.sel(aggregation='error').item(),
                            lambda ceiling: ceiling.attrs['error']]
        for retrievor in error_retrievors:
            try:
                ceiling_error = retrievor(ceiling)
                break
            except Exception:  # if we can't find an error estimate, ignore
                pass
        bench_inst.ceiling_error = ceiling_error
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


def score_to_database(score: ScoreObject, entry: Score):
    if not hasattr(score, 'ceiling'):  # many engineering benchmarks do not have a primate ceiling
        # only store raw (unceiled) value
        entry.score_raw = score.sel(aggregation='center').item(0)
    else:  # score has a ceiling. Store ceiled as well as raw value
        assert score.raw.sel(aggregation='center') is not None
        entry.score_raw = score.raw.sel(aggregation='center').item(0)
        entry.score_ceiled = score.sel(aggregation='center').item(0)
        entry.error = score.sel(aggregation='error').item(0)
    entry.save()
