import json
import logging
from datetime import datetime
from peewee import PostgresqlDatabase, SqliteDatabase, DoesNotExist
from pybtex.database.input import bibtex
from typing import List, Union

from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score as ScoreObject
from brainscore_core.submission.database_models import (database_proxy, \
    Submission, Model, User, BenchmarkType, BenchmarkInstance, Reference, Score, ModelMeta, BenchmarkMeta, PeeweeBase,
    BenchmarkDataMeta, BenchmarkMetricMeta, BenchmarkStimuliMeta)
from brainscore_core.submission.utils import get_secret

logger = logging.getLogger(__name__)


def connect_db(db_secret):
    if 'sqlite3' not in db_secret:
        secret = get_secret(db_secret)
        db_configs = json.loads(secret)
        postgres = PostgresqlDatabase(db_configs['dbInstanceIdentifier'],
                                      **{'host': db_configs['host'], 'port': 5432,
                                         'user': db_configs['username'], 'password': db_configs['password']})
        database_proxy.initialize(postgres)
        database_proxy.connect()
    else:
        sqlite = SqliteDatabase(db_secret)
        database_proxy.initialize(sqlite)
        all_orm_models = PeeweeBase.__subclasses__()
        for orm_model in all_orm_models:
            orm_model._meta.schema = None  # do not use schema for sqlite
        sqlite.create_tables(all_orm_models)


def uid_from_email(author_email: str) -> Union[None, int]:
    """
    Retrieve the user id belonging to an email.
    If no user is found, returns None.
    """
    entries = User.select().where(User.email == author_email)
    if not entries:
        return None
    assert len(entries) == 1, f"Expected exactly one user with email {author_email}, but found {len(entries)}"
    user_id = [entry.id for entry in entries][0]
    return user_id


def email_from_uid(user_id: int) -> Union[None, str]:
    """
    Retrieve the email belonging to a user id.
    If no user is found, returns None.
    """
    entries = User.select().where(User.id == user_id)
    if not entries:
        return None
    assert len(entries) == 1, f"Expected exactly one user with id {user_id}, but found {len(entries)}"
    user_id = [entry.email for entry in entries][0]
    return user_id


def submissionentry_from_meta(jenkins_id: int, user_id: int, model_type: str) -> Submission:
    now = datetime.now()
    submission = Submission.create(jenkins_id=jenkins_id, submitter=user_id, model_type=model_type,
                                   timestamp=now, status='running')
    return submission


def public_model_identifiers(domain: str) -> List[str]:
    entries = Model.select().where(Model.public & (Model.domain == domain))
    identifiers = [entry.name for entry in entries]
    return identifiers


def public_benchmark_identifiers(domain: str) -> List[str]:
    entries = BenchmarkInstance.select(BenchmarkType.identifier).join(BenchmarkType).where(
        (BenchmarkType.domain == domain) & (BenchmarkType.visible)
    )
    identifiers = [entry.benchmark.identifier for entry in entries]
    return identifiers


def modelentry_from_model(model_identifier: str, public: bool, competition: Union[None, str],
                          submission: Submission, domain: str, model_meta: Union[None, dict],
                          bibtex: Union[None, str] = None) -> Model:
    model_entry, created = Model.get_or_create(name=model_identifier,
                                               defaults={
                                                   'owner': submission.submitter,
                                                   'domain': domain,
                                                   'public': public,
                                                   'submission': submission,
                                                   'competition': competition})
    if bibtex and created:  # model entry was just created and we can add bibtex
        reference = reference_from_bibtex(bibtex)
        model_entry.reference = reference
        model_entry.save()
    # if model_meta is not None:
    #     meta_entry = create_model_meta_entry(model_identifier, model_meta)
    return model_entry


def create_model_meta_entry(model_identifier: str, metadata: dict) -> ModelMeta:
    """
    Given a model identifier and a metadata dict, get or create a ModelMeta record.
    The metadata dict can include keys such as architecture, model_family, etc.
    """
    # using get here in case certain fields don't exist
    defaults = {
        'architecture': metadata.get('architecture'),
        'model_family': metadata.get('model_family'),
        'total_parameter_count': metadata.get('total_parameter_count'),
        'trainable_parameter_count': metadata.get('trainable_parameter_count'),
        'total_layers': metadata.get('total_layers'),
        'trainable_layers': metadata.get('trainable_layers'),
        'model_size_MB': metadata.get('model_size_MB'),
        'training_dataset': metadata.get('training_dataset'),
        'task_specialization': metadata.get('task_specialization'),
        'brainscore_link': metadata.get('brainscore_link'),
        'huggingface_link': metadata.get('huggingface_link'),
        'extra_notes': metadata.get('extra_notes')
    }
    try:  # if model exists, overwrite all fields
        modelmeta = ModelMeta.get(ModelMeta.identifier == model_identifier)
        for key, value in defaults.items():
            setattr(modelmeta, key, value)
        modelmeta.save()
        logger.info(f"Updated existing ModelMeta record for {model_identifier}")
    except ModelMeta.DoesNotExist:  # otherwise create a new entry
        # filter out fields that don't exist in the ModelMeta model
        valid_defaults = {}
        for key, value in defaults.items():
            try:
                if hasattr(ModelMeta, key):
                    valid_defaults[key] = value
            except Exception:
                logger.warning(f"Field '{key}' not found in ModelMeta schema - skipping")

        modelmeta = ModelMeta.create(identifier=model_identifier, **valid_defaults)
        logger.info(f"Created new ModelMeta record for {model_identifier}")
    return modelmeta


def create_benchmark_meta_entry(benchmark_identifier: str, metadata: dict):
    """
    Given a benchmark identifier and a metadata dict, create metadata entries for the benchmark
    and update the BenchmarkInstance table to reference these entries.
    """
    logger.info(f"Processing benchmark metadata for {benchmark_identifier}")

    # Find all BenchmarkInstance entries for this benchmark identifier
    benchmark_instances = BenchmarkInstance.select().join(BenchmarkType).where(
        BenchmarkType.identifier == benchmark_identifier
    )

    if not benchmark_instances:
        logger.warning(f"No BenchmarkInstance found for {benchmark_identifier}. "
                       "Metadata will still be created but not linked.")

    # Process stimulus_set metadata
    stimuli_meta_id = None
    if 'stimulus_set' in metadata:
        stimulus_data = metadata['stimulus_set']
        stimuli_meta = BenchmarkStimuliMeta.create(
            num_stimuli=stimulus_data.get('num_stimuli'),
            datatype=stimulus_data.get('datatype'),
            stimuli_subtype=stimulus_data.get('stimuli_subtype'),
            total_size_mb=stimulus_data.get('total_size_MB'),  # Case conversion from YAML to DB
            brainscore_link=stimulus_data.get('brainscore_link'),
            extra_notes=stimulus_data.get('extra_notes')
        )
        stimuli_meta_id = stimuli_meta.id

    # Process data metadata
    data_meta_id = None
    if 'data' in metadata:
        data_info = metadata['data']
        data_meta = BenchmarkDataMeta.create(
            benchmark_type=data_info.get('benchmark_type'),
            task=data_info.get('task'),
            region=data_info.get('region'),
            hemisphere=data_info.get('hemisphere'),
            num_recording_sites=data_info.get('num_recording_sites'),
            duration_ms=data_info.get('duration_ms'),
            species=data_info.get('species'),
            datatype=data_info.get('datatype'),
            num_subjects=data_info.get('num_subjects'),
            pre_processing=data_info.get('pre_processing'),
            brainscore_link=data_info.get('brainscore_link'),
            extra_notes=data_info.get('extra_notes')
        )
        data_meta_id = data_meta.id

    # Process metric metadata
    metric_meta_id = None
    if 'metric' in metadata:
        metric_info = metadata['metric']
        metric_meta = BenchmarkMetricMeta.create(
            type=metric_info.get('type'),
            reference=metric_info.get('reference'),
            public=metric_info.get('public'),
            brainscore_link=metric_info.get('brainscore_link'),
            extra_notes=metric_info.get('extra_notes')
        )
        metric_meta_id = metric_meta.id

    # Update all BenchmarkInstance entries with new metadata IDs
    for instance in benchmark_instances:
        update_fields = {}
        if stimuli_meta_id is not None:
            update_fields['stimuli_meta'] = stimuli_meta_id
        if data_meta_id is not None:
            update_fields['data_meta'] = data_meta_id
        if metric_meta_id is not None:
            update_fields['metric_meta'] = metric_meta_id

        if update_fields:
            query = BenchmarkInstance.update(**update_fields).where(BenchmarkInstance.id == instance.id)
            query.execute()
            logger.info(f"Updated BenchmarkInstance {instance.id} with metadata IDs")

    logger.info(f"Successfully processed metadata for benchmark {benchmark_identifier}")

    # Return the created metadata IDs for reference
    return {
        'stimuli_meta_id': stimuli_meta_id,
        'data_meta_id': data_meta_id,
        'metric_meta_id': metric_meta_id
    }


def benchmarkinstance_from_benchmark(benchmark: Benchmark, domain: str) -> BenchmarkInstance:
    benchmark_identifier = benchmark.identifier
    benchmark_type, created = BenchmarkType.get_or_create(identifier=benchmark_identifier,
                                                          defaults=dict(domain=domain, order=999, owner_id=2))
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
    if 'comment' in score.attrs:
        entry.comment = score.attrs['comment']
    logger.debug(f"updating raw score: {entry.score_raw}, ceiled score: {entry.score_ceiled}, error: {entry.error}, "
                 f"comment: {entry.comment}")
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
