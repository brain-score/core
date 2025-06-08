import json
import logging
import time
from datetime import datetime
from peewee import PostgresqlDatabase, SqliteDatabase, DoesNotExist
from pybtex.database.input import bibtex
from typing import List, Union, Tuple

from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score as ScoreObject
from brainscore_core.submission.database_models import (database_proxy, \
    Submission, Model, User, BenchmarkType, BenchmarkInstance, Reference, Score, ModelMeta, PeeweeBase)
from brainscore_core.submission.utils import get_secret
from brainscore_core.plugin_management.metadata_utils import (get_existing_meta_ids, process_stimuli_metadata,
                                                              process_data_metadata, process_metric_metadata,
                                                              update_benchmark_instances, BenchmarkMetaResult)

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
                          submission: Submission, domain: str, bibtex: Union[None, str] = None) -> Model:
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
    return model_entry


def safe_create_model_meta_entry(model_identifier: str, metadata: dict, max_attempts: int = 10, sleep_seconds: int = 60) -> ModelMeta:
    """
    Retry `create_model_meta_entry` until the model exists, or a timeout is reached.
    """
    for attempt in range(max_attempts):
        try:
            return create_model_meta_entry(model_identifier, metadata)
        except ValueError as e:
            if "not found" in str(e):
                logger.warning(
                    f"[Retry {attempt + 1}/{max_attempts}] Model '{model_identifier}' not found yet – retrying in {sleep_seconds}s..."
                )
                time.sleep(sleep_seconds)
            else:
                raise  # Let unexpected errors through
    raise TimeoutError(f"Timed out waiting for model '{model_identifier}' to exist after {max_attempts} attempts.")


def create_model_meta_entry(model_identifier: str, metadata: dict) -> ModelMeta:
    """
    Given a model identifier and a metadata dict, get or create a ModelMeta record.
    The metadata dict can include keys such as architecture, model_family, etc.
    
    Uses the exact same model-finding logic as scoring by calling modelentry_from_model().
    This ensures perfect consistency between scoring and metadata systems.
    """
    # Use the exact same function that scoring uses to find/get the model
    # We need to provide dummy values for scoring-specific parameters
    try:
        # Get a dummy user and create a temporary submission for the lookup
        dummy_user = User.select().first()
        if not dummy_user:
            raise ValueError("No users found in database - cannot perform model lookup")
        
        # Create a temporary submission for the model lookup
        temp_submission = Submission.create(
            jenkins_id=0,  # Dummy jenkins_id for metadata lookup
            submitter=dummy_user,
            model_type="metadata_lookup",
            status="temporary",
            timestamp=datetime.now()
        )
        
        # Use the EXACT same function that scoring uses
        model_entry = modelentry_from_model(
            model_identifier=model_identifier,
            domain="unknown",  # Default domain - will use existing model's domain if it exists
            submission=temp_submission,
            public=False,  # Default public status - will use existing model's status if it exists  
            competition=None,
            bibtex=None
        )
        
        # Clean up the temporary submission
        temp_submission.delete_instance()
        
        logger.info(f"Found model for metadata using scoring logic: ID={model_entry.id}, identifier='{model_identifier}'")
        
    except Exception as e:
        # Clean up temp submission if it was created
        try:
            temp_submission.delete_instance()
        except:
            pass
        raise ValueError(f"Model with identifier '{model_identifier}' not found. "
                        f"Model must exist before creating metadata. Error: {e}")
    
    # using get here in case certain fields don't exist
    defaults = {
        'architecture': metadata.get('architecture'),
        'model_family': metadata.get('model_family'),
        'total_parameter_count': metadata.get('total_parameter_count'),
        'trainable_parameter_count': metadata.get('trainable_parameter_count'),
        'total_layers': metadata.get('total_layers'),
        'trainable_layers': metadata.get('trainable_layers'),
        'model_size_mb': metadata.get('model_size_mb'),
        'training_dataset': metadata.get('training_dataset'),
        'task_specialization': metadata.get('task_specialization'),
        'brainscore_link': metadata.get('brainscore_link'),
        'huggingface_link': metadata.get('huggingface_link'),
        'extra_notes': metadata.get('extra_notes'),
        'runnable': metadata.get('runnable')
    }
    
    try:  # if model metadata exists, overwrite all fields
        modelmeta = ModelMeta.get(ModelMeta.model == model_entry.id)
        for key, value in defaults.items():
            setattr(modelmeta, key, value)
        modelmeta.save()
        logger.info(f"Updated existing ModelMeta record for model_id {model_entry.id} (identifier: {model_identifier})")
    except ModelMeta.DoesNotExist:  # otherwise create a new entry
        # filter out fields that don't exist in the ModelMeta model
        valid_defaults = {}
        for key, value in defaults.items():
            try:
                if hasattr(ModelMeta, key):
                    valid_defaults[key] = value
            except Exception:
                logger.warning(f"Field '{key}' not found in ModelMeta schema - skipping")

        modelmeta = ModelMeta.create(model=model_entry, **valid_defaults)
        logger.info(f"Created new ModelMeta record for model_id {model_entry.id} (identifier: {model_identifier})")
    return modelmeta


def get_model_metadata_by_identifier(model_identifier: str) -> Union[ModelMeta, None]:
    """
    Retrieve ModelMeta record by model identifier string.
    Returns None if model or metadata doesn't exist.
    """
    try:
        model_entry = Model.get(Model.name == model_identifier)
        return ModelMeta.get(ModelMeta.model == model_entry.id)
    except (Model.DoesNotExist, ModelMeta.DoesNotExist):
        return None


def get_model_with_metadata(model_identifier: str) -> Union[Tuple[Model, ModelMeta], Tuple[Model, None]]:
    """
    Retrieve Model and its associated ModelMeta record by identifier.
    Returns (Model, ModelMeta) if both exist, (Model, None) if only model exists.
    Raises Model.DoesNotExist if model doesn't exist.

    Only used during `text_alexnet_consistency_integration.py`
    """
    model_entry = Model.get(Model.name == model_identifier)
    try:
        metadata = ModelMeta.get(ModelMeta.model == model_entry.id)
        return model_entry, metadata
    except ModelMeta.DoesNotExist:
        return model_entry, None


def get_benchmark_metadata_by_identifier(benchmark_identifier: str) -> Union[BenchmarkMetaResult, None]:
    """
    Retrieve benchmark metadata by benchmark identifier string.
    Returns BenchmarkMetaResult if benchmark exists, None if benchmark doesn't exist.
    """
    try:
        # Find benchmark instance - use first instance if multiple versions exist
        benchmark_instance = BenchmarkInstance.select().join(BenchmarkType).where(
            BenchmarkType.identifier == benchmark_identifier
        ).first()
        
        if not benchmark_instance:
            return None
            
        # Collect metadata IDs from the instance
        metadata_ids = {
            'stimuli_meta_id': benchmark_instance.stimuli_meta_id if hasattr(benchmark_instance, 'stimuli_meta_id') else None,
            'data_meta_id': benchmark_instance.data_meta_id if hasattr(benchmark_instance, 'data_meta_id') else None,
            'metric_meta_id': benchmark_instance.metric_meta_id if hasattr(benchmark_instance, 'metric_meta_id') else None
        }
        
        # Return BenchmarkMetaResult object
        return BenchmarkMetaResult(
            benchmark_identifier=benchmark_identifier,
            **metadata_ids
        )
        
    except Exception:
        return None


def get_benchmark_with_metadata(benchmark_identifier: str) -> Union[Tuple[BenchmarkInstance, BenchmarkMetaResult], Tuple[BenchmarkInstance, None]]:
    """
    Retrieve BenchmarkInstance and its associated metadata by identifier.
    Returns (BenchmarkInstance, BenchmarkMetaResult) if both exist, (BenchmarkInstance, None) if only benchmark exists.
    Raises BenchmarkType.DoesNotExist if benchmark doesn't exist.

    Only used during `text_alexnet_consistency_integration.py`
    """
    # Find benchmark instance - use first instance if multiple versions exist
    benchmark_instance = BenchmarkInstance.select().join(BenchmarkType).where(
        BenchmarkType.identifier == benchmark_identifier
    ).first()
    
    if not benchmark_instance:
        raise BenchmarkType.DoesNotExist(f"Benchmark with identifier '{benchmark_identifier}' not found")
        
    # Try to get metadata
    metadata = get_benchmark_metadata_by_identifier(benchmark_identifier)
    return benchmark_instance, metadata


def create_benchmark_meta_entry(benchmark: Benchmark, domain: str, metadata: dict):
    """
    Given a loaded benchmark object, domain, and a metadata dict, create metadata entries for the benchmark
    and update the BenchmarkInstance table to reference these entries.
        
    :param benchmark: The loaded benchmark object (same as scoring uses)
    :param domain: The domain (e.g., "vision", "language") 
    :param metadata: Dictionary containing benchmark metadata
    :return: A BenchmarkMetaResult object with the IDs of the created metadata entries
    """
    benchmark_identifier = benchmark.identifier
    logger.info(f"Processing benchmark metadata for {benchmark_identifier}")

    try:
        # Same benchmark-finding logic as scoring uses
        benchmark_instance = benchmarkinstance_from_benchmark(benchmark, domain=domain)
        logger.info(f"Found benchmark instance using scoring logic: {benchmark_instance.id} for '{benchmark_identifier}' version {benchmark.version}")
        
    except Exception as e:
        raise ValueError(f"Could not find benchmark instance for '{benchmark_identifier}' (version {benchmark.version}). "
                        f"Error: {e}")

    # Process metadata for this specific benchmark instance (same as scoring uses)
    benchmark_instances = [benchmark_instance]  # Single instance from scoring logic
    existing_meta_ids = get_existing_meta_ids(benchmark_instances)

    # process each metadata type
    meta_ids = {}
    if 'stimulus_set' in metadata:
        meta_ids['stimuli_meta_id'] = process_stimuli_metadata(
            metadata['stimulus_set'], existing_meta_ids['stimuli_meta_id'])
    if 'data' in metadata:
        meta_ids['data_meta_id'] = process_data_metadata(
            metadata['data'], existing_meta_ids['data_meta_id'])
    if 'metric' in metadata:
        meta_ids['metric_meta_id'] = process_metric_metadata(
            metadata['metric'], existing_meta_ids['metric_meta_id'])

    update_benchmark_instances(benchmark_instances, meta_ids)
    logger.info(f"Successfully processed metadata for benchmark {benchmark_identifier}")

    return BenchmarkMetaResult(
        benchmark_identifier=benchmark_identifier,
        stimuli_meta_id=meta_ids.get('stimuli_meta_id'),
        data_meta_id=meta_ids.get('data_meta_id'),
        metric_meta_id=meta_ids.get('metric_meta_id')
    )


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
