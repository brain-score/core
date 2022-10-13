"""
Process plugin submissions (data, metrics, benchmarks, models) and score models on benchmarks.
"""
import logging
from brainscore_language import load_model, load_benchmark, score
from datetime import datetime
from typing import List

from brainscore_core.submission import database_models
from brainscore_core.submission.database import connect_db, modelentry_from_model, submissionentry_from_meta, \
    benchmarkinstance_from_benchmark, update_score

logger = logging.getLogger(__name__)


def process_zip_submission(zip_filepath: str):
    """
    Triggered when a zip file is submitted via the website.
    Opens a pull request on GitHub with the plugin contents of the zip file.
    The merge of this PR will potentially trigger `process_github_submission`.
    """
    pass  # TODO @Katherine


def process_github_submission():
    """
    Triggered when changed are merged to the GitHub repository, if those changes affect benchmarks or models.
    Starts parallel runs to score models on benchmarks (`run_scoring`).
    """
    pass  # TODO @Katherine


def run_scoring(models: List[str], benchmarks: List[str],
                submission_meta: dict,
                db_secret: str):
    """
    Run the `models` on the `benchmarks`, and write resulting scores to the database.
    """
    logger.info(f"Connecting to db using secret '{db_secret}'")
    connect_db(db_secret)

    # setup entry for this entire submission
    submission_entry = submissionentry_from_meta(**submission_meta)
    entire_submission_successful = True

    # iterate over all model-benchmark pairs
    for model_identifier in models:
        for benchmark_identifier in benchmarks:
            logger.info(f"Scoring {model_identifier} on {benchmark_identifier}")
            # TODO: I am worried about reloading models inside the loop. E.g. a keras model where layer names are
            #  automatic and will be consecutive from previous layers
            #  (e.g. on first load layers are [1, 2, 3], on second load layers are [4, 5, 6])
            #  which can lead to issues with layer assignment
            try:
                _score_model_on_benchmark(model_identifier=model_identifier,
                                          benchmark_identifier=benchmark_identifier,
                                          submission_entry=submission_entry)
            except Exception as e:
                entire_submission_successful = False
                logging.error(
                    f'Could not run model {model_identifier} on benchmark {benchmark_identifier} because of {e}',
                    exc_info=True)

    # finalize status of submission
    submission_status = 'successful' if entire_submission_successful else 'failure'
    submission_entry.status = submission_status
    logger.info(f'Submission is stored as {submission_status}')
    submission_entry.save()


def _score_model_on_benchmark(model_identifier: str, benchmark_identifier: str,
                              submission_entry: database_models.Submission):
    # TODO: the following is somewhat ugly because we're afterwards loading model and benchmark again
    #  in the `score` method.
    logger.info(f'Model database entry')
    model = load_model(model_identifier)
    model_entry = modelentry_from_model(model=model, model_identifier=model_identifier,
                                        submission=submission_entry)
    logger.info(f'Benchmark database entry')
    benchmark = load_benchmark(benchmark_identifier)
    benchmark_entry = benchmarkinstance_from_benchmark(benchmark)

    # Check if the model is already scored on the benchmark
    start_timestamp = datetime.now()
    score_entry, created = database_models.Score.get_or_create(benchmark=benchmark_entry, model=model_entry,
                                                               defaults={'start_timestamp': start_timestamp, })
    if not created and score_entry.score_raw is not None:
        logger.warning(f'A score for model {model_identifier} and benchmark {benchmark_identifier} already exists')
        return

    if not created:  # previous score entry exists, but no score was stored
        score_entry.start_timestamp = datetime.now()
        score_entry.comment = None
        logger.warning('An entry already exists but was not evaluated successful, we rerun!')

    # run actual scoring mechanism
    score_result = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)
    score_entry.end_timestamp = datetime.now()

    # store in database
    logger.info(f'Score from running {model_identifier} on {benchmark_identifier}: {score_result}')
    update_score(score_result, score_entry)
