"""
Process plugin submissions (data, metrics, benchmarks, models) and score models on benchmarks.
"""
import logging
import os
import json
import random
import smtplib
import string
import traceback
import yaml
from abc import ABC
from argparse import ArgumentParser
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Tuple, Union

import requests
from requests.auth import HTTPBasicAuth
from brainscore_core import Benchmark, Score
from brainscore_core.submission import database_models

# User management imports
from brainscore_core.submission.database import (
    connect_db, uid_from_email, email_from_uid, submissionentry_from_meta)

# Model imports  
from brainscore_core.submission.database import (
    modelentry_from_model, create_model_meta_entry, safe_create_model_meta_entry,
    get_model_metadata_by_identifier, get_model_with_metadata,
    public_model_identifiers)

# Benchmark imports
from brainscore_core.submission.database import (
    benchmarkinstance_from_benchmark, create_benchmark_meta_entry,
    get_benchmark_metadata_by_identifier, get_benchmark_with_metadata,
    public_benchmark_identifiers)

# Scoring imports
from brainscore_core.submission.database import update_score

logger = logging.getLogger(__name__)


class UserManager:
    """
    Retrieve user information (UID from email / email from UID)
    Create new user from email address
    Send email to user
    """

    def __init__(self, db_secret: str):
        logger.info(f"Connecting to db using secret '{db_secret}")
        connect_db(db_secret=db_secret)

    def _generate_temp_pass(self, length: int) -> str:
        chars = string.ascii_letters + string.digits + string.punctuation
        temp_pass = ''.join(random.choice(chars) for i in range(length))
        return temp_pass

    def create_new_user(self, user_email: str):
        signup_url = 'https://www.brain-score.org/signup'
        temp_pass = self._generate_temp_pass(length=10)
        try:
            response = requests.get(signup_url)
            cookies = response.cookies
            csrf_token = [x.value for x in cookies][0]
            data = f'email={user_email}&a=1&csrfmiddlewaretoken={csrf_token}&password1={temp_pass}&password2={temp_pass}&is_from_pr'
            response = requests.post(signup_url,
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                     cookies=cookies, data=data)
            assert response.status_code == 200, f"Response error: {response.status_code}"
        except Exception as e:
            logging.error(f'Could not create Brain-Score account for {user_email} because of {e}')
            raise e

    def get_uid(self, author_email: str) -> int:
        """
        Returns the Brain-Score user ID associated with a given email address.
        If no user ID exists, creates a new account, then returns user ID.
        """
        uid = uid_from_email(author_email)
        if not uid:
            self.create_new_user(author_email)
            uid = uid_from_email(author_email)
            assert uid
        return uid

    def send_user_email(self, uid: int, subject: str, body: str, sender: str, password: str):
        """ Send user an email. """
        user_email = email_from_uid(uid)
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = "Brain-Score"
        msg['To'] = user_email

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
                smtp_server.login(sender, password)
                smtp_server.sendmail(sender, user_email, msg.as_string())

            print(f"Email sent to {user_email}")

        except Exception as e:
            logging.error(f'Could not send email to {user_email} because of {e}')
            raise e


class DomainPlugins(ABC):
    """
    Interface for domain-specific model + benchmark loaders and the score method.
    """

    def load_model(self, model_identifier: str):
        raise NotImplementedError()

    def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
        raise NotImplementedError()

    def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
        raise NotImplementedError()

class MetadataEndpoint:
    """
    Endpoint for processing metadata submissions. This endpoint is used when only
    metadata.yml files have changed and no full scoring run is triggered.
      - Connects to the database.
      - Loads and validates the metadata.yml file from a plugin directory.
      - Iterates through each plugin entry (e.g. each model) and updates db.
    """

    def __init__(self, domain_plugins: DomainPlugins, db_secret: str):
        self.domain_plugins = domain_plugins
        logger.info(f"Connecting to db using secret '{db_secret}'")
        connect_db(db_secret=db_secret)

    def process_metadata(self, plugin_dir: str, plugin_type: str, domain: str = None) -> dict:
        """
        Process the metadata file for a plugin.

        :param plugin_dir: The directory containing the metadata.yml file.
        :param plugin_type: The type of plugin ('models' or 'benchmarks').
        :param domain: The domain ('vision' or 'language'). Required for benchmarks.
        :return: A dictionary mapping model identifiers to their updated ModelMeta records.
        """
        metadata_path = os.path.join(plugin_dir, "metadata.yml")
        with open(metadata_path, 'r') as f:
            data = yaml.safe_load(f)

        if plugin_type not in data:
            raise ValueError(f"Expected top-level key '{plugin_type}' in metadata file.")

        plugin_metadata = data[plugin_type]
        results = {}
        for identifier, metadata in plugin_metadata.items():
            logger.info(f"Updating metadata for plugin '{identifier}'")
            if plugin_type == 'models':
                # overwrite any existing entry with new metadata
                result = safe_create_model_meta_entry(identifier, metadata)
            elif plugin_type == 'benchmarks':
                if domain is None:
                    raise ValueError("Domain parameter is required for benchmark metadata processing")
                
                # Load benchmark in ENDPOINT layer (same pattern as RunScoringEndpoint)
                try:
                    benchmark = self.domain_plugins.load_benchmark(identifier)
                    if benchmark is None:
                        raise ValueError(f"Failed to load benchmark '{identifier}'")
                    logger.info(f"Loaded benchmark '{identifier}' in endpoint layer")
                except Exception as e:
                    raise ValueError(f"Could not load benchmark '{identifier}': {e}")
                
                # Pass loaded benchmark object to DATABASE layer
                result = create_benchmark_meta_entry(benchmark, domain, metadata)
            else:
                raise NotImplementedError(f"Plugin type not implemented yet: '{plugin_type}'")
            results[identifier] = result
            logger.info(f"Updated metadata for plugin '{identifier}': {json.dumps(metadata)}")
        return results

    def __call__(self, plugin_dir: str, plugin_type: str, domain: str = None) -> None:
        try:
            results = self.process_metadata(plugin_dir, plugin_type, domain)
            logger.info("Metadata processing completed successfully.")
            for plugin_id, record in results.items():
                logger.info(f"Plugin '{plugin_id}' updated in db.")
        except Exception as e:
            logger.error(f"Error processing metadata: {e}", exc_info=True)
            raise e


class RunScoringEndpoint:
    ALL_PUBLIC = "all_public"  """ key to reference models or benchmarks to all public entries """

    def __init__(self, domain_plugins: DomainPlugins, db_secret: str):
        self.domain_plugins = domain_plugins
        logger.info(f"Connecting to db using secret '{db_secret}'")
        connect_db(db_secret=db_secret)

    def __call__(self, domain: str, jenkins_id: int, model_identifier: str, benchmark_identifier: str,
                 user_id: int, model_type: str, public: bool, competition: Union[None, str]):
        """
        Run the `model_identifier` on the `benchmark_identifier`, and write resulting score to the database.

        Explanation of subset of parameters:
        :param domain: "language" or "vision"
        :param model_identifier: a string of a model identifier
        :param benchmark_identifier: a string of a model identifier
        """
        # setup entry for this submission
        submission_entry = submissionentry_from_meta(jenkins_id=jenkins_id, user_id=user_id, model_type=model_type)
        is_run_successful = True

        logger.debug(f"Scoring {model_identifier} on {benchmark_identifier}")

        try:
            self._score_model_on_benchmark(model_identifier=model_identifier,
                                           benchmark_identifier=benchmark_identifier,
                                           submission_entry=submission_entry, domain=domain,
                                           public=public, competition=competition)
        except Exception as e:
            is_run_successful = False
            logging.error(
                f'Could not run model {model_identifier} on benchmark {benchmark_identifier} because of {e}',
                exc_info=True)

        # finalize status of submission
        submission_status = 'successful' if is_run_successful else 'failure'
        if getattr(submission_entry, 'status', "successful") != 'failure':
            submission_entry.status = submission_status
        logger.info(f'Submission is stored as {submission_status}')
        submission_entry.save()

    def _score_model_on_benchmark(self, model_identifier: str, benchmark_identifier: str,
                                  submission_entry: database_models.Submission, domain: str,
                                  public: bool, competition: Union[None, str]):
        # TODO: the following is somewhat ugly because we're afterwards loading model and benchmark again
        #  in the `score` method.
        logger.info(f'Model database entry')
        model = self.domain_plugins.load_model(model_identifier)
        model_entry = modelentry_from_model(model_identifier=model_identifier, domain=domain,
                                            submission=submission_entry, public=public, competition=competition,
                                            bibtex=model.bibtex if hasattr(model, 'bibtex') else None)

        logger.info(f'Benchmark database entry')
        benchmark = self.domain_plugins.load_benchmark(benchmark_identifier)
        benchmark_entry = benchmarkinstance_from_benchmark(benchmark, domain=domain)

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
            score_entry.save()
            logger.warning('A score entry exists but does not have a score value, so we run it again')

        # run actual scoring mechanism
        try:
            score_result = self.domain_plugins.score(
                model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)
            score_entry.end_timestamp = datetime.now()
            # store in database
            logger.info(f'Score from running {model_identifier} on {benchmark_identifier}: {score_result}')
            update_score(score_result, score_entry)
        except Exception as e:
            stacktrace = traceback.format_exc()
            error_message = f'Model {model_identifier} could not run on benchmark {benchmark_identifier}: ' \
                            f'{repr(e)}. \n{stacktrace}'
            error_message = shorten_text(error_message, max_length=database_models.Score.comment.max_length)
            score_entry.comment = error_message
            score_entry.save()
            raise e


def resolve_models(domain: str, models: Union[List[str], str]) -> List[str]:
    """
    Identify the set of models by resolving `models` to the list of public models if `models` is `ALL_PUBLIC`
    :param domain: "language" or "vision"
    :param models: either a list of model identifiers or the string
        :attr:`~brainscore_core.submission.endpoints.RunScoringEndpoint.ALL_PUBLIC` to select all public models
    """
    if models == RunScoringEndpoint.ALL_PUBLIC:
        models = public_model_identifiers(domain)
    return models


def resolve_benchmarks(domain: str, benchmarks: Union[List[str], str]) -> List[str]:
    """
    Identify the set of benchmarks by resolving `benchmarks` to the list of public benchmarks if `benchmarks` is `ALL_PUBLIC`
    :param domain: "language" or "vision"
    :param benchmarks: either a list of benchmark identifiers or the string
        :attr:`~brainscore_core.submission.endpoints.RunScoringEndpoint.ALL_PUBLIC` to select all public benchmarks
    """
    if benchmarks == RunScoringEndpoint.ALL_PUBLIC:
        benchmarks = public_benchmark_identifiers(domain)
    return benchmarks


def resolve_models_benchmarks(domain: str, args_dict: Dict[str, Union[str, List]]):
    """
    Identify the set of model/benchmark pairs to score by resolving `new_models` and `new_benchmarks` in the user input.
    Prints the names of models and benchmarks to stdout.
    :param domain: "language" or "vision"
    :param args_dict: a map containing `new_models`, `new_benchmarks`, and `specified_only`, specifying which the
        model/benchmark names to be resolved.
    """
    benchmarks, models = retrieve_models_and_benchmarks(args_dict)

    benchmark_ids = resolve_benchmarks(domain=domain, benchmarks=benchmarks)
    model_ids = resolve_models(domain=domain, models=models)

    print("BS_NEW_MODELS=" + " ".join(model_ids))
    print("BS_NEW_BENCHMARKS=" + " ".join(benchmark_ids))
    return model_ids, benchmark_ids


def shorten_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    spacer = '[...]'
    early_stop = (max_length // 2)
    part1 = text[:early_stop - len(spacer)]
    part2 = text[-(max_length - early_stop):]
    return part1 + spacer + part2


def noneable_string(val: str) -> Union[None, str]:
    """ For argparse """
    if val is None or val == 'None':
        return None
    return val


def make_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('jenkins_id', type=int,
                        help='The id of the current jenkins run')
    parser.add_argument('--public', default=False, action="store_true",
                        help='Public (or private) submission?')
    parser.add_argument('--competition', type=noneable_string, nargs='?', default=None,
                        help='Name of competition for which submission is being scored')
    parser.add_argument('--user_id', type=int, nargs='?', default=None,
                        help='ID of submitting user in the postgres DB')
    parser.add_argument('--author_email', type=str, nargs='?', default=None,
                        help='email associated with PR author GitHub username')
    parser.add_argument('--specified_only', default=False, action="store_true",
                        help='Only score the plugins specified by new_models and new_benchmarks')
    parser.add_argument('--new_models', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted models to score on all benchmarks')
    parser.add_argument('--new_benchmarks', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted benchmarks on which to score all models')
    parser.add_argument('--fn', type=str, nargs='?', default='run_scoring',
                        choices=['run_scoring', 'resolve_models_benchmarks'],
                        help='The endpoint method to run. `run_scoring` to score `new_models` on `new_benchmarks`, '
                             'or `resolve_models_benchmarks` to respond with a list of models and benchmarks to score.')
    return parser


# used by domain libraries in `score_new_plugins.yml`
def call_jenkins(plugin_info: Union[str, Dict[str, Union[List[str], str]]]):
    """
    Triggered when changes are merged to the GitHub repository, if those changes affect benchmarks or models.
    Starts run to score models on benchmarks (`run_scoring`).
    """
    jenkins_base = "http://www.brain-score-jenkins.com:8080"
    jenkins_user = os.environ['JENKINS_USER']
    jenkins_token = os.environ['JENKINS_TOKEN']
    jenkins_trigger = os.environ['JENKINS_TRIGGER']
    jenkins_job = "dev_score_plugins"

    url = f'{jenkins_base}/job/{jenkins_job}/buildWithParameters?token={jenkins_trigger}'

    if isinstance(plugin_info, str):
        # Check if plugin_info is a String object, in which case JSON-deserialize it into Dict
        plugin_info = json.loads(plugin_info)

    payload = {k: v for k, v in plugin_info.items() if plugin_info[k]}
    try:
        auth_basic = HTTPBasicAuth(username=jenkins_user, password=jenkins_token)
        requests.get(url, params=payload, auth=auth_basic)
    except Exception as e:
        print(f'Could not initiate Jenkins job because of {e}')


def retrieve_models_and_benchmarks(args_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """ prepares parameters for the `run_scoring_endpoint`. """
    new_models = _get_ids(args_dict, 'new_models')
    new_benchmarks = _get_ids(args_dict, 'new_benchmarks')
    if args_dict['specified_only']:
        assert len(new_models) > 0, "No models specified"
        assert len(new_benchmarks) > 0, "No benchmarks specified"
        models = new_models
        benchmarks = new_benchmarks
    else:
        if new_models and new_benchmarks:
            models = RunScoringEndpoint.ALL_PUBLIC
            benchmarks = RunScoringEndpoint.ALL_PUBLIC
        elif new_benchmarks:
            models = RunScoringEndpoint.ALL_PUBLIC
            benchmarks = new_benchmarks
        elif new_models:
            models = new_models
            benchmarks = RunScoringEndpoint.ALL_PUBLIC
        else:
            raise ValueError("Unexpected condition")
    return benchmarks, models


def _get_ids(args_dict: Dict[str, Union[str, List]], key: str) -> Union[List, str, None]:
    return args_dict[key] if key in args_dict else None


def get_user_id(email: str, db_secret: str) -> int:
    user_manager = UserManager(db_secret=db_secret)
    user_id = user_manager.get_uid(email)
    return user_id


def send_email_to_submitter(uid: int, domain: str, db_secret: str, pr_number: str,
                            mail_username: str, mail_password: str):
    """ Send submitter an email if their web-submitted PR fails. """
    subject = "Brain-Score submission failed"
    body = "Your Brain-Score submission did not pass checks. " \
           "Please review the test results and update the PR at " \
           f"https://github.com/brain-score/{domain}/pull/{pr_number} " \
           "or send in an updated submission via the website."
    user_manager = UserManager(db_secret=db_secret)
    return user_manager.send_user_email(uid=uid, subject=subject, body=body,
                                        sender=mail_username, password=mail_password)
