"""
Process plugin submissions (data, metrics, benchmarks, models) and score models on benchmarks.
"""
import traceback
import logging
from abc import ABC
from datetime import datetime
from email.mime.text import MIMEText
import os
import random
import requests
from requests.auth import HTTPBasicAuth
import smtplib
import string
from typing import List, Union, Dict

from brainscore_core import Benchmark, Score
from brainscore_core.submission import database_models
from brainscore_core.submission.database import connect_db, modelentry_from_model, \
    submissionentry_from_meta, benchmarkinstance_from_benchmark, update_score, \
    public_model_identifiers, public_benchmark_identifiers, uid_from_email, email_from_uid

logger = logging.getLogger(__name__)


class UserManager:
    """
    Returns the Brain-Score user ID associated with a given email address.
    If no user ID exists, creates a new account, then returns user ID.
    """

    def __init__(self, domain: str, author_email: str, db_secret: str):
        self.domain = domain
        self.author_email = author_email
        logger.info(f"Connecting to db using secret '{db_secret}")
        connect_db(db_secret=db_secret)

    def __call__(self):
        uid = uid_from_email(author_email=self.author_email)
        if not uid:
            self._create_new_user(domain=self.domain, user_email=self.author_email)
            uid = uid_from_email(author_email=self.author_email)
            assert uid
        return uid

    def _generate_temp_pass(self, length:int) -> str:
        chars = string.ascii_letters + string.digits + string.punctuation
        temp_pass = ''.join(random.choice(chars) for i in range(length))
        return temp_pass

    def _create_new_user(self, domain:str, user_email:str):
        signup_url = f'http://www.brain-score.org/signup/{domain}'
        temp_pass = self._generate_temp_pass(length=10)
        try:
            response = requests.get(signup_url, cookies=cookies)
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


class RunScoringEndpoint:
    ALL_PUBLIC = "all_public"  """ key to reference models or benchmarks to all public entries """

    def __init__(self, domain_plugins: DomainPlugins, db_secret: str):
        self.domain_plugins = domain_plugins
        logger.info(f"Connecting to db using secret '{db_secret}'")
        connect_db(db_secret=db_secret)

    def __call__(self, domain: str, jenkins_id: int, models: List[str], benchmarks: List[str],
                 user_id: int, model_type: str, public: bool, competition: Union[None, str]):
        """
        Run the `models` on the `benchmarks`, and write resulting score to the database.

        Explanation of subset of parameters:
        :param domain: "language" or "vision"
        :param models: either a list of model identifiers or the string
            :attr:`~brainscore_core.submission.endpoints.RunScoringEndpoint.ALL_PUBLIC` to select all public models
        :param benchmarks: either a list of benchmark identifiers or the string
            :attr:`~brainscore_core.submission.endpoints.RunScoringEndpoint.ALL_PUBLIC` to select all public benchmarks
        """
        # setup entry for this entire submission
        submission_entry = submissionentry_from_meta(jenkins_id=jenkins_id, user_id=user_id, model_type=model_type)
        entire_submission_successful = True

        # resolve settings
        if models == self.ALL_PUBLIC:
            models = public_model_identifiers(domain)
        if benchmarks == self.ALL_PUBLIC:
            benchmarks = public_benchmark_identifiers(domain)

        logger.info(f"Models: {models}")
        logger.info(f"Benchmarks: {benchmarks}")

        for model_identifier in models:
            for benchmark_identifier in benchmarks:
                logger.debug(f"Scoring {model_identifier} on {benchmark_identifier}")
                # TODO: I am worried about reloading models inside the loop. E.g. a keras model where layer names are
                #  automatic and will be consecutive from previous layers
                #  (e.g. on first load layers are [1, 2, 3], on second load layers are [4, 5, 6])
                #  which can lead to issues with layer assignment
                try:
                    self._score_model_on_benchmark(model_identifier=model_identifier,
                                                   benchmark_identifier=benchmark_identifier,
                                                   submission_entry=submission_entry, domain=domain,
                                                   public=public, competition=competition)
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


def send_user_email(uid: int, domain: str, pr_number: str):
    """ Send user an email if their web-submitted PR fails. """
    user_email = email_from_uid(uid)

    body = f"Your Brain-Score submission did not pass checks. Please review the test results and update the PR at https://github.com/brain-score/{domain}/pull/{pr_number} or send in an updated submission via the website."
    msg = MIMEText(body)
    msg['Subject'] = "Brain-Score submission failed"
    msg['From'] = "Brain-Score"
    msg['To'] = recipient

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    
    print(f"Email sent to {user_email}")


def shorten_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    spacer = '[...]'
    early_stop = (max_length // 2)
    part1 = text[:early_stop - len(spacer)]
    part2 = text[-(max_length - early_stop):]
    return part1 + spacer + part2
