"""
Process plugin submissions (data, metrics, benchmarks, models) and score models on benchmarks.
"""
from typing import List


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


def run_scoring(models: List[str], benchmarks: List[str]):
    """
    Run the `models` on the `benchmarks`, and write resulting scores to the database.
    """
    pass  # TODO @Katherine
