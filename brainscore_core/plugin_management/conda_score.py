import os
import pickle
import subprocess
import tempfile
from pathlib import Path

from brainscore_core.metrics import Score
from .environment_manager import EnvironmentManager

SCORE_PATH = tempfile.NamedTemporaryFile(delete=False).name
""" file for sub-process to write the score to, and for us to read back in """


class CondaScore(EnvironmentManager):
    """ run scoring in conda environment """

    def __init__(self, library_root: str, model_identifier: str, benchmark_identifier: str):
        """
        :param library_root: the domain-specific library, e.g. `brainscore_language`.
            Must have a `score` method accepting parameters `model_identifier` and `benchmark_identifier`
            and call :meth:`~brainscore_core.plugin_management.CondaScore.save_score`
        """
        super(CondaScore, self).__init__()

        self.library_root = library_root
        self.model = model_identifier
        self.benchmark = benchmark_identifier
        self.env_name = f'{self.model}_{self.benchmark}'
        self.script_path = f'{Path(__file__).parent}/conda_score.sh'

    def __call__(self):
        self.result = self.score_in_env()
        return self.read_score()

    def score_in_env(self) -> 'subprocess.CompletedProcess[bytes]':
        """ 
        calls bash script to create conda environment, then
        hands execution back to score()
        """
        run_command = f"bash {self.script_path} \
                {self.library_root} {self.model} {self.benchmark} {self.env_name}"

        completed_process = self.run_in_env(run_command)
        completed_process.check_returncode()

        return completed_process

    @staticmethod
    def read_score():
        with open(SCORE_PATH, 'rb') as f:
            score = pickle.load(f)
            os.remove(SCORE_PATH)
            return score

    @staticmethod
    def save_score(score: Score):
        with open(SCORE_PATH, 'wb') as f:
            pickle.dump(score, f, pickle.HIGHEST_PROTOCOL)
