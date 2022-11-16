import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Union

from brainscore_core.metrics import Score
from .environment_manager import EnvironmentManager
from .import_plugin import installation_preference

SCORE_PATH = tempfile.NamedTemporaryFile(delete=False).name
""" file for sub-process to write the score to, and for us to read back in """


class CondaScore(EnvironmentManager):
    """ run scoring in conda environment """

    def __init__(self, library_path: Path, model_identifier: str, benchmark_identifier: str):
        """
        :param library_path: path to the domain-specific library module, e.g. `/home/user/brainscore_language`.
            Must have a `score` method accepting parameters `model_identifier` and `benchmark_identifier`
            and call :meth:`~brainscore_core.plugin_management.conda_score.CondaScore.save_score`,
            preferably via :meth:`~brainscore_core.plugin_management.conda_score.wrap_score`.
        """
        super(CondaScore, self).__init__()

        self.library_path = library_path
        self.model = model_identifier
        self.benchmark = benchmark_identifier
        self.env_name = f'{self.model}_{self.benchmark}'
        self.script_path = f'{Path(__file__).parent}/conda_score.sh'

    def __call__(self):
        self.result = self.score_in_env()
        return self.read_score()

    def score_in_env(self) -> 'subprocess.CompletedProcess[bytes]':
        """
        calls bash script to create conda environment, then hands execution back to score()
        """
        run_command = f"bash {self.script_path} \
                {self.library_path.parent} {self.library_path.name} {self.model} {self.benchmark} {self.env_name}"

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


def wrap_score(library_root: Union[str, Path], model_identifier: str, benchmark_identifier: str,
               score_function: Callable[[str, str], Score]):
    if installation_preference() == 'newenv':
        conda_score = CondaScore(Path(library_root), model_identifier, benchmark_identifier)
        result = conda_score()
    else:
        result = score_function(model_identifier, benchmark_identifier)
        CondaScore.save_score(result)

    return result
