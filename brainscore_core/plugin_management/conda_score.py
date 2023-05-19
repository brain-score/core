import os
import pickle
import subprocess
from pathlib import Path
from typing import Callable, Union

from brainscore_core.metrics import Score
from .environment_manager import EnvironmentManager
from .import_plugin import installation_preference


class CondaScore(EnvironmentManager):
    """ run scoring in conda environment """

    def __init__(self, library_path: Path, model_identifier: str, benchmark_identifier: str):
        super(CondaScore, self).__init__()

        self.library_path = library_path.parent
        self.model = model_identifier
        self.benchmark = benchmark_identifier
        self.env_name = f'{self.model}_{self.benchmark}'
        self.script_path = f'{Path(__file__).parent}/conda_score.sh'

    def __call__(self):
        self.result = self.score_in_env()
        return self.consume_score(self.library_path, self.env_name)

    def score_in_env(self) -> 'subprocess.CompletedProcess[bytes]':
        """
        calls bash script to create conda environment, then hands execution back to score()
        """
        run_command = f"bash {self.script_path} \
                {self.library_path.parent} {self.library_path.name} \
                {self.model} {self.benchmark} {self.env_name} {self.envs_dir}"

        completed_process = self.run_in_env(run_command)
        completed_process.check_returncode()

        return completed_process

    @staticmethod
    def consume_score(library_path: Path, env_name: str) -> Score:
        score_path = CondaScore._score_path(library_path, env_name)
        with open(score_path, 'rb') as f:
            score = pickle.load(f)
            os.remove(score_path)
            return score

    @staticmethod
    def save_score(score: Score, library_path: Path, env_name: str):
        score_path = CondaScore._score_path(library_path.parent, env_name)
        with open(score_path, 'wb') as f:
            pickle.dump(score, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _score_path(library_path: Path, env_name: str) -> Path:
        """
        File path inside the given library for sub-process to write the score to, and for us to read back in.
        Since separate scoring runs typically occur in separately installed libraries,
        the score paths between them will be different and not conflict.
        There is still a remaining race condition when two scoring runs are started within the same library,
        which is currently not handled.
        """
        return library_path.parent / f'conda_score--{env_name}.pkl'


def wrap_score(library_path: Union[str, Path], model_identifier: str, benchmark_identifier: str,
               score_function: Callable[[str, str], Score], conda_active: bool):
    """
    If :meth:`~brainscore_core.plugin_management.import_plugin.installation_preference` is not `newenv`,
    simply run the `score_function` and return its result directly.
    If :meth:`~brainscore_core.plugin_management.import_plugin.installation_preference` is `newenv`,
    create a new environment with model and benchmark dependencies installed,
    run the `score_function` inside that environment and return the result.
    Communication between this environment and the scoring environment occurs by the scoring environment storing
    the result in a temporary file (:meth:`~brainscore_core.plugin_management.conda_score.CondaScore._score_path`)
    and then this environment reading the file in again to return its contents.

    :param library_path: path to the domain-specific library module, e.g. `/home/user/brainscore_language/__init__.py`.
        Must have a `score` method accepting parameters `model_identifier` and `benchmark_identifier`
        and call :meth:`~brainscore_core.plugin_management.conda_score.CondaScore.save_score`,
        preferably via :meth:`~brainscore_core.plugin_management.conda_score.wrap_score`.
    """
    if installation_preference() == 'newenv' and not conda_active:
        conda_score = CondaScore(Path(library_path), model_identifier, benchmark_identifier)
        result = conda_score()
    else:
        result = score_function(model_identifier, benchmark_identifier)
        CondaScore.save_score(result, Path(library_path), f'{model_identifier}_{benchmark_identifier}')

    return result
