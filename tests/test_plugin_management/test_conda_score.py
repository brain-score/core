import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.random import RandomState

from brainio.assemblies import DataAssembly
from brainscore_core.metrics import Score
from brainscore_core.plugin_management.conda_score import CondaScore


def _create_dummy_score() -> Score:
    score = Score(.8)
    score.attrs['model_identifier'] = 'distilgpt2'
    score.attrs['benchmark_identifier'] = 'Pereira2018.243sentences-linear'
    score.attrs['raw'] = DataAssembly(RandomState(1).standard_normal(30 * 25).reshape((30, 25)), coords={
        'neuroid_id': ('neuroid', np.arange(30)), 'network': ('neuroid', ['language'] * 30),
        'split': np.arange(25)}, dims=['neuroid', 'split'])
    ceiling = Score(0.35378928)
    ceiling.attrs['raw'] = Score(RandomState(1).standard_normal(30), coords={
        'neuroid_id': ('neuroid', np.arange(30)), 'network': ('neuroid', ['language'] * 30)}, dims=['neuroid'])
    score.attrs['ceiling'] = ceiling
    return score


def test_save_and_consume_score():
    score = _create_dummy_score()
    library_path = Path(tempfile.mkdtemp()) / '__init__.py'
    env_name = 'dummy-model_dummy-benchmark'
    expected_score_path = library_path.parent.parent / f'conda_score--{env_name}.pkl'
    CondaScore.save_score(score, library_path=library_path, env_name=env_name)
    assert expected_score_path.is_file()
    result = CondaScore.consume_score(library_path=library_path.parent, env_name=env_name)
    assert not expected_score_path.is_file()
    assert score == result


class TestCondaScoreInEnv:
    dummy_container_dirpath = Path(tempfile.mkdtemp("-brainscore-dummy"))

    def setup_method(self):
        sys.path.append(str(self.dummy_container_dirpath))
        local_resource = Path(__file__).parent / 'test_conda_score__brainscore_dummy'  # contains dummy-library scripts
        for local_file in local_resource.iterdir():
            shutil.copy(local_file, self.dummy_container_dirpath)

    def teardown_method(self):
        shutil.rmtree(self.dummy_container_dirpath)
        sys.path.remove(str(self.dummy_container_dirpath))

    @pytest.mark.memory_intense
    def test_score_in_env(self):
        scorer = CondaScore(library_path=self.dummy_container_dirpath / 'brainscore_dummy.py',
                            model_identifier='dummy-model', benchmark_identifier='dummy-benchmark')
        score = scorer()
        assert score == 0.42
