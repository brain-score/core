import shutil
import sys
import tempfile
from pathlib import Path
import importlib.util

import numpy as np
import pytest
from numpy.random import RandomState

from brainio.assemblies import DataAssembly
from brainscore_core.metrics import Score
from brainscore_core.plugin_management.conda_score import CondaScore


def find_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml.
    First tries to find it relative to the current file, then falls back to
    finding it relative to the installed package location.
    """
    # First try: look for pyproject.toml in parent directories of current file
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:  # Stop at root directory
        if (current_dir / 'pyproject.toml').exists():
            return current_dir
        current_dir = current_dir.parent
    
    # Second try: look for pyproject.toml relative to installed package
    spec = importlib.util.find_spec('brainscore_core')
    if spec is not None and spec.origin is not None:
        package_dir = Path(spec.origin).parent
        while package_dir != package_dir.parent:  # Stop at root directory
            if (package_dir / 'pyproject.toml').exists():
                return package_dir
            package_dir = package_dir.parent
    
    raise RuntimeError("Could not find project root directory (containing pyproject.toml)")


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
    def setup_method(self):
        # Create temp directory with a specific name for easier debugging
        self.dummy_container_dirpath = Path(tempfile.mkdtemp("-brainscore-dummy"))
        print(f"Created temp directory at: {self.dummy_container_dirpath}")
        
        # Create the library directory structure
        self.library_dir = self.dummy_container_dirpath / "brainscore_dummy"
        self.library_dir.mkdir(exist_ok=True)
        
        # Copy all files from the resources directory
        local_resource = Path(__file__).parent / 'test_conda_score__brainscore_dummy'
        print(f"Copying files from: {local_resource}")
        
        # Copy all files from the resources directory
        for file in local_resource.glob('*'):
            shutil.copy(file, self.library_dir / file.name)
        
        # Add the temp directory to Python path
        sys.path.append(str(self.dummy_container_dirpath))
        print(f"Added to Python path: {self.dummy_container_dirpath}")
        
        # Verify files exist
        assert (self.library_dir / 'pyproject.toml').exists(), "pyproject.toml not found"
        assert (self.library_dir / '__init__.py').exists(), "__init__.py not found"
        assert (self.library_dir / 'brainscore_dummy.py').exists(), "brainscore_dummy.py not found"
        
        # Print directory structure for debugging
        print("Created package structure:")
        for path in self.library_dir.rglob("*"):
            print(f"  {path.relative_to(self.library_dir)}")
            
        # Create a requirements.txt in the library directory to install local brainscore-core
        project_root = find_project_root()
        print(f"Found project root at: {project_root}")
        with open(self.library_dir / 'requirements.txt', 'w') as f:
            f.write(f"-e {project_root}\n")  # Install local package in editable mode
            print(f"Created requirements.txt pointing to: {project_root}")

    def teardown_method(self):
        shutil.rmtree(self.dummy_container_dirpath)
        sys.path.remove(str(self.dummy_container_dirpath))

    @pytest.mark.memory_intense
    def test_score_in_env(self):
        # Use the library directory as the library path
        library_path = self.library_dir / '__init__.py'
        print(f"Using library path: {library_path}")
        
        scorer = CondaScore(library_path=library_path,
                          model_identifier='dummy-model', 
                          benchmark_identifier='dummy-benchmark')
        score = scorer()
        assert score == 0.42
