import logging
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.submission import database_models
from brainscore_core.submission.configuration import object_decoder
from brainscore_core.submission.database import connect_db
from brainscore_core.submission.evaluation import get_reference, database_instance_for_benchmark
from brainscore_core.submission.repository import extract_zip_file, find_submission_directory
from tests.test_submission.test_db import clear_schema, init_user

logger = logging.getLogger(__name__)
database = 'brainscore-ohio-test'  # test database


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestSubmission:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(database)
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database')
        init_user()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_get_reference(self):
        bibtex = """@Article{Freeman2013,
                                author={Freeman, Jeremy
                                and Ziemba, Corey M.
                                and Heeger, David J.
                                and Simoncelli, Eero P.
                                and Movshon, J. Anthony},
                                title={A functional and perceptual signature of the second visual area in primates},
                                journal={Nature Neuroscience},
                                year={2013},
                                month={Jul},
                                day={01},
                                volume={16},
                                number={7},
                                pages={974-981},
                                abstract={The authors examined neuronal responses in V1 and V2 to synthetic texture stimuli that replicate higher-order statistical dependencies found in natural images. V2, but not V1, responded differentially to these textures, in both macaque (single neurons) and human (fMRI). Human detection of naturalistic structure in the same images was predicted by V2 responses, suggesting a role for V2 in representing natural image structure.},
                                issn={1546-1726},
                                doi={10.1038/nn.3402},
                                url={https://doi.org/10.1038/nn.3402}
                                }
                            """
        ref = get_reference(bibtex)
        assert isinstance(ref, database_models.Reference)
        assert ref.url == 'https://doi.org/10.1038/nn.3402'
        assert ref.year == '2013'
        assert ref.author is not None
        ref2 = get_reference(bibtex)
        assert ref2.id == ref.id

    class MockBenchmark(BenchmarkBase):
        def __init__(self):
            dummy_ceiling = Score(0.6)
            dummy_ceiling.attrs['error'] = 0.1
            super(TestSubmission.MockBenchmark, self).__init__(
                identifier='dummy', ceiling=dummy_ceiling, version=0, parent='neural')

    def test_get_benchmark_instance_no_parent(self):
        benchmark = TestSubmission.MockBenchmark()
        instance = database_instance_for_benchmark(benchmark)
        type = database_models.BenchmarkType.get(identifier=instance.benchmark)
        assert instance.ceiling == 0.6
        assert instance.ceiling_error == 0.1
        assert not type.parent

    def test_get_benchmark_instance_existing_parent(self):
        # initially create the parent to see if the benchmark properly links to it
        database_models.BenchmarkType.create(identifier='neural', order=3)
        benchmark = TestSubmission.MockBenchmark()
        instance = database_instance_for_benchmark(benchmark)
        assert instance.benchmark.parent.identifier == 'neural'

    def get_test_models(self):
        submission = database_models.Submission.create(id=33, submitter=1, timestamp=datetime.now(),
                                                       model_type='BaseModel', status='running')
        model_instances = []
        model_instances.append(
            database_models.Model.create(name='alexnet', owner=submission.submitter, public=False,
                                         submission=submission))
        return model_instances, submission


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestConfig:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(database)
        clear_schema()
        init_user()

    @classmethod
    def teardown_class(cls):
        logger.info('Connect to database')
        clear_schema()

    def test_base_config(self):
        config = {"model_type": "BaseModel",
                  "user_id": 1,
                  "public": "False",
                  "competition": "cosyne2022"}
        submission_config = object_decoder(config, 'work_dir', 'config_path', 'db_secret', 33)
        assert submission_config.db_secret == 'db_secret'
        assert submission_config.work_dir == 'work_dir'
        assert submission_config.jenkins_id == 33
        assert submission_config.submission is not None
        assert not submission_config.public

    def test_resubmit_config(self):
        model = database_models.Model.create(id=19, name='alexnet', public=True, submission=33, owner=1)
        config = {
            "model_ids": [model.id],
            "user_id": 1,
            "competition": "cosyne2022"
        }
        submission_config = object_decoder(config, 'work_dir', 'config_path', 'db_secret', 33)
        assert len(submission_config.submission_entries) == 1
        assert len(submission_config.models) == 1
        assert submission_config.models[0].name == 'alexnet'


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestRepository:
    working_dir = None
    config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))

    @classmethod
    def setup_class(cls):
        connect_db(database)
        clear_schema()
        init_user()

    @classmethod
    def tear_down_class(cls):
        clear_schema()

    def setup_method(self):
        tmpdir = tempfile.mkdtemp()
        TestRepository.working_dir = tmpdir

    def tear_down_method(self):
        os.rmdir(TestRepository.working_dir)

    def test_extract_zip_file(self):
        path = extract_zip_file(33, TestRepository.config_dir, TestRepository.working_dir)
        assert str(path) == f'{TestRepository.working_dir}/candidate_models'

    def test_find_correct_dir(self):
        Path(f'{TestRepository.working_dir}/.temp').touch()
        Path(f'{TestRepository.working_dir}/_MACOS').touch()
        Path(f'{TestRepository.working_dir}/candidate_models').touch()
        dir = find_submission_directory(TestRepository.working_dir)
        assert dir == 'candidate_models'
        with pytest.raises(Exception):
            Path(f'{TestRepository.working_dir}/candidate_models2').touch()
            find_submission_directory(TestRepository.working_dir)
