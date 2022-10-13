import logging

import os
import pytest
from datetime import datetime

from brainscore_core.submission.database import connect_db
from brainscore_core.submission.database_models import Score, BenchmarkType
from tests.test_submission import clear_schema, init_user

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
        ref = reference_from_bibtex(bibtex)
        assert isinstance(ref, database_models.Reference)
        assert ref.url == 'https://doi.org/10.1038/nn.3402'
        assert ref.year == '2013'
        assert ref.author is not None
        ref2 = reference_from_bibtex(bibtex)
        assert ref2.id == ref.id

    class MockBenchmark(BenchmarkBase):
        def __init__(self):
            dummy_ceiling = Score(0.6)
            dummy_ceiling.attrs['error'] = 0.1
            super(TestSubmission.MockBenchmark, self).__init__(
                identifier='dummy', ceiling=dummy_ceiling, version=0, parent='neural')

    def test_get_benchmark_instance_no_parent(self):
        benchmark = TestSubmission.MockBenchmark()
        instance = benchmarkinstance_from_benchmark(benchmark)
        type = database_models.BenchmarkType.get(identifier=instance.benchmark)
        assert instance.ceiling == 0.6
        assert instance.ceiling_error == 0.1
        assert not type.parent

    def test_get_benchmark_instance_existing_parent(self):
        # initially create the parent to see if the benchmark properly links to it
        database_models.BenchmarkType.create(identifier='neural', order=3)
        benchmark = TestSubmission.MockBenchmark()
        instance = benchmarkinstance_from_benchmark(benchmark)
        assert instance.benchmark.parent.identifier == 'neural'

    def get_test_models(self):
        submission = database_models.Submission.create(id=33, submitter=1, timestamp=datetime.now(),
                                                       model_type='BaseModel', status='running')
        model_instances = []
        model_instances.append(
            database_models.Model.create(name='alexnet', owner=submission.submitter, public=False,
                                         submission=submission))
        return model_instances, submission


def init_benchmark_parents():
    BenchmarkType.create(identifier='neural', order=0)
    BenchmarkType.create(identifier='V1', parent='neural', order=0)
    BenchmarkType.create(identifier='V2', parent='neural', order=1)
    BenchmarkType.create(identifier='V4', parent='neural', order=2)
    BenchmarkType.create(identifier='IT', parent='neural', order=3)

    BenchmarkType.create(identifier='behavior', order=1)


@pytest.mark.memory_intense
@pytest.mark.private_access
# @pytest.mark.skip(reason="This test case only works locally due to some weird openmind error")
@pytest.mark.parametrize('database', ['brainscore-ohio-test'])  # test database
def test_evaluation(database, tmpdir):
    connect_db(database)
    clear_schema()
    init_user()
    working_dir = str(tmpdir.mkdir("sub"))
    config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
    run_evaluation(config_dir, working_dir, 33, database, models=['alexnet'],
                   benchmarks=['dicarlo.MajajHong2015.IT-pls'])
    scores = Score.select().dicts()
    assert len(scores) == 1
    # If comment is none the score was successfully stored, otherwise there would be an error message there
    assert scores[0]['comment'] is None
