from collections import namedtuple

import logging
import numpy as np
import pytest

from brainscore_core import Score as ScoreObject
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.submission.database import connect_db, reference_from_bibtex, benchmarkinstance_from_benchmark, \
    submissionentry_from_meta, modelentry_from_model, update_score
from brainscore_core.submission.database_models import Score, BenchmarkType, Reference
from tests.test_submission import clear_schema, init_user

logger = logging.getLogger(__name__)

database = 'brainscore-ohio-test'  # test database

SAMPLE_BIBTEX = """@Article{Freeman2013,
                                author={Freeman, Jeremy and Ziemba, Corey M. and Heeger, David J. 
                                        and Simoncelli, Eero P. and Movshon, J. Anthony},
                                title={A functional and perceptual signature of the second visual area in primates},
                                journal={Nature Neuroscience},
                                year={2013},
                                month={Jul},
                                day={01},
                                volume={16},
                                number={7},
                                pages={974-981},
                                issn={1546-1726},
                                doi={10.1038/nn.3402},
                                url={https://doi.org/10.1038/nn.3402}
                                }"""


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestEntries:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(database)
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database entries')
        init_user()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_submission(self):
        entry = submissionentry_from_meta(jenkins_id=123, user_id=1, model_type='artificial_subject')
        assert entry.status == 'running'

    def test_model_no_bibtex(self):
        model_dummy = namedtuple('Model', field_names=[])()
        submission_entry = submissionentry_from_meta(jenkins_id=123, user_id=1, model_type='base_model')
        entry = modelentry_from_model(model_dummy, model_identifier='dummy', public=False, competition='cosyne2022',
                                      submission=submission_entry)
        with pytest.raises(Exception):
            entry.reference

    def test_model_with_bibtex(self):
        model_dummy = namedtuple('Model', field_names=['bibtex'])(bibtex=SAMPLE_BIBTEX)
        submission_entry = submissionentry_from_meta(jenkins_id=123, user_id=1, model_type='base_model')
        entry = modelentry_from_model(model_dummy, model_identifier='dummy', public=True, competition=None,
                                      submission=submission_entry)
        assert entry.reference.year == '2013'

    class MockBenchmark(BenchmarkBase):
        def __init__(self):
            dummy_ceiling = ScoreObject(0.6)
            dummy_ceiling.attrs['error'] = 0.1
            super(TestEntries.MockBenchmark, self).__init__(
                identifier='dummy', ceiling=dummy_ceiling, version=0, parent='neural')

    def test_benchmark_instance_no_parent(self):
        benchmark = TestEntries.MockBenchmark()
        instance = benchmarkinstance_from_benchmark(benchmark)
        type = BenchmarkType.get(identifier=instance.benchmark)
        assert instance.ceiling == 0.6
        assert instance.ceiling_error == 0.1
        assert not type.parent

    def test_benchmark_instance_existing_parent(self):
        # initially create the parent to see if the benchmark properly links to it
        BenchmarkType.create(identifier='neural', order=3)
        benchmark = TestEntries.MockBenchmark()
        instance = benchmarkinstance_from_benchmark(benchmark)
        assert instance.benchmark.parent.identifier == 'neural'

    def test_reference(self):
        ref = reference_from_bibtex(SAMPLE_BIBTEX)
        assert isinstance(ref, Reference)
        assert ref.url == 'https://doi.org/10.1038/nn.3402'
        assert ref.year == '2013'
        assert ref.author is not None
        ref2 = reference_from_bibtex(SAMPLE_BIBTEX)
        assert ref2.id == ref.id

    def _create_score_entry(self):
        model_dummy = namedtuple('Model', field_names=[])()
        submission_entry = submissionentry_from_meta(jenkins_id=123, user_id=1, model_type='brain_model')
        model_entry = modelentry_from_model(model_dummy, model_identifier='dummy', public=True, competition=None,
                                            submission=submission_entry)
        benchmark_entry = benchmarkinstance_from_benchmark(TestEntries.MockBenchmark())
        entry, created = Score.get_or_create(benchmark=benchmark_entry, model=model_entry)
        return entry

    def test_score_no_ceiling(self):
        score = ScoreObject([.123, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        entry = self._create_score_entry()
        update_score(score, entry)
        assert entry.score_ceiled is None
        assert entry.error is None
        assert entry.score_raw == .123

    def test_score_with_ceiling(self):
        score = ScoreObject([.42, .1], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score.attrs['raw'] = ScoreObject([.336, .08], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score.attrs['ceiling'] = ScoreObject(.8)
        entry = self._create_score_entry()
        update_score(score, entry)
        assert entry.score_ceiled == .42
        assert entry.error == .1
        assert entry.score_raw == .336
