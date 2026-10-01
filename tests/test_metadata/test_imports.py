"""Check that metadata stays independent of scoring imports."""

import subprocess
import sys
import unittest


class ImportTests(unittest.TestCase):
    def test_metadata_import_does_not_load_scoring_dependencies(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import brainscore_core.metadata
import brainscore_core.metadata.review
assert not {'numpy', 'pandas', 'xarray', 'brainscore_core.metrics',
            'brainscore_core.benchmarks'} & sys.modules.keys()
""",
            ],
            check=True,
        )

    def test_public_scoring_exports_keep_their_identity(self):
        from brainscore_core import Benchmark, Metric, Score
        from brainscore_core.benchmarks import Benchmark as OriginalBenchmark
        from brainscore_core.metrics import (
            Metric as OriginalMetric,
            Score as OriginalScore,
        )

        self.assertIs(Benchmark, OriginalBenchmark)
        self.assertIs(Metric, OriginalMetric)
        self.assertIs(Score, OriginalScore)

    def test_unknown_public_attribute_is_an_attribute_error(self):
        import brainscore_core

        with self.assertRaises(AttributeError):
            getattr(brainscore_core, "unknown_metadata_attribute")
