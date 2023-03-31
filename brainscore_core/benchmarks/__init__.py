"""
A :class:`~brainscore.benchmarks.Benchmark` runs an experiment on a :class:`~brainscore.model_interface.BrainModel`
and tests the resulting measurements against primate `data <https://github.com/brain-score/brainio>`_.
This comparison is done by a :class:`~brainscore.metrics.Metric` which outputs a score of how well model and data match.
This score is normalized with data ceilings and the benchmark returns this ceiled score.
"""

from abc import ABC

from brainscore_core.metrics import Score


class Benchmark(ABC):
    """
    Standard Benchmark interface defining the method interfaces.
    """

    def __call__(self, candidate):
        """
        Evaluate a candidate `BrainModel` and return a :class:`~brainscore_core.metrics.Score` denoting the
        alignment of the model to natural intelligence measurements under this benchmark.
        Typically this involves reproducing the experiment on the model and then comparing model measurements
        (e.g. neural/behavioral) against experimental recordings from biological subjects (e.g. primates)
        using a :class:`~brainscore_core.metrics.Metric`.
        The output of this method is a normalized score between 0 and 1 where 0 means the model does not match the
        measurements at all and 1 means the model matches the measurements at ceiling level
        (e.g. if the model obtains a score of 0.8 and the data ceiling is also 0.8,
        the score output by this method should be 1).

        :param candidate: a candidate model implementing the `BrainModel` interface. Benchmarks are agnostic of the
                exact implementation and only interact with models through the methods defined in the interface.
        :return: a :class:`~brainscore_core.metrics.Score` of how aligned to natural intelligence the candidate model is under this
                benchmark. The score is normalized by this benchmark's ceiling such that 1 means the model matches the
                data to ceiling level.
        """
        raise NotImplementedError()

    @property
    def bibtex(self) -> str:
        """
        bibtex string to build the reference.
        Should include an `url` to build a proper link.
        """
        raise NotImplementedError()

    @property
    def identifier(self) -> str:
        """
        Unique identifier for this benchmark.
        Standard format is `<data identifier>-<metric identifier>`, e.g. `dicarlo.Rajalingham2018-i2n`.

        :return: a unique identifier for this benchmark
        """
        raise NotImplementedError()

    @property
    def parent(self) -> str:
        """
        The identifier for the parent of this benchmark. Typically this is one of `behavioral`, `neural`, `engineering`.

        For benchmarks composed of sub-benchmarks, the sub-benchmark's parent can also be an aggregate benchmark
        identifier; for instance the sub-benchmarks 'Geirhos2021colour-error_consistency' and
        'Geirhos2021contrast-error_consistency' might have 'Geirhos2021-error_consistency' as parent.
        """
        raise NotImplementedError()

    @property
    def version(self):
        """
        :return: a version number that is increased every time the model scores for this benchmark change
                (but not for code changes that do not change scores).
        """
        raise NotImplementedError()

    @property
    def ceiling(self) -> Score:
        """
        The ceiling of this benchmark. Scores need to be normalized by this value.
        Typically this represents the signal in the data and how well we expect the best possible model to score.

        :return: a Score object, denoting the ceiling of this benchmark.
                Typically has two values indexed by an `aggregation` coordinate:
                `center` for the averaged ceiling value, and `error` for the uncertainty.
        """
        raise NotImplementedError()


class BenchmarkBase(Benchmark):
    """
    Helper class for implementing standard functions of the :class:`~brainscore_core.benchmarks.Benchmark` interface.
    """

    def __init__(self, identifier: str, ceiling: Score, version, parent: str, bibtex: str = None):
        self._identifier = identifier
        self._ceiling = ceiling
        self._version = version
        self._parent = parent
        self._bibtex = bibtex

    @property
    def bibtex(self):
        return self._bibtex

    @property
    def identifier(self):
        return self._identifier

    @property
    def parent(self):
        return self._parent

    @property
    def version(self):
        return self._version

    @property
    def ceiling(self):
        return self._ceiling


def ceil_score(score, ceiling):
    ceiled_center = score.sel(aggregation='center').values / ceiling.sel(aggregation='center').values
    ceiled_score = type(score)([ceiled_center, score.sel(aggregation='error').values],
                               coords=score.coords, dims=score.dims)
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    ceiled_score.attrs['ceiling'] = ceiling
    return ceiled_score
