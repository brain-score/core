import fire

from brainscore_core.metrics import Score
from brainscore_core.plugin_management.conda_score import wrap_score


def _run_score(model_identifier, benchmark_identifier):
    assert model_identifier == 'dummy-model'
    assert benchmark_identifier == 'dummy-benchmark'
    return Score(.42)


def score(model_identifier, benchmark_identifier, conda_active=False):
    result = wrap_score(__file__,
                        model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                        score_function=_run_score, conda_active=conda_active)
    print(result)


if __name__ == '__main__':
    fire.Fire()
