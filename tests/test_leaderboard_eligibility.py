"""Publication guards use fake entries and never connect to a database or S3."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brainscore_core import Score
from brainscore_core.submission import database, endpoints


@pytest.mark.parametrize('flag', [False, 'false', None, 0])
def test_ineligible_or_invalid_score_cannot_mutate_database_entry(flag):
    score = Score(0.25)
    score.attrs['leaderboard_eligible'] = flag
    entry = SimpleNamespace(score_raw=0.8, score_ceiled=0.9, error=None, comment=None, save=Mock())
    with pytest.raises((ValueError, TypeError), match='leaderboard_eligible'):
        database.update_score(score, entry)
    assert (entry.score_raw, entry.score_ceiled) == (0.8, 0.9)
    entry.save.assert_not_called()


@pytest.mark.parametrize('attrs', [{}, {'leaderboard_eligible': True}])
@pytest.mark.parametrize('ceiled', [False, True])
def test_legacy_and_eligible_scores_keep_raw_and_ceiled_values(attrs, ceiled):
    score = Score(0.8)
    score.attrs.update(attrs)
    if ceiled:
        score.attrs.update(raw=Score(0.4), ceiling=Score(0.5))
    entry = SimpleNamespace(score_raw=None, score_ceiled=None, error=None, comment=None, save=Mock())
    database.update_score(score, entry)
    assert entry.score_raw == (0.4 if ceiled else 0.8)
    assert entry.score_ceiled == (0.8 if ceiled else None)
    entry.save.assert_called_once_with()


def test_ineligible_benchmark_rejected_before_model_or_database_use(monkeypatch):
    benchmark = SimpleNamespace(identifier='local-only', leaderboard_eligible=False)
    plugins = SimpleNamespace(load_benchmark=Mock(return_value=benchmark), load_model=Mock(), score=Mock())
    endpoint = endpoints.RunScoringEndpoint.__new__(endpoints.RunScoringEndpoint)
    endpoint.domain_plugins = plugins
    model_entry = Mock()
    monkeypatch.setattr(endpoints, 'modelentry_from_model', model_entry)
    with pytest.raises(ValueError, match='leaderboard_eligible'):
        endpoint._score_model_on_benchmark('model', 'local-only', None, 'vision', True, None)
    plugins.load_model.assert_not_called()
    plugins.score.assert_not_called()
    model_entry.assert_not_called()


def test_direct_benchmark_registration_cannot_bypass_guard(monkeypatch):
    get_or_create = Mock()
    monkeypatch.setattr(database.BenchmarkType, 'get_or_create', get_or_create)
    with pytest.raises(ValueError, match='leaderboard_eligible'):
        database.benchmarkinstance_from_benchmark(
            SimpleNamespace(identifier='local-only', leaderboard_eligible=False), domain='vision',
        )
    get_or_create.assert_not_called()


def test_direct_upload_cannot_bypass_score_guard():
    endpoint = endpoints.RunScoringEndpoint.__new__(endpoints.RunScoringEndpoint)
    endpoint._is_production = Mock(side_effect=AssertionError('Must reject before upload setup'))
    score = Score(0.2)
    score.attrs['leaderboard_eligible'] = False
    with pytest.raises(ValueError, match='leaderboard_eligible'):
        endpoint._upload_score_to_s3(score, 'model', 'benchmark', 'vision')
    endpoint._is_production.assert_not_called()


@pytest.mark.parametrize('excluded', [False, True])
def test_submission_checks_returned_score_before_upload(monkeypatch, excluded):
    score = Score(0.25)
    if excluded:
        score.attrs['leaderboard_eligible'] = False
    benchmark = SimpleNamespace(identifier='benchmark')
    endpoint = endpoints.RunScoringEndpoint.__new__(endpoints.RunScoringEndpoint)
    endpoint.domain_plugins = SimpleNamespace(
        load_benchmark=Mock(return_value=benchmark),
        load_model=Mock(return_value=SimpleNamespace()), score=Mock(return_value=score),
    )
    endpoint._upload_score_to_s3 = Mock()
    entry = SimpleNamespace(score_raw=None, score_ceiled=None, error=None, comment=None, save=Mock())
    monkeypatch.setattr(endpoints, 'modelentry_from_model', Mock())
    monkeypatch.setattr(endpoints, 'benchmarkinstance_from_benchmark', Mock())
    monkeypatch.setattr(endpoints.database_models.Score, 'get_or_create', Mock(return_value=(entry, True)))
    if excluded:
        with pytest.raises(ValueError, match='leaderboard_eligible'):
            endpoint._score_model_on_benchmark('model', 'benchmark', None, 'vision', True, None)
        assert entry.score_raw is entry.score_ceiled is None
        endpoint._upload_score_to_s3.assert_not_called()
    else:
        endpoint._score_model_on_benchmark('model', 'benchmark', None, 'vision', True, None)
        assert entry.score_raw == 0.25
        endpoint._upload_score_to_s3.assert_called_once_with(score, 'model', 'benchmark', 'vision')
